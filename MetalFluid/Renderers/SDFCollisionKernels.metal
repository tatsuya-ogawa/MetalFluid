#include <metal_stdlib>
#include "../MPMTypes.h"
#include "ComputeShaders.h"
#include "SDFCollision.h"
using namespace metal;

// MARK: - SDF Rigid Body Physics Integration Kernel

// GPU kernel to integrate SDF collision impulses and update collision transform
// This replaces the CPU-based applySDFImpulseAggregationToCollisionTransform function
kernel void applySdfImpulseToTransform(
    device CollisionUniforms* collisionUniforms [[buffer(0)]],  // Collision uniforms to update
    device SDFPhysicsState* physicsStates [[buffer(1)]],        // SDF physics state with accumulated impulses
    constant ComputeShaderUniforms& uniforms [[buffer(2)]],     // Simulation uniforms
    constant uint& sdfIndex [[buffer(3)]],                      // Which SDF to process (0 for primary)
    constant bool& enableGravity [[buffer(4)]],                 // Enable gravity for this SDF
    constant bool& isDynamic [[buffer(5)]],                     // Is this SDF dynamic (can move)
    uint tid [[thread_position_in_grid]]
) {
    // Only process the first thread for single SDF update
    if (tid != 0) return;
    
    // Skip if not dynamic
    if (!isDynamic) {
        // Clear impulses for static objects
        atomic_store_explicit(&physicsStates[sdfIndex].impulse_x, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&physicsStates[sdfIndex].impulse_y, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&physicsStates[sdfIndex].impulse_z, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&physicsStates[sdfIndex].torque_x, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&physicsStates[sdfIndex].torque_y, 0.0f, memory_order_relaxed);
        atomic_store_explicit(&physicsStates[sdfIndex].torque_z, 0.0f, memory_order_relaxed);
        return;
    }
    
    device CollisionUniforms& collision = collisionUniforms[sdfIndex];
    device SDFPhysicsState& physics = physicsStates[sdfIndex];
    
    // Read accumulated impulses
    float3 J = float3(
        atomic_load_explicit(&physics.impulse_x, memory_order_relaxed),
        atomic_load_explicit(&physics.impulse_y, memory_order_relaxed),
        atomic_load_explicit(&physics.impulse_z, memory_order_relaxed)
    );
    float3 Tau = float3(
        atomic_load_explicit(&physics.torque_x, memory_order_relaxed),
        atomic_load_explicit(&physics.torque_y, memory_order_relaxed),
        atomic_load_explicit(&physics.torque_z, memory_order_relaxed)
    );
    
    // Early exit if no significant forces
    if (length(J) < 1e-6 && length(Tau) < 1e-6 && !enableGravity) {
        return;
    }
    
    // Physical parameters
    float dt = uniforms.deltaTime;
    float mass = max(1e-4, collision.sdfMass);
    
    // Compute inertia tensor from SDF size (approximated as solid box)
    float3 size = collision.sdfSize;
    float hx = max(1e-4, size.x * 0.5);
    float hy = max(1e-4, size.y * 0.5);
    float hz = max(1e-4, size.z * 0.5);
    
    float3 I = float3(
        (mass / 12.0) * (hy*hy + hz*hz),
        (mass / 12.0) * (hx*hx + hz*hz),
        (mass / 12.0) * (hx*hx + hy*hy)
    );
    
    // Read current velocities
    float3 linearVel = physics.linearVelocity;
    float3 angularVel = physics.angularVelocity;
    
    // Apply gravity if enabled
    if (enableGravity) {
        linearVel.y += uniforms.gravity * dt;
    }
    
    // Apply damping
    float linearDamping = 0.2;
    float angularDamping = 0.1;
    linearVel *= exp(-linearDamping * dt);
    angularVel *= exp(-angularDamping * dt);
    
    // Apply impulses
    linearVel += J / mass;
    angularVel += float3(
        Tau.x / max(I.x, 1e-6),
        Tau.y / max(I.y, 1e-6),
        Tau.z / max(I.z, 1e-6)
    );
    
    // Clamp velocities for stability
    float maxLin = 5.0;
    float maxAng = 5.0;
    linearVel = clamp(linearVel, -float3(maxLin), float3(maxLin));
    angularVel = clamp(angularVel, -float3(maxAng), float3(maxAng));
    
    // Build transformation matrices
    float4x4 dTrans = float4x4(
        float4(1, 0, 0, 0),
        float4(0, 1, 0, 0),
        float4(0, 0, 1, 0),
        float4(linearVel * dt, 1)
    );
    
    // Angular rotation matrix from axis-angle
    float3 omegaDt = angularVel * dt;
    float angle = length(omegaDt);
    float4x4 dRot = float4x4(
        float4(1, 0, 0, 0),
        float4(0, 1, 0, 0),
        float4(0, 0, 1, 0),
        float4(0, 0, 0, 1)
    );
    
    if (angle > 1e-6) {
        float3 axis = normalize(omegaDt);
        float c = cos(angle);
        float s = sin(angle);
        float t = 1.0 - c;
        float x = axis.x, y = axis.y, z = axis.z;
        
        float4x4 rot = float4x4(
            float4(t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,  0),
            float4(t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,  0),
            float4(t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,    0),
            float4(0,            0,            0,            1)
        );
        dRot = rot;
    }
    
    // Compute world-space center of mass
    float3 comLocal = collision.sdfOrigin + 0.5 * collision.sdfSize;
    float4 comWorld4 = collision.collisionTransform * float4(comLocal, 1.0);
    float3 comWorld = comWorld4.xyz;
    
    // Translation matrices for rotation about COM
    float4x4 Tcom = float4x4(
        float4(1, 0, 0, 0),
        float4(0, 1, 0, 0),
        float4(0, 0, 1, 0),
        float4(comWorld, 1)
    );
    float4x4 TcomInv = float4x4(
        float4(1, 0, 0, 0),
        float4(0, 1, 0, 0),
        float4(0, 0, 1, 0),
        float4(-comWorld, 1)
    );
    
    // Apply transformation: translate, then rotate about COM
    float4x4 newTransform = dTrans * (Tcom * dRot * TcomInv) * collision.collisionTransform;
    
    // Boundary collision handling
    // Compute world AABB of transformed SDF using |M| * h method
    float3 h = 0.5 * collision.sdfSize;
    float3 centerLocal = collision.sdfOrigin + h;
    float4 centerWorld4 = newTransform * float4(centerLocal, 1.0);
    float3 centerWorld = centerWorld4.xyz;
    
    // Extract upper-left 3x3 rotation-scale matrix
    float3x3 m = float3x3(
        newTransform.columns[0].xyz,
        newTransform.columns[1].xyz,
        newTransform.columns[2].xyz
    );
    float3x3 absM = float3x3(
        abs(m.columns[0]),
        abs(m.columns[1]),
        abs(m.columns[2])
    );
    float3 extents = absM * h;
    float3 worldMin = centerWorld - extents;
    float3 worldMax = centerWorld + extents;
    
    // Boundary collision correction
    float wallThickness = uniforms.gridSpacing;
    float3 innerMin = uniforms.boundaryMin + float3(wallThickness);
    float3 innerMax = uniforms.boundaryMax - float3(wallThickness);
    
    float3 correction = float3(0.0);
    float3 hitNormal = float3(0.0);
    
    if (worldMin.x < innerMin.x) { correction.x += (innerMin.x - worldMin.x); hitNormal.x += 1.0; }
    if (worldMax.x > innerMax.x) { correction.x -= (worldMax.x - innerMax.x); hitNormal.x -= 1.0; }
    if (worldMin.y < innerMin.y) { correction.y += (innerMin.y - worldMin.y); hitNormal.y += 1.0; }
    if (worldMax.y > innerMax.y) { correction.y -= (worldMax.y - innerMax.y); hitNormal.y -= 1.0; }
    if (worldMin.z < innerMin.z) { correction.z += (innerMin.z - worldMin.z); hitNormal.z += 1.0; }
    if (worldMax.z > innerMax.z) { correction.z -= (worldMax.z - innerMax.z); hitNormal.z -= 1.0; }
    
    if (any(correction != 0.0)) {
        // Apply position correction
        newTransform.columns[3].xyz += correction;
        
        // Apply velocity reflection with restitution
        float restitution = 0.2;
        if (hitNormal.x != 0.0) linearVel.x = -linearVel.x * restitution;
        if (hitNormal.y != 0.0) linearVel.y = -linearVel.y * restitution;
        if (hitNormal.z != 0.0) linearVel.z = -linearVel.z * restitution;
        
        // Damp angular velocity on collision
        angularVel *= 0.8;
    }
    
    // Update collision uniforms
    collision.collisionTransform = newTransform;
    collision.collisionInvTransform = transpose(newTransform); // Approximate inverse for rigid transforms
    
    // Update physics state
    physics.linearVelocity = linearVel;
    physics.angularVelocity = angularVel;
    
    // Clear accumulated impulses
    atomic_store_explicit(&physics.impulse_x, 0.0f, memory_order_relaxed);
    atomic_store_explicit(&physics.impulse_y, 0.0f, memory_order_relaxed);
    atomic_store_explicit(&physics.impulse_z, 0.0f, memory_order_relaxed);
    atomic_store_explicit(&physics.torque_x, 0.0f, memory_order_relaxed);
    atomic_store_explicit(&physics.torque_y, 0.0f, memory_order_relaxed);
    atomic_store_explicit(&physics.torque_z, 0.0f, memory_order_relaxed);
}