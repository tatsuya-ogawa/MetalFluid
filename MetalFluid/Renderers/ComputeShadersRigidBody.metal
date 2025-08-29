#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
#include "ComputeShaders.h"
using namespace metal;

// MARK: - Rigid Body Material Functions
// Note: P2G and G2P Stage 1 kernels have been removed as they contained only noop implementations

// ==============================================
// RIGID BODY PROJECTION SYSTEM
// ==============================================

// --- Utility functions for rigid body dynamics ---

// Convert quaternion to rotation matrix
inline float3x3 quatToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    
    return float3x3(
        float3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y)),
        float3(2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)),
        float3(2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y))
    );
}

// Normalize quaternion to avoid drift
inline float4 normalizeQuat(float4 q) {
    float len = sqrt(dot(q, q));
    if (len < 1e-8) {
        return float4(0, 0, 0, 1); // Identity quaternion
    }
    return q / len;
}

// Integrate angular velocity into quaternion
inline float4 integrateAngularVelocity(float4 q, float3 omega, float dt) {
    if (length(omega) < 1e-8) return q;
    
    float angle = length(omega) * dt * 0.5;
    float3 axis = normalize(omega);
    
    float4 deltaQ = float4(sin(angle) * axis.x, sin(angle) * axis.y, sin(angle) * axis.z, cos(angle));
    
    // Quaternion multiplication: deltaQ * q
    float4 result;
    result.w = deltaQ.w * q.w - dot(deltaQ.xyz, q.xyz);
    result.xyz = deltaQ.w * q.xyz + q.w * deltaQ.xyz + cross(deltaQ.xyz, q.xyz);
    
    return normalizeQuat(result);
}

// Stage 1: Accumulate forces and torques from particles to rigid bodies using atomic operations
kernel void accumulateRigidBodyForces(
    device const MPMParticle* particles [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    device RigidBodyState* rigidBodies [[buffer(2)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    // noop
}

// Stage 2: Update rigid body dynamics (one thread per rigid body)
kernel void updateRigidBodyDynamics(
    device RigidBodyState* rigidBodies [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.rigidBodyCount) return;
    
    device RigidBodyState& rb = rigidBodies[id];
    if (!rb.isActive) return;
    
    float dt = uniforms.deltaTime;
    
    // Linear dynamics: F = ma → a = F/m (following taichi-mpm approach)
    // Use accumulated force from Stage 2 (includes gravity and MPM forces)
    float3 totalForce = rb.accumulatedForce;
    
    // Apply linear damping (exponential decay)
    rb.linearVelocity *= exp(-rb.linearDamping * dt);
    
    // Integrate acceleration
    float3 linearAcceleration = totalForce / max(rb.totalMass, 1e-6);
    rb.linearVelocity += linearAcceleration * dt;
    
    // Integrate position
    rb.centerOfMass += rb.linearVelocity * dt;
    
    // Angular dynamics: τ = Iα → α = I⁻¹τ (proper rigid body dynamics)
    // Apply angular damping first (exponential decay)
    rb.angularVelocity *= exp(-rb.angularDamping * dt);
    
    // Transform inertia tensor to world space: I_world = R * I_local * R^T
    float3x3 R = quatToMatrix(rb.orientation);
    float3x3 worldInvInertiaTensor = R * rb.invInertiaTensor * transpose(R);
    
    // Integrate angular acceleration using world-space inverse inertia tensor
    float3 angularAcceleration = worldInvInertiaTensor * rb.accumulatedTorque;
    
    // Limit angular acceleration to prevent runaway rotation
    const float max_angular_accel = 50.0;
    angularAcceleration = clamp(angularAcceleration,
                               float3(-max_angular_accel),
                               float3(max_angular_accel));
    
    rb.angularVelocity += angularAcceleration * dt;
    
    // Limit angular velocity to prevent excessive rotation
    const float max_angular_vel = 20.0;
    rb.angularVelocity = clamp(rb.angularVelocity,
                              float3(-max_angular_vel),
                              float3(max_angular_vel));
    
    // Update orientation using angular velocity
    rb.orientation = integrateAngularVelocity(rb.orientation, rb.angularVelocity, dt);
    
    // Clear accumulated forces and torques for next frame
    rb.accumulatedForce = float3(0, 0, 0);
    rb.accumulatedTorque = float3(0, 0, 0);
}

// Stage 3: Project particles to maintain rigid body constraints
kernel void projectRigidBodyParticles(
    device MPMParticle* particles [[buffer(0)]],
    device const RigidBodyState* rigidBodies [[buffer(1)]],
    constant ComputeShaderUniforms& uniforms [[buffer(2)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    //noop
}

// Initialize rigid body state (called once during setup)
kernel void initializeRigidBodies(
    device RigidBodyState* rigidBodies [[buffer(0)]],
    device const MPMParticle* particles [[buffer(1)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(3)]],
    constant ComputeShaderUniforms& uniforms [[buffer(2)]],
    uint rigidBodyId [[thread_position_in_grid]]
) {
    if (rigidBodyId >= uniforms.rigidBodyCount) return;
    
    device RigidBodyState& rb = rigidBodies[rigidBodyId];
    uint actualRigidId = rigidBodyId + 1; // Convert to 1-based ID
    
    // Calculate center of mass and total mass
    float3 centerOfMass = float3(0, 0, 0);
    float totalMass = 0.0;
    uint particleCount = 0;
    
    for (uint i = 0; i < uniforms.particleCount; ++i) {
        uint origIdx = particles[i].originalIndex;
        if (rigidInfo[origIdx].rigidId == actualRigidId) {
            centerOfMass += particles[i].position * particles[i].mass;
            totalMass += particles[i].mass;
            particleCount++;
        }
    }
    
    if (totalMass > 0.0) {
        centerOfMass /= totalMass;
    }
    
    // Initialize rigid body state (following taichi-mpm approach)
    rb.centerOfMass = centerOfMass;
    rb.linearVelocity = float3(0, 0, 0);
    rb.angularVelocity = float3(0, 0, 0);
    rb.orientation = float4(0, 0, 0, 1); // Identity quaternion
    rb.totalMass = totalMass;
    rb.particleCount = particleCount;
    rb.isActive = (particleCount > 0) ? 1 : 0;
    rb.accumulatedForce = float3(0, 0, 0);
    rb.accumulatedTorque = float3(0, 0, 0);
    
    // Set physical parameters similar to taichi-mpm
    rb.linearDamping = 0.1;    // Moderate linear damping
    rb.angularDamping = 0.2;   // More angular damping to prevent excessive rotation
    rb.restitution = 0.3;      // Some bounciness
    rb.friction = 0.5;         // Moderate friction
    
    // Calculate inertia tensor based on particle distribution (taichi-mpm style)
    // Calculate the extent of the rigid body
    float3 minPos = float3(1e6), maxPos = float3(-1e6);
    for (uint i = 0; i < uniforms.particleCount; ++i) {
        uint origIdx = particles[i].originalIndex;
        if (rigidInfo[origIdx].rigidId == actualRigidId) {
            minPos = min(minPos, particles[i].position);
            maxPos = max(maxPos, particles[i].position);
        }
    }
    
    float3 extent = maxPos - minPos;
    float avgSize = (extent.x + extent.y + extent.z) / 3.0;
    
    // More conservative inertia tensor calculation
    float I = totalMass * avgSize * avgSize / 12.0; // Box inertia tensor approximation
    I = max(I, totalMass * 0.001); // Smaller minimum inertia for more responsive rotation
    
    rb.invInertiaTensor = float3x3(
        float3(1.0/I, 0, 0),
        float3(0, 1.0/I, 0),
        float3(0, 0, 1.0/I)
    );

    // Store half extents and bounding radius for broad-phase collision
    rb.halfExtents = extent * 0.5;
    rb.boundingRadius = length(rb.halfExtents);
}

// Broad-phase sphere test then narrow-phase AABB impulse resolution between rigid bodies
kernel void solveRigidBodyCollisions(
    device RigidBodyState* rigidBodies [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Pair index mapping: id enumerates upper triangle (i<j)
    uint n = uniforms.rigidBodyCount;
    if (n < 2) return;
    // Compute i,j from linear id: id in [0, n*(n-1)/2)
    uint totalPairs = n * (n - 1) / 2;
    if (id >= totalPairs) return;
    // Invert triangular number
    uint i = uint(floor((sqrt(8.0f * (float)id + 1.0f) - 1.0f) * 0.5f));
    uint firstIndexOfRow = i * (n - 1) - (i * (i - 1)) / 2;
    uint j = id - firstIndexOfRow + i + 1;
    if (i >= n || j >= n) return;

    device RigidBodyState &A = rigidBodies[i];
    device RigidBodyState &B = rigidBodies[j];
    if (!A.isActive || !B.isActive) return;

    // Broad-phase sphere overlap (cheap test)
    float3 d = B.centerOfMass - A.centerOfMass;
    float dist2 = dot(d,d);
    float rSum = A.boundingRadius + B.boundingRadius;
    if (dist2 > rSum*rSum) return; // no overlap
    float dist = sqrt(max(dist2, 1e-8f));
    float3 normal = (dist > 1e-6f) ? d / dist : float3(0,1,0);

    // Narrow-phase (very approximate) OBB-OBB along broad-phase normal.
    // We build support points along the normal using oriented half extents (like GJK support but simplified)
    float3x3 RA = quatToMatrix(A.orientation);
    float3x3 RB = quatToMatrix(B.orientation);
    // Columns of R are world axes of the box
    float3 Ax = RA[0]; float3 Ay = RA[1]; float3 Az = RA[2];
    float3 Bx = RB[0]; float3 By = RB[1]; float3 Bz = RB[2];

    // Support point for A in direction of normal
    float3 supA = A.centerOfMass
        + Ax * (sign(dot(Ax, normal)) * A.halfExtents.x)
        + Ay * (sign(dot(Ay, normal)) * A.halfExtents.y)
        + Az * (sign(dot(Az, normal)) * A.halfExtents.z);
    // Support point for B in direction opposite of normal
    float3 supB = B.centerOfMass
        - (Bx * (sign(dot(Bx, normal)) * B.halfExtents.x)
        +   By * (sign(dot(By, normal)) * B.halfExtents.y)
        +   Bz * (sign(dot(Bz, normal)) * B.halfExtents.z));

    float3 separation = supB - supA; // along normal ideally
    float penetration = -dot(separation, normal); // if negative, they overlap along normal
    if (penetration <= 0.0f) {
        // Fallback to sphere penetration if oriented test fails (e.g. parallel axes)
        penetration = rSum - dist;
        if (penetration <= 0) return;
    }

    // Contact point (midpoint of support points)
    float3 contactPoint = 0.5f * (supA + supB);
    float3 rA = contactPoint - A.centerOfMass;
    float3 rB = contactPoint - B.centerOfMass;

    // Relative velocity at contact point (including angular)
    float3 vA = A.linearVelocity + cross(A.angularVelocity, rA);
    float3 vB = B.linearVelocity + cross(B.angularVelocity, rB);
    float3 relV = vB - vA;
    float relVN = dot(relV, normal);

    // Restitution & friction combine (geometric mean like Taichi)
    float restitution = sqrt(A.restitution * B.restitution);
    float friction = sqrt(A.friction * B.friction);

    // Baumgarte positional correction factor (stabilizes penetration)
    float baumgarte = 0.2f; // bias factor
    float invMassA = (A.totalMass > 0) ? 1.0f / A.totalMass : 0.0f;
    float invMassB = (B.totalMass > 0) ? 1.0f / B.totalMass : 0.0f;
    float massSum = invMassA + invMassB;
    if (massSum <= 0.0f) return;

    // Positional correction (split based on inverse masses)
    float3 correction = (baumgarte * penetration / massSum) * normal;
    A.centerOfMass -= correction * invMassA;
    B.centerOfMass += correction * invMassB;

    // Recompute relative velocity at contact point after correction
    vA = A.linearVelocity + cross(A.angularVelocity, rA);
    vB = B.linearVelocity + cross(B.angularVelocity, rB);
    relV = vB - vA;
    relVN = dot(relV, normal);

    // Only resolve if closing
    if (relVN > 0.0f) return;

    // Transform inertia tensors to world space for collision resolution
    float3x3 R_A = quatToMatrix(A.orientation);
    float3x3 R_B = quatToMatrix(B.orientation);
    float3x3 worldInvInertiaTensorA = R_A * A.invInertiaTensor * transpose(R_A);
    float3x3 worldInvInertiaTensorB = R_B * B.invInertiaTensor * transpose(R_B);

    // Compute effective mass along normal including angular terms (using world-space tensors)
    float3 rnA = cross(rA, normal);
    float3 rnB = cross(rB, normal);
    float3 angA = cross( (worldInvInertiaTensorA * rnA), rA );
    float3 angB = cross( (worldInvInertiaTensorB * rnB), rB );
    float angularDenom = dot(normal, angA + angB);
    float denom = massSum + angularDenom;
    if (denom < 1e-6f) return;
    float J = -(1.0f + restitution) * relVN / denom;
    if (J < 0.0f) return;
    float3 impulse = J * normal;

    // Apply linear & angular impulses using world-space inverse inertia tensors
    A.linearVelocity -= impulse * invMassA;
    B.linearVelocity += impulse * invMassB;
    A.angularVelocity -= worldInvInertiaTensorA * cross(rA, impulse);
    B.angularVelocity += worldInvInertiaTensorB * cross(rB, impulse);

    // Friction (Coulomb) using tangent at contact point
    vA = A.linearVelocity + cross(A.angularVelocity, rA);
    vB = B.linearVelocity + cross(B.angularVelocity, rB);
    relV = vB - vA;
    float3 vt = relV - dot(relV, normal) * normal;
    float vtLen = length(vt);
    if (vtLen > 1e-6f) {
        float3 tangent = vt / vtLen;
        float3 rtA = cross(rA, tangent);
        float3 rtB = cross(rB, tangent);
        float3 angTA = cross( (worldInvInertiaTensorA * rtA), rA );
        float3 angTB = cross( (worldInvInertiaTensorB * rtB), rB );
        float denomT = massSum + dot(tangent, angTA + angTB);
        if (denomT > 1e-6f) {
            float jt = -dot(relV, tangent) / denomT;
            float maxFriction = friction * J;
            jt = clamp(jt, -maxFriction, maxFriction);
            float3 fImpulse = jt * tangent;
            A.linearVelocity -= fImpulse * invMassA;
            B.linearVelocity += fImpulse * invMassB;
            A.angularVelocity -= worldInvInertiaTensorA * cross(rA, fImpulse);
            B.angularVelocity += worldInvInertiaTensorB * cross(rB, fImpulse);
        }
    }
}
