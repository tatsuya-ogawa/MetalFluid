#include <metal_stdlib>
#include "../MPMTypes.h"
#include "ComputeShaders.h"
#include "SDFCollision.h"
using namespace metal;



// MARK: - Kernel Functions for Projection-Based Collision Processing

// Kernel for particle-SDF collision processing using projection-based dynamics
kernel void processParticleSDFCollisions(
    device MPMParticle* particles [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    constant CollisionUniforms& collision [[buffer(2)]],
    texture3d<float> sdfTexture [[texture(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    
    // Use projection-based constraint solver
    projectConstraints(particles, uniforms, sdfTexture, collision, id, 0.9);
}

// Multi-iteration projection-based particle collision solver
kernel void solveParticleConstraintsIterative(
    device MPMParticle* particles [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    constant CollisionUniforms& collision [[buffer(2)]],
    texture3d<float> sdfTexture [[texture(0)]],
    constant uint& iterationCount [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    
    // Multiple constraint projection iterations for better convergence
    float stiffness = 0.8; // Slightly relaxed for stability
    for (uint iter = 0; iter < iterationCount; iter++) {
        projectConstraints(particles, uniforms, sdfTexture, collision, id, stiffness);
        // Reduce stiffness slightly each iteration for convergence
        stiffness *= 0.95;
    }
}

// Kernel for rigid body-SDF collision processing using projection-based dynamics
kernel void processRigidBodySDFCollisions(
    device RigidBodyState* rigidBodies [[buffer(0)]],
    device MPMParticle* particles [[buffer(1)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(2)]],
    constant ComputeShaderUniforms& uniforms [[buffer(3)]],
    constant CollisionUniforms& collision [[buffer(4)]],
    texture3d<float> sdfTexture [[texture(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.rigidBodyCount) return;
    
    // Use projection-based constraint solver for rigid bodies
    projectRigidBodyConstraints(rigidBodies, particles, rigidInfo, uniforms, 
                               sdfTexture, collision, id, 0.8);
}

// Multi-iteration rigid body constraint solver
kernel void solveRigidBodyConstraintsIterative(
    device RigidBodyState* rigidBodies [[buffer(0)]],
    device MPMParticle* particles [[buffer(1)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(2)]],
    constant ComputeShaderUniforms& uniforms [[buffer(3)]],
    constant CollisionUniforms& collision [[buffer(4)]],
    texture3d<float> sdfTexture [[texture(0)]],
    constant uint& iterationCount [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.rigidBodyCount) return;
    
    // Multiple iterations for stable constraint solving
    for (uint iter = 0; iter < iterationCount; iter++) {
        float stiffness = 0.9 - float(iter) * 0.1; // Gradually reduce stiffness
        projectRigidBodyConstraints(rigidBodies, particles, rigidInfo, uniforms, 
                                   sdfTexture, collision, id, stiffness);
    }
}

// Advanced rigid body to rigid body collision using dual SDFs
kernel void solveRigidBodyToRigidBodyCollisions(
    device RigidBodyState* rigidBodies [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    constant CollisionUniforms& collisionA [[buffer(2)]],
    constant CollisionUniforms& collisionB [[buffer(3)]],
    texture3d<float> sdfTextureA [[texture(0)]],
    texture3d<float> sdfTextureB [[texture(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Map thread ID to rigid body pair
    uint n = uniforms.rigidBodyCount;
    if (n < 2) return;
    
    uint totalPairs = n * (n - 1) / 2;
    if (id >= totalPairs) return;
    
    // Decode pair indices from thread ID
    uint i = uint(floor((sqrt(8.0 * float(id) + 1.0) - 1.0) * 0.5));
    uint firstIndexOfRow = i * (n - 1) - (i * (i - 1)) / 2;
    uint j = id - firstIndexOfRow + i + 1;
    
    if (i >= n || j >= n || i == j) return;
    
    device RigidBodyState& rigidBodyA = rigidBodies[i];
    device RigidBodyState& rigidBodyB = rigidBodies[j];
    
    if (!rigidBodyA.isActive || !rigidBodyB.isActive) return;
    
    // Compute collision impulses using dual SDFs
    float3 impulseA, impulseB, angularImpulseA, angularImpulseB;
    computeRigidBodyToRigidBodySDFCollision(
        rigidBodyA, rigidBodyB, sdfTextureA, sdfTextureB,
        collisionA, collisionB,
        impulseA, impulseB, angularImpulseA, angularImpulseB,
        uniforms.deltaTime
    );
    
    // Apply impulses
    if (length(impulseA) > 1e-6 || length(impulseB) > 1e-6) {
        // Transform inertia tensors to world space
        float3x3 RA = quatToMatrix(rigidBodyA.orientation);
        float3x3 RB = quatToMatrix(rigidBodyB.orientation);
        float3x3 worldInvInertiaTensorA = RA * rigidBodyA.invInertiaTensor * transpose(RA);
        float3x3 worldInvInertiaTensorB = RB * rigidBodyB.invInertiaTensor * transpose(RB);
        
        // Apply linear impulses
        if (rigidBodyA.totalMass > 0) {
            rigidBodyA.linearVelocity += impulseA / rigidBodyA.totalMass;
        }
        if (rigidBodyB.totalMass > 0) {
            rigidBodyB.linearVelocity += impulseB / rigidBodyB.totalMass;
        }
        
        // Apply angular impulses
        rigidBodyA.angularVelocity += worldInvInertiaTensorA * angularImpulseA;
        rigidBodyB.angularVelocity += worldInvInertiaTensorB * angularImpulseB;
        
        // Clamp velocities for stability
        const float maxLinearVel = 35.0;
        const float maxAngularVel = 12.0;
        
        rigidBodyA.linearVelocity = clamp(rigidBodyA.linearVelocity,
                                         float3(-maxLinearVel), float3(maxLinearVel));
        rigidBodyB.linearVelocity = clamp(rigidBodyB.linearVelocity,
                                         float3(-maxLinearVel), float3(maxLinearVel));
        rigidBodyA.angularVelocity = clamp(rigidBodyA.angularVelocity,
                                          float3(-maxAngularVel), float3(maxAngularVel));
        rigidBodyB.angularVelocity = clamp(rigidBodyB.angularVelocity,
                                          float3(-maxAngularVel), float3(maxAngularVel));
    }
}
