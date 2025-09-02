//
//  SDFCollision.h
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/08/29.
//
#pragma once
#include <metal_stdlib>
#include "../MPMTypes.h"
#include "ComputeShaders.h"

// MARK: - Quaternion Utility Functions

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

constant bool USE_DIRECT_POSITION_FIX = false;

// MARK: - Collision Flag Helper Functions
// Bit flags for collisionFlags field
constant uint COLLISION_ENABLE_BIT = 1u << 0;  // bit 0: enable collision
constant uint COLLISION_STATIC_BIT = 1u << 1;  // bit 1: is static SDF

inline bool isCollisionEnabled(constant CollisionUniforms& collision) {
    return (collision.collisionFlags & COLLISION_ENABLE_BIT) != 0;
}

inline bool isStaticSDF(constant CollisionUniforms& collision) {
    return (collision.collisionFlags & COLLISION_STATIC_BIT) != 0;
}

inline bool needsPhysicsAccumulation(constant CollisionUniforms& collision) {
    return isCollisionEnabled(collision) && !isStaticSDF(collision);
}

// MARK: - SDF Collision Impulse Functions (Taichi-MPM Style)
// Taichi-MPM style collision impulse for particle-SDF collision
inline float3 computeParticleSDFCollisionImpulse(
    float3 particlePos,
    float3 particleVel,
    float particleMass,
    texture3d<float> sdfTexture,
    constant CollisionUniforms &collision,
    float dt
) {
    float4 sdfData = sampleSDFWithGradient(particlePos, sdfTexture, collision);
    float phi = sdfData.x;
    float3 normal = sdfData.yzw;
    
    // No collision if particle is outside SDF
    if (phi > 0.0) {
        return float3(0.0);
    }
    
    // Taichi-MPM collision response parameters
    const float restitution = 0.4;  // Coefficient of restitution
    const float friction = 0.3;     // Coefficient of friction
    
    // Velocity relative to surface (assuming static surface)
    float3 relativeVel = particleVel;
    
    // Normal and tangential velocity components
    float vn = dot(relativeVel, normal);
    float3 vt = relativeVel - vn * normal;
    
    // Only process if particle is moving into the surface
    if (vn >= 0.0) {
        return float3(0.0);
    }
    
    // Normal impulse (prevent penetration + restitution)
    float normalImpulseMagnitude = -(1.0 + restitution) * vn * particleMass;
    
    // Tangential impulse (friction)
    float vtMagnitude = length(vt);
    float3 tangentialImpulse = float3(0.0);
    
    if (vtMagnitude > 1e-6) {
        float3 tangentDirection = vt / vtMagnitude;
        
        // Coulomb friction model
        float maxFrictionImpulse = friction * normalImpulseMagnitude;
        float tangentialImpulseMagnitude = min(maxFrictionImpulse, vtMagnitude * particleMass);
        
        tangentialImpulse = -tangentialImpulseMagnitude * tangentDirection;
    }
    if(USE_DIRECT_POSITION_FIX){
        particlePos += normal * -phi;
        float3 normalImpulse = normalImpulseMagnitude * normal;
        return normalImpulse + tangentialImpulse;
    }else{
        // Position correction to prevent penetration (Taichi-MPM approach)
        float positionCorrection = -phi; // Move particle out of SDF
        float3 positionImpulse = positionCorrection * normal * particleMass / dt;
        
        // Total impulse
        float3 normalImpulse = normalImpulseMagnitude * normal;
        return normalImpulse + tangentialImpulse + positionImpulse;
    }
}

// MARK: - Projection-Based Dynamics Functions

// Advanced projection-based collision constraint solver
inline void projectConstraints(
    device MPMParticle* particles,
    constant ComputeShaderUniforms& uniforms,
    texture3d<float> sdfTexture,
    constant CollisionUniforms& collision,
    uint particleId,
    float constraintStiffness = 1.0
) {
    device MPMParticle& particle = particles[particleId];
    
    float4 sdfData = sampleSDFWithGradient(particle.position, sdfTexture, collision);
    float phi = sdfData.x;
    float3 normal = sdfData.yzw;
    
    // No constraint violation if particle is outside SDF
    if (phi >= 0.0) {
        return;
    }
    
    // Position-based constraint projection
    float constraintValue = phi; // C(x) = phi(x) < 0 means violation
    float3 constraintGradient = normal; // ∇C(x) = normal
    
    // Constraint correction using position-based dynamics
    float gradientMagnitude = dot(constraintGradient, constraintGradient);
    if (gradientMagnitude > 1e-6) {
        // Compute Lagrange multiplier: λ = -C(x) / (∇C·∇C + compliance)
        float compliance = 1e-6; // Small compliance for numerical stability
        float lambda = -constraintValue / (gradientMagnitude + compliance);
        
        // Apply constraint correction with stiffness
        float3 correction = constraintStiffness * lambda * constraintGradient;
        
        // Project position to satisfy constraint
        particle.position += correction;
        
        // Velocity correction for realistic response
        float3 velocityCorrection = correction / uniforms.deltaTime;
        particle.velocity += velocityCorrection;
    }
}
