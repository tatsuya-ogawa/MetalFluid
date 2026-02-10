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
// Taichi-MPM style collision resolution
// Modifies particle state directly for stability (Hard Constraint)
// Returns the impulse applied to the particle (for rigid body coupling)
inline float3 resolveParticleSDFCollision(
    device MPMParticle& p,
    texture3d<float> sdfTexture,
    constant ComputeShaderUniforms& uniforms,
    constant CollisionUniforms &collision,
    float dt
) {
    float4 sdfData = sampleSDFWithGradient(p.position, sdfTexture, uniforms, collision);
    float phi = sdfData.x;
    float3 normal = sdfData.yzw;
    
    // Guard against invalid SDF samples to prevent NaN propagation into particle state.
    if (!isfinite(phi) || !all(isfinite(normal))) {
        return float3(0.0);
    }
    
    // No collision if particle is outside SDF
    if (phi > 0.0) {
        return float3(0.0);
    }
    
    float nLen = length(normal);
    if (!isfinite(nLen) || nLen < 1e-6) {
        return float3(0.0);
    }
    normal /= nLen;
    
    // 1. Position Projection (Hard Constraint)
    // Move particle out of the SDF along the normal
    // We only project if phi is negative (inside)
    float3 projectedPos = p.position - phi * normal;
    if (!all(isfinite(projectedPos))) {
        return float3(0.0);
    }
    p.position = projectedPos;
    
    // 2. Velocity Constraint
    // Calculate relative velocity
    float3 colliderVel = float3(0.0); // Assuming static or handled via physics state
    // Note: If we had collider velocity, we'd subtract it here:
    // float3 relVel = p.velocity - colliderVel;
    float3 relVel = p.velocity;
    
    float v_n = dot(relVel, normal);
    
    // If separating, no velocity correction needed (unless sticky)
    if (v_n >= 0.0) {
        return float3(0.0);
    }
    
    float3 oldVelocity = p.velocity;
    
    // Apply normal constraint (inelastic collision)
    // Remove normal component of velocity
    float3 normalVel = v_n * normal;
    relVel -= normalVel; // Now relVel is purely tangential
    
    // Apply friction to tangential component
    const float friction = 0.3;
    float v_t_len = length(relVel);
    
    if (v_t_len > 1e-6) {
        // Coulomb friction: max tangential impulse <= mu * |normal_impulse|
        // Here we operate on velocities directly.
        // The "normal impulse" equivalent in velocity space is |v_n|.
        // We want to reduce tangential velocity by at most friction * |v_n|.
        float frictionDeltaV = friction * abs(v_n);
        float new_v_t_len = max(0.0, v_t_len - frictionDeltaV);
        relVel *= (new_v_t_len / v_t_len);
    }
    
    // Update particle velocity
    p.velocity = relVel; // + colliderVel if moving
    
    // Calculate impulse J = m * delta_v
    // This is the impulse applied TO the particle
    float3 deltaV = p.velocity - oldVelocity;
    return p.mass * deltaV;
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
    
    float4 sdfData = sampleSDFWithGradient(particle.position, sdfTexture, uniforms,collision);
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
