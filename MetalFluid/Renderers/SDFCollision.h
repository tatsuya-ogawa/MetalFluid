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
    
    // Position correction to prevent penetration (Taichi-MPM approach)
    float positionCorrection = -phi; // Move particle out of SDF
    float3 positionImpulse = positionCorrection * normal * particleMass / dt;
    
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
    
    // Total impulse
    float3 normalImpulse = normalImpulseMagnitude * normal;
    return normalImpulse + tangentialImpulse + positionImpulse;
}

// Taichi-MPM style rigid body to SDF collision impulse
inline float3 computeRigidBodySDFCollisionImpulse(
    float3 contactPoint,
    constant RigidBodyState &rigidBody,
    texture3d<float> sdfTexture,
    constant CollisionUniforms &collision,
    float dt
) {
    float4 sdfData = sampleSDFWithGradient(contactPoint, sdfTexture, collision);
    float phi = sdfData.x;
    float3 normal = sdfData.yzw;
    
    // No collision if contact point is outside SDF
    if (phi > 0.0) {
        return float3(0.0);
    }
    
    // Collision response parameters
    const float restitution = 0.5;  // Coefficient of restitution for rigid bodies
    const float friction = 0.4;     // Coefficient of friction
    
    // Contact point relative to center of mass
    float3 r = contactPoint - rigidBody.centerOfMass;
    
    // Velocity at contact point (linear + angular)
    float3 contactVelocity = rigidBody.linearVelocity + cross(rigidBody.angularVelocity, r);
    
    // Normal and tangential velocity components
    float vn = dot(contactVelocity, normal);
    float3 vt = contactVelocity - vn * normal;
    
    // Only process if moving into the surface
    if (vn >= 0.0) {
        return float3(0.0);
    }
    
    // Transform inertia tensor to world space
    float3x3 R = quatToMatrix(rigidBody.orientation);
    float3x3 worldInvInertiaTensor = R * rigidBody.invInertiaTensor * transpose(R);
    
    // Effective mass calculation for collision response
    float invMass = (rigidBody.totalMass > 0) ? 1.0 / rigidBody.totalMass : 0.0;
    float3 rCrossN = cross(r, normal);
    float3 temp = worldInvInertiaTensor * rCrossN;
    float3 rCrossNInvI = cross(temp, r);
    float effectiveInvMass = invMass + dot(normal, rCrossNInvI);
    
    if (effectiveInvMass < 1e-6) {
        return float3(0.0);
    }
    
    // Normal impulse magnitude
    float penetrationCorrection = -phi / dt; // Position correction term
    float normalImpulseMagnitude = -(1.0 + restitution) * vn + penetrationCorrection;
    normalImpulseMagnitude /= effectiveInvMass;
    
    // Tangential impulse (friction)
    float3 tangentialImpulse = float3(0.0);
    float vtMagnitude = length(vt);
    
    if (vtMagnitude > 1e-6) {
        float3 tangentDirection = vt / vtMagnitude;
        
        // Effective mass for tangential direction
        float3 rCrossT = cross(r, tangentDirection);
        float3 tempT = worldInvInertiaTensor * rCrossT;
        float3 rCrossTInvI = cross(tempT, r);
        float effectiveInvMassT = invMass + dot(tangentDirection, rCrossTInvI);
        
        if (effectiveInvMassT > 1e-6) {
            float tangentialImpulseMagnitude = -vtMagnitude / effectiveInvMassT;
            
            // Apply Coulomb friction limit
            float maxFriction = friction * abs(normalImpulseMagnitude);
            tangentialImpulseMagnitude = clamp(tangentialImpulseMagnitude, -maxFriction, maxFriction);
            
            tangentialImpulse = tangentialImpulseMagnitude * tangentDirection;
        }
    }
    
    // Total linear impulse
    float3 normalImpulse = normalImpulseMagnitude * normal;
    return normalImpulse + tangentialImpulse;
}

// Rigid body to rigid body SDF collision impulse (using both SDFs)
inline void computeRigidBodyToRigidBodySDFCollision(
    device RigidBodyState &rigidBodyA,
    device RigidBodyState &rigidBodyB,
    texture3d<float> sdfTextureA,
    texture3d<float> sdfTextureB,
    constant CollisionUniforms &collisionA,
    constant CollisionUniforms &collisionB,
    thread float3 &impulseA,
    thread float3 &impulseB,
    thread float3 &angularImpulseA,
    thread float3 &angularImpulseB,
    float dt
) {
    // Initialize impulses
    impulseA = float3(0.0);
    impulseB = float3(0.0);
    angularImpulseA = float3(0.0);
    angularImpulseB = float3(0.0);
    
    // Simple approach: check center of mass of A against SDF of B
    float4 sdfDataB = sampleSDFWithGradient(rigidBodyA.centerOfMass, sdfTextureB, collisionB);
    float phiB = sdfDataB.x;
    
    // No collision if A's center is outside B's SDF
    if (phiB > 0.0) {
        return;
    }
    
    float3 normal = sdfDataB.yzw; // Normal points from B towards A
    float3 contactPoint = rigidBodyA.centerOfMass; // Simplified contact point
    
    // Collision parameters
    const float restitution = 0.6;
    const float friction = 0.3;
    
    // Contact point relative to centers of mass
    float3 rA = contactPoint - rigidBodyA.centerOfMass;
    float3 rB = contactPoint - rigidBodyB.centerOfMass;
    
    // Velocities at contact point
    float3 vA = rigidBodyA.linearVelocity + cross(rigidBodyA.angularVelocity, rA);
    float3 vB = rigidBodyB.linearVelocity + cross(rigidBodyB.angularVelocity, rB);
    float3 relativeVelocity = vA - vB;
    
    // Normal and tangential components
    float vn = dot(relativeVelocity, normal);
    float3 vt = relativeVelocity - vn * normal;
    
    // Only process if bodies are approaching
    if (vn >= 0.0) {
        return;
    }
    
    // Transform inertia tensors to world space
    float3x3 RA = quatToMatrix(rigidBodyA.orientation);
    float3x3 RB = quatToMatrix(rigidBodyB.orientation);
    float3x3 worldInvInertiaTensorA = RA * rigidBodyA.invInertiaTensor * transpose(RA);
    float3x3 worldInvInertiaTensorB = RB * rigidBodyB.invInertiaTensor * transpose(RB);
    
    // Effective mass calculation
    float invMassA = (rigidBodyA.totalMass > 0) ? 1.0 / rigidBodyA.totalMass : 0.0;
    float invMassB = (rigidBodyB.totalMass > 0) ? 1.0 / rigidBodyB.totalMass : 0.0;
    
    float3 rACrossN = cross(rA, normal);
    float3 rBCrossN = cross(rB, normal);
    float3 tempA = worldInvInertiaTensorA * rACrossN;
    float3 tempB = worldInvInertiaTensorB * rBCrossN;
    float3 rACrossNInvI = cross(tempA, rA);
    float3 rBCrossNInvI = cross(tempB, rB);
    
    float effectiveInvMass = invMassA + invMassB + dot(normal, rACrossNInvI + rBCrossNInvI);
    
    if (effectiveInvMass < 1e-6) {
        return;
    }
    
    // Normal impulse magnitude
    float penetrationCorrection = -phiB / dt; // Position correction
    float normalImpulseMagnitude = -(1.0 + restitution) * vn + penetrationCorrection;
    normalImpulseMagnitude /= effectiveInvMass;
    
    // Tangential impulse (friction)
    float3 tangentialImpulseVector = float3(0.0);
    float vtMagnitude = length(vt);
    
    if (vtMagnitude > 1e-6) {
        float3 tangentDirection = vt / vtMagnitude;
        
        // Effective mass for tangential direction
        float3 rACrossT = cross(rA, tangentDirection);
        float3 rBCrossT = cross(rB, tangentDirection);
        float3 tempAT = worldInvInertiaTensorA * rACrossT;
        float3 tempBT = worldInvInertiaTensorB * rBCrossT;
        float3 rACrossTInvI = cross(tempAT, rA);
        float3 rBCrossTInvI = cross(tempBT, rB);
        
        float effectiveInvMassT = invMassA + invMassB + dot(tangentDirection, rACrossTInvI + rBCrossTInvI);
        
        if (effectiveInvMassT > 1e-6) {
            float tangentialImpulseMagnitude = -vtMagnitude / effectiveInvMassT;
            
            // Apply Coulomb friction limit
            float maxFriction = friction * abs(normalImpulseMagnitude);
            tangentialImpulseMagnitude = clamp(tangentialImpulseMagnitude, -maxFriction, maxFriction);
            
            tangentialImpulseVector = tangentialImpulseMagnitude * tangentDirection;
        }
    }
    
    // Total impulse
    float3 totalImpulse = normalImpulseMagnitude * normal + tangentialImpulseVector;
    
    // Linear impulses (Newton's third law)
    impulseA = totalImpulse;
    impulseB = -totalImpulse;
    
    // Angular impulses
    angularImpulseA = cross(rA, totalImpulse);
    angularImpulseB = cross(rB, -totalImpulse);
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

// Rigid body constraint projection with SDF
inline void projectRigidBodyConstraints(
    device RigidBodyState* rigidBodies,
    device MPMParticle* particles,
    device const MPMParticleRigidInfo* rigidInfo,
    constant ComputeShaderUniforms& uniforms,
    texture3d<float> sdfTexture,
    constant CollisionUniforms& collision,
    uint rigidBodyId,
    float constraintStiffness = 0.8
) {
    device RigidBodyState& rigidBody = rigidBodies[rigidBodyId];
    if (!rigidBody.isActive) return;
    
    // Multiple contact points for robust collision detection
    const int numContactPoints = 12;
    float3 contactOffsets[12] = {
        // Cube vertices
        float3(-1, -1, -1), float3( 1, -1, -1), float3(-1,  1, -1), float3( 1,  1, -1),
        float3(-1, -1,  1), float3( 1, -1,  1), float3(-1,  1,  1), float3( 1,  1,  1),
        // Face centers
        float3( 0,  0, -1), float3( 0,  0,  1), float3(-1,  0,  0), float3( 1,  0,  0)
    };
    
    float3 totalPositionCorrection = float3(0.0);
    float3 totalAngularCorrection = float3(0.0);
    int activeContacts = 0;
    
    // Check constraints at multiple contact points
    for (int i = 0; i < numContactPoints; i++) {
        float3 contactPoint = rigidBody.centerOfMass +
                             quatToMatrix(rigidBody.orientation) *
                             (contactOffsets[i] * rigidBody.halfExtents);
        
        float4 sdfData = sampleSDFWithGradient(contactPoint, sdfTexture, collision);
        float phi = sdfData.x;
        float3 normal = sdfData.yzw;
        
        // Process constraint violation
        if (phi < 0.0) {
            activeContacts++;
            
            // Position constraint: move rigid body to satisfy SDF constraint
            float constraintValue = phi;
            float3 constraintGradient = normal;
            
            // Rigid body constraint correction
            float gradientMagnitude = dot(constraintGradient, constraintGradient);
            if (gradientMagnitude > 1e-6) {
                float lambda = -constraintValue / (gradientMagnitude + 1e-6);
                float3 linearCorrection = constraintStiffness * lambda * constraintGradient;
                
                // Accumulate linear correction
                totalPositionCorrection += linearCorrection;
                
                // Compute angular correction to maintain rigid body constraints
                float3 r = contactPoint - rigidBody.centerOfMass;
                float3 angularCorrection = cross(r, linearCorrection);
                totalAngularCorrection += angularCorrection;
            }
        }
    }
    
    // Apply averaged corrections
    if (activeContacts > 0) {
        float invContacts = 1.0 / float(activeContacts);
        
        // Apply linear correction
        rigidBody.centerOfMass += totalPositionCorrection * invContacts;
        
        // Apply angular correction (convert to quaternion update)
        float3 avgAngularCorrection = totalAngularCorrection * invContacts;
        float angularMagnitude = length(avgAngularCorrection);
        
        if (angularMagnitude > 1e-6) {
            float3 axis = avgAngularCorrection / angularMagnitude;
            float angle = angularMagnitude * 0.1; // Damping factor for stability
            
            // Update orientation quaternion
            float4 deltaQ = float4(sin(angle * 0.5) * axis, cos(angle * 0.5));
            rigidBody.orientation = normalizeQuat(float4(
                deltaQ.w * rigidBody.orientation.x + deltaQ.x * rigidBody.orientation.w +
                deltaQ.y * rigidBody.orientation.z - deltaQ.z * rigidBody.orientation.y,
                deltaQ.w * rigidBody.orientation.y - deltaQ.x * rigidBody.orientation.z +
                deltaQ.y * rigidBody.orientation.w + deltaQ.z * rigidBody.orientation.x,
                deltaQ.w * rigidBody.orientation.z + deltaQ.x * rigidBody.orientation.y -
                deltaQ.y * rigidBody.orientation.x + deltaQ.z * rigidBody.orientation.w,
                deltaQ.w * rigidBody.orientation.w - deltaQ.x * rigidBody.orientation.x -
                deltaQ.y * rigidBody.orientation.y - deltaQ.z * rigidBody.orientation.z
            ));
        }
        
        // Velocity update based on position corrections
        float3 velocityCorrection = totalPositionCorrection * invContacts / uniforms.deltaTime;
        rigidBody.linearVelocity += velocityCorrection;
        
        float3 angularVelocityCorrection = avgAngularCorrection / uniforms.deltaTime;
        rigidBody.angularVelocity += angularVelocityCorrection * 0.5; // Damping
        
        // Clamp velocities for stability
        const float maxLinearVel = 30.0;
        const float maxAngularVel = 10.0;
        
        rigidBody.linearVelocity = clamp(rigidBody.linearVelocity,
                                        float3(-maxLinearVel), float3(maxLinearVel));
        rigidBody.angularVelocity = clamp(rigidBody.angularVelocity,
                                         float3(-maxAngularVel), float3(maxAngularVel));
    }
}
