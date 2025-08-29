#pragma once
#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
using namespace metal;
constant float GRADIENT_EPSILON=0.01;
constant float COLLISION_THRESHOLD=1.0;
// Material mode constants
constant int MATERIAL_FLUID = 0;
constant int MATERIAL_ELASTIC = 1;
constant int MATERIAL_RIGID = 2;

// Check if gravity should be applied on grid for this material mode
inline bool needGravityOnGrid(uint32_t materialMode) {
    // Skip gravity on grid for rigid bodies (they handle gravity in Stage 2)
    return materialMode != MATERIAL_RIGID;
}

// --- Utility: clamp velocity to avoid NaN/Inf and large values ---
inline float3 clampVelocity(float3 v) {
    return v;
    const float vmax = 1.0e+1f;
    float3 outv;
    outv.x = clamp(v.x, -vmax, vmax);
    outv.y = clamp(v.y, -vmax, vmax);
    outv.z = clamp(v.z, -vmax, vmax);
    return outv;
}

inline float clampDensity(float d){
    return d;
    const float dmax = 1.0e+6f;
    return clamp(d,-dmax, dmax);
}

// --- Utility: clamp Affine matrix C to avoid large values ---
inline float3x3 clampAffineC(float3x3 C) {
    return C;
    const float Cmax = 0.5;//1.0e+7f;
    // Frobenius norm
    float n2 = dot(C[0], C[0]) + dot(C[1], C[1]) + dot(C[2], C[2]);
    float n  = sqrt(max(n2, 1e-12f));
    if (n <= Cmax) return C;
    
    // Soft coefficient: avoid sharp cuts (tanh-like)
    // Example: scale = Cmax / (n + (n-Cmax)) = Cmax/n (pure scale) would also work
    float scale = Cmax / n;                      // Close to hard but sufficiently smooth
    return float3x3(C[0]*scale, C[1]*scale, C[2]*scale);
}

inline uint gridIndex(uint x, uint y, uint z, int3 res) {
    return x * (res.z * res.y) + y * res.z + z;
}

inline uint3 gridXYZ(uint id,constant ComputeShaderUniforms &uniforms){
    // Get grid coordinates
    uint resZ = uniforms.gridResolution.z;
    uint resY = uniforms.gridResolution.y;
    uint x = id / resZ / resY;
    uint y = (id / resZ) % resY;
    uint z = id % resZ;
    return uint3(x,y,z);
}

// --- Utility: Convert particle position to grid coordinates ---
inline float3 particleToGridPosition(float3 particlePos, constant ComputeShaderUniforms &uniforms) {
    return (particlePos - uniforms.domainOrigin) / uniforms.gridSpacing;
}

// --- Utility: Get grid cell indices from grid position ---
inline int3 getGridCellIndices(float3 gridPos, float offset = 0.5) {
    return int3(floor(gridPos - offset));
}

// --- Utility: Get cell difference for B-spline weights ---
inline float3 getCellDifference(float3 gridPos, int3 cellIds) {
    return gridPos - (float3(cellIds) + 0.5);
}

// --- Particle to grid coordinate conversion ---
inline void particleToGridCoords(float3 particlePos,
                                       constant ComputeShaderUniforms &uniforms,
                                       thread float3 &gridPos,
                                       thread int3 &cellIds,
                                       thread float3 &cellDiff) {
    // Use particle position directly as grid position
    gridPos = particlePos;
    cellIds = getGridCellIndices(gridPos, 0.0);
    cellDiff = getCellDifference(gridPos, cellIds);
}

// --- Utility: atomic add for float with uniform (for future optimization/fixed-point) ---
inline void atomicAddWithUniform(device MPM_ATOMIC_FLOAT *ptr, float value, constant ComputeShaderUniforms &uniforms) {
    // Conversion and fixed-point conversion using uniforms possible here
    atomic_fetch_add_explicit(ptr, value, memory_order_relaxed);
}

// --- Utility: atomic load/store for float with uniform (for future optimization/fixed-point) ---
inline float atomicLoadWithUniform(const device MPM_ATOMIC_FLOAT *ptr, constant ComputeShaderUniforms &uniforms) {
    // Conversion and fixed-point conversion using uniforms possible here
    return atomic_load_explicit(ptr, memory_order_relaxed);
}

inline void atomicStoreWithUniform(device MPM_ATOMIC_FLOAT *ptr, float value, constant ComputeShaderUniforms &uniforms) {
    // Conversion and fixed-point conversion using uniforms possible here
    atomic_store_explicit(ptr, value, memory_order_relaxed);
}

inline float nonAtomicLoadWithUniform(const device MPM_NON_ATOMIC_FLOAT *ptr, constant ComputeShaderUniforms &uniforms) {
    // Conversion and fixed-point conversion using uniforms possible here
    return *ptr;
}

inline void nonAtomicStoreWithUniform(device MPM_NON_ATOMIC_FLOAT *ptr, float value, constant ComputeShaderUniforms &uniforms) {
    // Conversion and fixed-point conversion using uniforms possible here
    *ptr = value;
}

inline int3 getOffsetCellIndex(int3 cell_idx,int gx,int gy,int gz){
    return cell_idx + int3(gx-1, gy-1, gz-1);
}

// ---- Common Quadratic B-spline helpers (Taichi/MPM style with fx in [-0.5, 0.5]) ----
inline void bsplineWeights(const float3 fx, thread float3 weights[3]) {
    // weights[k].x/y/z = wk along each axis
    weights[0] = 0.5 * (0.5 - fx) * (0.5 - fx);
    weights[1] = 0.75 - fx * fx;
    weights[2] = 0.5 * (0.5 + fx) * (0.5 + fx);
}

// Compute deformation gradient from affine momentum matrix
inline float3x3 computeDeformationGradient(float3x3 C, float dt) {
    float3x3 I = float3x3(1.0);  // Identity matrix
    return I + dt * C;  // F = I + dt * C
}

// Friction projection following taichi-mpm approach
// Projects velocity considering boundary friction and normal constraints
inline float3 frictionProject(float3 velocity, float3 baseVelocity, float3 normal, float friction) {
    float3 relativeVel = velocity - baseVelocity;
    
    // Special case: sticky boundary (mu = -1)
    if (friction == -1.0) {
        return baseVelocity;  // Completely stick to boundary
    }
    
    // Check for slip boundary (mu <= -2)
    bool slip = friction <= -2.0;
    if (slip) {
        friction = -friction - 2.0;  // Extract actual friction value
    }
    
    // Project relative velocity
    float normalComponent = dot(normal, relativeVel);
    float3 tangentialVel = relativeVel - normalComponent * normal;
    float tangentialNorm = length(tangentialVel);
    
    // Compute tangential scaling factor (Coulomb friction law)
    float tangentialScale = max(tangentialNorm + min(normalComponent, 0.0) * friction, 0.0) /
                           max(1e-30, tangentialNorm);
    
    // Compute projected relative velocity
    float3 projectedRelativeVel = tangentialScale * tangentialVel +
                                 max(0.0, normalComponent * (slip ? 0.0 : 1.0)) * normal;
    
    return projectedRelativeVel + baseVelocity;
}

// --- SDF Collision Detection Utilities ---
inline float sampleSDF(float3 worldPos, texture3d<float> sdfTexture, constant CollisionUniforms &collision) {
    // Transform world position to mesh space using inverse transform
    float4 worldPos4 = float4(worldPos, 1.0);
    float4 meshSpacePos4 = collision.collisionInvTransform * worldPos4;
    float3 meshSpacePos = meshSpacePos4.xyz;
    
    // Convert mesh space position to SDF texture coordinates
    float3 texCoord = (meshSpacePos - collision.sdfOrigin) / collision.sdfSize;
    
    // Check bounds
    if (any(texCoord < 0.0) || any(texCoord > 1.0)) {
        return 1.0; // Outside SDF volume, no collision
    }
    
    constexpr sampler sdfSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    return sdfTexture.sample(sdfSampler, texCoord).r;
}

inline float3 computeSDFNormal(float3 worldPos, texture3d<float> sdfTexture, constant CollisionUniforms &collision) {
    const float eps = GRADIENT_EPSILON;
    float3 gradient;
    
    gradient.x = sampleSDF(worldPos + float3(eps, 0, 0), sdfTexture, collision) -
                 sampleSDF(worldPos - float3(eps, 0, 0), sdfTexture, collision);
    gradient.y = sampleSDF(worldPos + float3(0, eps, 0), sdfTexture, collision) -
                 sampleSDF(worldPos - float3(0, eps, 0), sdfTexture, collision);
    gradient.z = sampleSDF(worldPos + float3(0, 0, eps), sdfTexture, collision) -
                 sampleSDF(worldPos - float3(0, 0, eps), sdfTexture, collision);
    
    gradient = gradient / (2.0 * eps);
    
    // Safely normalize gradient, avoid division by zero
    float gradLength = length(gradient);
    if (gradLength < 1e-6) {
        return float3(0, 1, 0); // Default to up vector if gradient is too small
    }
    
    return gradient / gradLength;
}
inline void handleCollision(device float3& particlePos,
                            device float3& particleVel,
                            float particleMass,
                            texture3d<float> sdfTexture,
                            constant CollisionUniforms &collision,
                            float dt) {
    if (!collision.enableCollision) return;
    
    // Transform world position to mesh space using inverse transform
    float4 worldPos4 = float4(particlePos, 1.0);
    float4 meshSpacePos4 = collision.collisionInvTransform * worldPos4;
    float3 meshSpacePos = meshSpacePos4.xyz;
    
    // Check if mesh space position is within reasonable bounds to avoid sampling issues
    float3 relativePos = (meshSpacePos - collision.sdfOrigin) / collision.sdfSize;
    if (any(relativePos < 0.0) || any(relativePos > 1.0)) {
        return; // Outside SDF bounds, no collision
    }
    
    float sdfValue = sampleSDF(particlePos, sdfTexture, collision);
    
    // Check for valid SDF value
    if (!isfinite(sdfValue)) {
        return; // Invalid SDF value, skip collision
    }
    
    // Handle collision with larger detection threshold
    const float collisionThreshold = COLLISION_THRESHOLD; // Larger threshold for early detection
    if (sdfValue < collisionThreshold) {
        float3 normal = computeSDFNormal(particlePos, sdfTexture, collision);
        
        // Move particle outside surface with safety margin
        float pushDistance = max(-sdfValue + 0.5, 0.5); // Always push at least 0.5 units out
        particlePos += normal * pushDistance;
        
        // Strong collision response
        const float restitution = 1.2;  // Strong bounce
        const float friction = 0.1;     // Low friction for now
        
        // Decompose velocity into normal and tangential components
        float vn = dot(particleVel, normal);  // Normal component
        float3 vt = particleVel - normal * vn; // Tangential component
        
        // Always apply strong outward velocity
        if (vn < 2.0) { // If not already moving fast outward
            vn = max(vn * -restitution, 3.0); // Strong outward push
        }
        
        // Apply minimal friction to tangential component
        float vt_magnitude = length(vt);
        if (vt_magnitude > 0.01) {
            vt *= (1.0 - friction);
        }
        
        // Recombine velocity components
        particleVel = normal * vn + vt;
    }
}
// Improved collision handling following taichi-mpm approaches
inline void handleCollisionTaichi(device float3 &position, device float3 &velocity,
                           float3 worldPos, texture3d<float> sdfTexture,
                           constant CollisionUniforms &collision) {
    if (!collision.enableCollision) return;
    
    // Transform world position to mesh space using inverse transform
    float4 worldPos4 = float4(worldPos, 1.0);
    float4 meshSpacePos4 = collision.collisionInvTransform * worldPos4;
    float3 meshSpacePos = meshSpacePos4.xyz;
    
    // Check if mesh space position is within reasonable bounds
    float3 relativePos = (meshSpacePos - collision.sdfOrigin) / collision.sdfSize;
    if (any(relativePos < 0.0) || any(relativePos > 1.0)) {
        return; // Outside SDF bounds, no collision
    }
    
    float phi = sampleSDF(worldPos, sdfTexture, collision);
    
    // Check for valid SDF value
    if (!isfinite(phi)) {
        return; // Invalid SDF value, skip collision
    }
    
    // Handle collision with larger detection threshold
    const float collisionThreshold = 0.0; // Larger threshold for early detection
    if (phi < collisionThreshold) {
        float3 gradient = computeSDFNormal(worldPos, sdfTexture, collision);
        
        // Taichi-MPM position correction: p.pos -= gradient * phi * delta_x
        // Move particle outside surface proportional to penetration depth
        position -= gradient * phi;
        
        // Taichi-MPM velocity projection: v = v - dot(gradient, v) * gradient
        // Remove normal component of velocity (no penetration)
        float normalVelocity = dot(gradient, velocity);
        velocity = velocity - normalVelocity * gradient;
        
        // Apply friction using improved friction projection
        const float friction = collision.collisionStiffness * 0.1; // Use collision stiffness as base
        const float restitution = 0.8; // Moderate restitution
        
        // Add slight restitution if particle was moving into surface
        if (normalVelocity < 0.0) {
            velocity += gradient * (-normalVelocity * restitution);
        }
        
        // Apply friction to tangential velocity using frictionProject
        float3 baseVelocity = float3(0.0); // Static surface
        velocity = frictionProject(velocity, baseVelocity, gradient, friction);
    }
}


#define MAX_RIGIDS 8
struct SDFSet {
  array<texture3d<float, access::sample>, MAX_RIGIDS> sdf [[id(0)]];
};
