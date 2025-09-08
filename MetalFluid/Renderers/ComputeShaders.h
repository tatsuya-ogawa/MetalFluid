#pragma once
#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
using namespace metal;
constant float GRADIENT_EPSILON=0.01;
constant float COLLISION_THRESHOLD=0.0;
// Material mode constants
constant int MATERIAL_FLUID = 0;
constant int MATERIAL_ELASTIC = 1;

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
    // Convert world position to grid position using gridSpacing
    gridPos = particleToGridPosition(particlePos, uniforms);
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

// --- Sparse sampling compatibility helpers ---
// Sample 3D SDF (single-channel). If the tile is not resident, return +inf to mean "no collision".
inline float sampleSparseSDFOrInf(
    texture3d<float, access::sample> sdfTexture,
    float3 texCoord)
{
    constexpr sampler sdfSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    auto sparseSample = sdfTexture.sparse_sample(sdfSampler, texCoord);
    return sparseSample.resident() ? sparseSample.value().r : INFINITY;
}

// --- SDF Collision Detection Utilities ---

// Compute normalized texture coordinates from world position
inline float3 worldToSDFTexCoord(float3 worldPos, constant CollisionUniforms &collision) {
    // Transform world position to mesh space using inverse transform
    float4 worldPos4 = float4(worldPos, 1.0);
    float4 meshSpacePos4 = collision.collisionInvTransform * worldPos4;
    float3 meshSpacePos = meshSpacePos4.xyz;
    
    // Convert mesh space position to SDF texture coordinates
    return (meshSpacePos - collision.sdfOrigin) / collision.sdfSize;
}

// Compute SDF normal/gradient only (more efficient when SDF value is already known)
inline float3 computeSDFNormal(float3 worldPos, texture3d<float> sdfTexture, constant CollisionUniforms &collision) {
    // Compute gradient (normal) using central differences in world space
    const float eps = GRADIENT_EPSILON; // Small epsilon in world space units
    float3 gradient;
    
    // Compute texture coordinates for gradient sampling points in world space
    float3 texCoordXPos = worldToSDFTexCoord(worldPos + float3(eps, 0, 0), collision);
    float3 texCoordXNeg = worldToSDFTexCoord(worldPos - float3(eps, 0, 0), collision);
    float3 texCoordYPos = worldToSDFTexCoord(worldPos + float3(0, eps, 0), collision);
    float3 texCoordYNeg = worldToSDFTexCoord(worldPos - float3(0, eps, 0), collision);
    float3 texCoordZPos = worldToSDFTexCoord(worldPos + float3(0, 0, eps), collision);
    float3 texCoordZNeg = worldToSDFTexCoord(worldPos - float3(0, 0, eps), collision);
    
    // Sample SDF values with bounds checking
    float sdfXPos = (any(texCoordXPos < 0.0) || any(texCoordXPos > 1.0)) ? 1.0 : sampleSparseSDFOrInf(sdfTexture, texCoordXPos);
    float sdfXNeg = (any(texCoordXNeg < 0.0) || any(texCoordXNeg > 1.0)) ? 1.0 : sampleSparseSDFOrInf(sdfTexture, texCoordXNeg);
    float sdfYPos = (any(texCoordYPos < 0.0) || any(texCoordYPos > 1.0)) ? 1.0 : sampleSparseSDFOrInf(sdfTexture, texCoordYPos);
    float sdfYNeg = (any(texCoordYNeg < 0.0) || any(texCoordYNeg > 1.0)) ? 1.0 : sampleSparseSDFOrInf(sdfTexture, texCoordYNeg);
    float sdfZPos = (any(texCoordZPos < 0.0) || any(texCoordZPos > 1.0)) ? 1.0 : sampleSparseSDFOrInf(sdfTexture, texCoordZPos);
    float sdfZNeg = (any(texCoordZNeg < 0.0) || any(texCoordZNeg > 1.0)) ? 1.0 : sampleSparseSDFOrInf(sdfTexture, texCoordZNeg);
    
    // Compute gradient using world space epsilon
    gradient.x = (sdfXPos - sdfXNeg) / (2.0 * eps);
    gradient.y = (sdfYPos - sdfYNeg) / (2.0 * eps);
    gradient.z = (sdfZPos - sdfZNeg) / (2.0 * eps);
    
    // Normalize gradient to get surface normal
    float gradLength = length(gradient);
    if (gradLength < 1e-6) {
        return float3(0, 1, 0); // Default to up vector
    } else {
        return gradient / gradLength;
    }
}

// Wrapper: SDF sampling with gradient computation (when both are needed)
inline float4 sampleSDFWithGradient(float3 particlePos, texture3d<float> sdfTexture, constant ComputeShaderUniforms& uniforms,constant CollisionUniforms &collision) {
    float3 worldPos = (uniforms.worldTransform *
                                         float4(particlePos, 1.0)).xyz;
    // Use the optimized coordinate transformation function
    float3 texCoord = worldToSDFTexCoord(worldPos, collision);
    
    // Check bounds
    if (any(texCoord < 0.0) || any(texCoord > 1.0)) {
        return float4(1.0, 0.0, 1.0, 0.0); // Outside bounds: no collision, default normal up
    }
    
    // Sample SDF value first
    float sdfValue = sampleSparseSDFOrInf(sdfTexture, texCoord);
    
    // Compute normal separately for efficiency
    float3 normal = computeSDFNormal(worldPos, sdfTexture, collision);
    
    return float4(sdfValue, normal);
}

// Compute SDF mass center in world space
inline float3 worldSDFMassCenter(constant CollisionUniforms &collision) {
    float4 p = collision.collisionTransform * float4(collision.sdfMassCenter, 1.0);
    return p.xyz;
}

// Threading and parallelization constants
constant int MAX_COLLISION_SDF = 8;
constant int DEFAULT_THREADGROUP_SIZE = 256;  // Default threadgroup size for compute shaders
constant int GRID_THREADGROUP_SIZE = 64;      // Threadgroup size for grid operations
// Argument buffer for SDF textures + per‑SDF collision uniforms (no physics here)
struct SDFSet {
    array<texture3d<float, access::sample>, MAX_COLLISION_SDF> sdf [[id(0)]];
    array<constant CollisionUniforms*, MAX_COLLISION_SDF> collision [[id(MAX_COLLISION_SDF)]];
};
