#pragma once
#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
using namespace metal;

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
