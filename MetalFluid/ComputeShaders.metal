#include <metal_stdlib>
#include "MPMTypes.h"  // Get struct definitions from shared header file
using namespace metal;
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

// ===== MLS-MPM FLUID SIMULATION KERNELS =====

// MLS-MPM Grid Clear
kernel void clearGrid(
                      device NonAtomicMPMGridNode* grid [[buffer(0)]],
                      constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                      uint id [[thread_position_in_grid]]
                      ) {
    if (id >= uniforms.gridNodeCount) return;
    
    nonAtomicStoreWithUniform(&grid[id].mass, 0.0f, uniforms);
    nonAtomicStoreWithUniform(&grid[id].velocity_x, 0.0f, uniforms);
    nonAtomicStoreWithUniform(&grid[id].velocity_y, 0.0f, uniforms);
    nonAtomicStoreWithUniform(&grid[id].velocity_z, 0.0f, uniforms);
}

// MLS-MPM Particle to Grid (P2G) Phase - With volume recalculation

// P2G1: Mass and momentum transfer from particles to grid (APIC/affine terms only)
kernel void particlesToGrid1(
                             device const MPMParticle* particles [[buffer(0)]],
                             constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                             device MPMGridNode* grid [[buffer(2)]],
                             uint id [[thread_position_in_grid]]
                             ) {
    if (id >= uniforms.particleCount) return;
    MPMParticle p = particles[id];
    float3 position, cell_diff;
    int3 cell_ids;
    
    // Use coordinate conversion
    particleToGridCoords(p.position, uniforms, position, cell_ids, cell_diff);
    
    float3 w[3];
    bsplineWeights(cell_diff, w);
    float3x3 C = clampAffineC(p.C);
    float3 velocity = clampVelocity(p.velocity);
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                int3 cell_idx = getOffsetCellIndex(cell_ids,gx, gy, gz);
                
                if (cell_idx.x < 0 || cell_idx.x >= uniforms.gridResolution.x ||
                    cell_idx.y < 0 || cell_idx.y >= uniforms.gridResolution.y ||
                    cell_idx.z < 0 || cell_idx.z >= uniforms.gridResolution.z) continue;
                float weight = w[gx].x * w[gy].y * w[gz].z;
                uint cell_index = gridIndex(cell_idx.x, cell_idx.y, cell_idx.z, uniforms.gridResolution);
                // Neighbor cell center corresponding to weight index (base + gx,gy,gz)
                float3 cell_dist = (float3(cell_idx) + 0.5f) - position;
                float3 Q = C * cell_dist;
                
                float mass_contrib = weight * p.mass;
                float3 vel_contrib = mass_contrib * (velocity + Q);
                atomicAddWithUniform(&grid[cell_index].mass, mass_contrib, uniforms);
                atomicAddWithUniform(&grid[cell_index].velocity_x, vel_contrib.x, uniforms);
                atomicAddWithUniform(&grid[cell_index].velocity_y, vel_contrib.y, uniforms);
                atomicAddWithUniform(&grid[cell_index].velocity_z, vel_contrib.z, uniforms);
            }
        }
    }
}

// P2G2: Add volume and stress-based momentum from particles to grid
kernel void particlesToGrid2(
                             device const MPMParticle* particles [[buffer(0)]],
                             constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                             device MPMGridNode* grid [[buffer(2)]],
                             uint id [[thread_position_in_grid]]
                             ) {
    if (id >= uniforms.particleCount) return;
    
    MPMParticle particle = particles[id];
    
    // Convert particle position to grid coordinates
    float3 position, cell_diff;
    int3 cell_ids;
    particleToGridCoords(particle.position, uniforms, position, cell_ids, cell_diff);
    
    // Quadratic B-spline weights
    float3 weights[3];
    bsplineWeights(cell_diff, weights);
    
    // Density calculation (from surrounding grid node masses)
    float density = 0.0;
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                int3 cell_idx = getOffsetCellIndex(cell_ids,gx, gy, gz);
                
                if (cell_idx.x >= 0 && cell_idx.x < uniforms.gridResolution.x &&
                    cell_idx.y >= 0 && cell_idx.y < uniforms.gridResolution.y &&
                    cell_idx.z >= 0 && cell_idx.z < uniforms.gridResolution.z) {
                    
                    uint cell_index = gridIndex(cell_idx.x, cell_idx.y, cell_idx.z, uniforms.gridResolution);
                    float grid_mass = atomicLoadWithUniform(&grid[cell_index].mass, uniforms);
                    density += grid_mass * weight;
                }
            }
        }
    }
    
    if(density==0)return;
    density = clampDensity(density);
    float volume = 1.0 / density;
    
    // Pressure calculation (equation of state)
    float pressure = max(0.0, uniforms.stiffness * (pow(density / uniforms.rest_density, 5.0) - 1.0));
    
    // Stress tensor calculation
    float3x3 stress = float3x3(-pressure, 0, 0, 0, -pressure, 0, 0, 0, -pressure);
    float3x3 dudv = particle.C;
    float3x3 strain = dudv + transpose(dudv);
    
    stress += uniforms.dynamic_viscosity * strain;
    
    // Momentum term calculation
    float3x3 eq_16_term0 = -volume * 4.0 * stress * uniforms.deltaTime;
    // Distribute momentum (impulse) to surrounding grid nodes: impulse = -V * (stress * grad_w) * dt
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                int3 cell_idx = getOffsetCellIndex(cell_ids,gx, gy, gz);
                if (cell_idx.x >= 0 && cell_idx.x < uniforms.gridResolution.x &&
                    cell_idx.y >= 0 && cell_idx.y < uniforms.gridResolution.y &&
                    cell_idx.z >= 0 && cell_idx.z < uniforms.gridResolution.z) {
                    uint cell_index = gridIndex(cell_idx.x, cell_idx.y, cell_idx.z, uniforms.gridResolution);
                    
                    float3 cell_dist = (float3(cell_idx) + 0.5) - position;
                    float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                    float3 momentum = (eq_16_term0 * cell_dist) * weight;
                    atomicAddWithUniform(&grid[cell_index].velocity_x, momentum.x, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_y, momentum.y, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_z, momentum.z, uniforms);
                }
            }
        }
    }
}

// MLS-MPM Grid velocity update and boundary conditions
kernel void updateGridVelocity(
                               device NonAtomicMPMGridNode* grid [[buffer(0)]],
                               constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                               uint id [[thread_position_in_grid]]
                               ) {
    if (id >= uniforms.gridNodeCount) return;
    
    float mass = nonAtomicLoadWithUniform(&grid[id].mass, uniforms);
    if (mass <= 0.0f) return;
    float3 velocity = float3(
                             nonAtomicLoadWithUniform(&grid[id].velocity_x, uniforms),
                             nonAtomicLoadWithUniform(&grid[id].velocity_y, uniforms),
                             nonAtomicLoadWithUniform(&grid[id].velocity_z, uniforms)
                             ) / mass;
    
    // Apply gravity
    velocity.y += uniforms.gravity * uniforms.deltaTime;
    
    // Get grid coordinates
    uint3 xyz = gridXYZ(id,uniforms);
    float3 zeroMin = uniforms.boundaryMin + float3(1.0,1.0,1.0);
    float3 zeroMax = uniforms.boundaryMax - float3(2.0,2.0,2.0);
    // Boundary conditions (sticky boundary, judged by world coordinates)
    float3 cellCenter = uniforms.domainOrigin + float3(xyz.x, xyz.y, xyz.z) * uniforms.gridSpacing;
    if (cellCenter.x < zeroMin.x || cellCenter.x > zeroMax.x) velocity.x = 0.0;
    if (cellCenter.y < zeroMin.y || cellCenter.y > zeroMax.y) velocity.y = 0.0;
    if (cellCenter.z < zeroMin.z || cellCenter.z > zeroMax.z) velocity.z = 0.0;
    
    // Save updated velocity
    nonAtomicStoreWithUniform(&grid[id].velocity_x, velocity.x, uniforms);
    nonAtomicStoreWithUniform(&grid[id].velocity_y, velocity.y, uniforms);
    nonAtomicStoreWithUniform(&grid[id].velocity_z, velocity.z, uniforms);
}

// MLS-MPM Grid to Particle (G2P) Phase - Volume recalculation version
kernel void gridToParticles(
                            device MPMParticle* particles [[buffer(0)]],
                            constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                            device const NonAtomicMPMGridNode* grid [[buffer(2)]],
                            uint id [[thread_position_in_grid]]
                            ) {
    if (id >= uniforms.particleCount) return;
    
    MPMParticle p = particles[id];
    
    // Convert particle position to grid coordinates (considering domainOrigin)
    float3 position, cell_diff;
    int3 cell_ids;
    particleToGridCoords(p.position, uniforms, position, cell_ids, cell_diff);
    
    // Quadratic B-spline weights (shared helper)
    float3 weights[3];
    bsplineWeights(cell_diff, weights);
    
    float3 new_velocity = float3(0.0);
    float3x3 B = float3x3(0.0); // Affine velocity gradient
    
    // Interpolate from surrounding grid nodes
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                int3 cell_idx = getOffsetCellIndex(cell_ids,gx, gy, gz);
                
                if (cell_idx.x < 0 || cell_idx.x >= uniforms.gridResolution.x ||
                    cell_idx.y < 0 || cell_idx.y >= uniforms.gridResolution.y ||
                    cell_idx.z < 0 || cell_idx.z >= uniforms.gridResolution.z) continue;
                
                float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                uint gridIdx = gridIndex(cell_idx.x, cell_idx.y, cell_idx.z, uniforms.gridResolution);
                
                // Read the velocity from the correct grid cell (use gridIdx, not particle id)
                float3 grid_velocity = float3(
                                              nonAtomicLoadWithUniform(&grid[gridIdx].velocity_x, uniforms),
                                              nonAtomicLoadWithUniform(&grid[gridIdx].velocity_y, uniforms),
                                              nonAtomicLoadWithUniform(&grid[gridIdx].velocity_z, uniforms)
                                              );
                float3 weighted_velocity = weight * grid_velocity;
                new_velocity += weighted_velocity;
                
                float3 cell_dist = (float3(cell_idx) + 0.5f) - position;
                
                float3x3 term = float3x3(weighted_velocity * cell_dist.x,weighted_velocity * cell_dist.y,weighted_velocity * cell_dist.z);
                B += term;
            }
        }
    }
    // C (Affine matrix) clamp processing
    float3x3 unclampedC = B * 4.0f;
    particles[id].C = unclampedC;
    particles[id].velocity = new_velocity;
    particles[id].position += particles[id].velocity * uniforms.deltaTime;
    
    // --- Velocity correction by wall repulsion force ---
    const float k = 3.0;
    const float wall_stiffness = 0.3;
    float3 wall_min = uniforms.boundaryMin + float3(1.0,1.0,1.0)*3.0 * uniforms.gridSpacing;
    float3 wall_max = uniforms.boundaryMax - float3(1.0,1.0,1.0)*4.0 * uniforms.gridSpacing;
    float3 x_n = particles[id].position + particles[id].velocity * uniforms.deltaTime * k;
    if (x_n.x < wall_min.x) {
        particles[id].velocity.x += wall_stiffness * (wall_min.x - x_n.x);
    }
    if (x_n.x > wall_max.x) {
        particles[id].velocity.x += wall_stiffness * (wall_max.x - x_n.x);
    }
    if (x_n.y < wall_min.y) {
        particles[id].velocity.y += wall_stiffness * (wall_min.y - x_n.y);
    }
    if (x_n.y > wall_max.y) {
        particles[id].velocity.y += wall_stiffness * (wall_max.y - x_n.y);
    }
    if (x_n.z < wall_min.z) {
        particles[id].velocity.z += wall_stiffness * (wall_min.z - x_n.z);
    }
    if (x_n.z > wall_max.z) {
        particles[id].velocity.z += wall_stiffness * (wall_max.z - x_n.z);
    }
    // Clamp position based on boundaries (direct clamp in world coordinates)
    particles[id].position = clamp(particles[id].position, uniforms.boundaryMin+1.0*uniforms.gridSpacing, uniforms.boundaryMax-2.0*uniforms.gridSpacing);
}
