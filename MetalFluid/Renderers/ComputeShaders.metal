#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
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
    const float eps = 0.01;
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

inline void handleCollision(device float3 &position, device float3 &velocity, 
                           float3 worldPos, texture3d<float> sdfTexture, 
                           constant CollisionUniforms &collision) {
    if (!collision.enableCollision) return;
    
    // Transform world position to mesh space using inverse transform
    float4 worldPos4 = float4(worldPos, 1.0);
    float4 meshSpacePos4 = collision.collisionInvTransform * worldPos4;
    float3 meshSpacePos = meshSpacePos4.xyz;
    
    // Check if mesh space position is within reasonable bounds to avoid sampling issues
    float3 relativePos = (meshSpacePos - collision.sdfOrigin) / collision.sdfSize;
    if (any(relativePos < 0.0) || any(relativePos > 1.0)) {
        return; // Outside SDF bounds, no collision
    }
    
    float sdfValue = sampleSDF(worldPos, sdfTexture, collision);
    
    // Check for valid SDF value
    if (!isfinite(sdfValue)) {
        return; // Invalid SDF value, skip collision
    }
    
    // Handle collision with larger detection threshold
    const float collisionThreshold = 1.0; // Larger threshold for early detection
    if (sdfValue < collisionThreshold) {
        float3 normal = computeSDFNormal(worldPos, sdfTexture, collision);
        
        // Move particle outside surface with safety margin
        float pushDistance = max(-sdfValue + 0.5, 0.5); // Always push at least 0.5 units out
        position += normal * pushDistance;
        
        // Strong collision response
        const float restitution = 1.2;  // Strong bounce
        const float friction = 0.1;     // Low friction for now
        
        // Decompose velocity into normal and tangential components
        float vn = dot(velocity, normal);  // Normal component
        float3 vt = velocity - normal * vn; // Tangential component
        
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
        velocity = normal * vn + vt;
    }
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

// MARK: - Neo-Hookean Elastic Material Functions

// Manual 3x3 matrix inverse function for Metal
inline float3x3 inverse3x3(float3x3 m) {
//    float det = determinant(m);
    if (abs(det) < 1e-12) {
        // Return identity matrix if determinant is too small
        return float3x3(1.0);
    }
    
    float inv_det = 1.0 / det;
    
    float3x3 inv_m;
    inv_m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv_m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv_m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    
    inv_m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv_m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv_m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    
    inv_m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv_m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv_m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    
    return inv_m;
}

// Convert Young's modulus and Poisson's ratio to Lamé parameters
inline float2 computeLameParameters(float E, float nu) {
    float lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));  // First Lamé parameter
    float mu = E / (2.0 * (1.0 + nu));                           // Shear modulus (second Lamé parameter)
    return float2(lambda, mu);
}

// Neo-Hookean energy density derivative (P = dΨ/dF)
inline float3x3 neoHookeanStress(float3x3 F, float lambda, float mu) {
    float J = determinant(F);
    float3x3 F_inv_T = transpose(inverse3x3(F));
    
    // Neo-Hookean model: P = μ(F - F^-T) + λ*ln(J)*F^-T
    float3x3 P = mu * (F - F_inv_T) + lambda * log(J) * F_inv_T;
    
    return P;
}

// Compute deformation gradient from affine momentum matrix
inline float3x3 computeDeformationGradient(float3x3 C, float dt) {
    float3x3 I = float3x3(1.0);  // Identity matrix
    return I + dt * C;  // F = I + dt * C
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
kernel void particlesToGridFluid1(
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
                
                float mass_contrib = weight * p.mass * uniforms.massScale;
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
kernel void particlesToGridFluid2(
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

// Neo-Hookean Elastic Particle to Grid Transfer (P2G)
kernel void particlesToGridElastic(
                                 device MPMParticle* particles [[buffer(0)]],
                                 constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                                 device MPMGridNode* grid [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]
                                 ) {
    if (id >= uniforms.particleCount) return;
    
    MPMParticle particle = particles[id];
    float3 particlePosition = particle.position;
    float3 velocity = particle.velocity;
    float3x3 C = particle.C;
    
    // Convert particle position to grid coordinates
    float3 position, cell_diff;
    int3 cell_ids;
    particleToGridCoords(particlePosition, uniforms, position, cell_ids, cell_diff);
    
    // Quadratic B-spline weights (shared helper)
    float3 weights[3];
    bsplineWeights(cell_diff, weights);
    
    // Get Lamé parameters from material properties
    float2 lame = computeLameParameters(uniforms.youngsModulus, uniforms.poissonsRatio);
    float lambda = lame.x;
    float mu = lame.y;
    
    // Compute deformation gradient
    float3x3 F = computeDeformationGradient(C, uniforms.deltaTime);
    
    // Compute Neo-Hookean stress
    float3x3 P = neoHookeanStress(F, lambda, mu);
    
    // Particle volume (constant for elastic materials)
    float volume = uniforms.massScale / uniforms.rest_density;
    
    // Transfer to grid nodes
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                int3 cell_idx = getOffsetCellIndex(cell_ids, gx, gy, gz);
                if (cell_idx.x >= 0 && cell_idx.x < uniforms.gridResolution.x &&
                    cell_idx.y >= 0 && cell_idx.y < uniforms.gridResolution.y &&
                    cell_idx.z >= 0 && cell_idx.z < uniforms.gridResolution.z) {
                    uint cell_index = gridIndex(cell_idx.x, cell_idx.y, cell_idx.z, uniforms.gridResolution);
                    
                    float3 cell_dist = (float3(cell_idx) + 0.5) - position;
                    float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                    
                    // Mass contribution
                    float mass_contrib = weight * uniforms.massScale;
                    atomicAddWithUniform(&grid[cell_index].mass, mass_contrib, uniforms);
                    
                    // Momentum contribution (mass * velocity)
                    float3 momentum = mass_contrib * velocity;
                    atomicAddWithUniform(&grid[cell_index].velocity_x, momentum.x, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_y, momentum.y, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_z, momentum.z, uniforms);
                    
                    // Elastic force contribution: -volume * P * grad_w * dt
                    float3 force = -volume * (P * cell_dist) * uniforms.deltaTime * weight;
                    atomicAddWithUniform(&grid[cell_index].velocity_x, force.x, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_y, force.y, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_z, force.z, uniforms);
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
kernel void gridToParticlesFluid1(
                            device MPMParticle* particles [[buffer(0)]],
                            constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                            device const NonAtomicMPMGridNode* grid [[buffer(2)]],
                            texture3d<float> sdfTexture [[texture(0)]],
                            constant CollisionUniforms& collision [[buffer(3)]],
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

    // Handle mesh collision detection (now safe from hanging)
    handleCollision(particles[id].position, particles[id].velocity,
                   particles[id].position, sdfTexture, collision);

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

// Neo-Hookean Elastic Grid to Particle Transfer (G2P)
kernel void gridToParticlesElastic(
                                  device MPMParticle* particles [[buffer(0)]],
                                  constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                                  device const NonAtomicMPMGridNode* grid [[buffer(2)]],
                                  texture3d<float> sdfTexture [[texture(0)]],
                                  constant CollisionUniforms& collision [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]
                                  ) {
    if (id >= uniforms.particleCount) return;
    
    MPMParticle p = particles[id];
    
    // Convert particle position to grid coordinates
    float3 position, cell_diff;
    int3 cell_ids;
    particleToGridCoords(p.position, uniforms, position, cell_ids, cell_diff);
    
    // Quadratic B-spline weights
    float3 weights[3];
    bsplineWeights(cell_diff, weights);
    
    float3 new_velocity = float3(0.0);
    float3x3 B = float3x3(0.0); // Velocity gradient for affine momentum
    
    // Interpolate from surrounding grid nodes
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                int3 cell_idx = getOffsetCellIndex(cell_ids, gx, gy, gz);
                
                if (cell_idx.x < 0 || cell_idx.x >= uniforms.gridResolution.x ||
                    cell_idx.y < 0 || cell_idx.y >= uniforms.gridResolution.y ||
                    cell_idx.z < 0 || cell_idx.z >= uniforms.gridResolution.z) continue;
                
                float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                uint gridIdx = gridIndex(cell_idx.x, cell_idx.y, cell_idx.z, uniforms.gridResolution);
                
                float3 grid_velocity = float3(
                    nonAtomicLoadWithUniform(&grid[gridIdx].velocity_x, uniforms),
                    nonAtomicLoadWithUniform(&grid[gridIdx].velocity_y, uniforms),
                    nonAtomicLoadWithUniform(&grid[gridIdx].velocity_z, uniforms)
                );
                
                float3 cell_dist = (float3(cell_idx) + 0.5) - position;
                
                new_velocity += weight * grid_velocity;
                
                // For elastic materials, we update the deformation gradient instead of traditional APIC
                // B = sum(w_ip * v_i * x_ip^T) where x_ip is the distance to grid node
                float3x3 outer_prod = float3x3(
                    grid_velocity * cell_dist.x,
                    grid_velocity * cell_dist.y,
                    grid_velocity * cell_dist.z
                );
                B += weight * outer_prod;
            }
        }
    }
    
    // Update particle velocity
    particles[id].velocity = new_velocity;
    
    // Update affine momentum matrix (stores velocity gradient information)
    // For elastic materials, this represents the local deformation rate
    particles[id].C = B * 4.0; // Scale factor for grid spacing
    
    // Update particle position
    particles[id].position += particles[id].velocity * uniforms.deltaTime;
    
    // Boundary conditions for elastic materials
    const float k = 3.0;
    const float wall_stiffness = 0.3;
    float3 wall_min = uniforms.boundaryMin + float3(3.0) * uniforms.gridSpacing;
    float3 wall_max = uniforms.boundaryMax - float3(4.0) * uniforms.gridSpacing;
    float3 x_n = particles[id].position + particles[id].velocity * uniforms.deltaTime * k;
    
    // Handle mesh collision detection
    handleCollision(particles[id].position, particles[id].velocity,
                   particles[id].position, sdfTexture, collision);
    
    // Wall collisions
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
    
    // Position clamping
    particles[id].position = clamp(particles[id].position, 
                                  uniforms.boundaryMin + uniforms.gridSpacing, 
                                  uniforms.boundaryMax - 2.0 * uniforms.gridSpacing);
}
