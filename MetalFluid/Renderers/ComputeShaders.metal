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

// MARK: - Rigid Body Material Functions

// Rigid body constraint forces to maintain shape
inline float3x3 rigidBodyConstraintForce(float3x3 F) {
    // For rigid body, F should remain close to identity (no deformation)
    float3x3 I = float3x3(1.0);
    float3x3 deviation = F - I;
    
    // Strong restoring force to maintain rigidity
    float constraint_strength = 1000.0;  // Very high to enforce rigidity
    return -constraint_strength * deviation;
}

// MARK: - Neo-Hookean Elastic Material Functions

// Manual 3x3 matrix inverse function for Metal
inline float3x3 inverse3x3(float3x3 m) {
    float det = determinant(m);
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

// Simplified Neo-Hookean stress based on working sample approach
inline float3x3 neoHookeanStress(float3x3 F, float lambda, float mu) {
    float J = clamp(determinant(F), 0.1, 10.0);  // More conservative bounds
    
    // Simplified approach from working sample: (F*F^T - I) + log(J)*I  
    float3x3 F_transpose = transpose(F);
    float3x3 deformTensor = F * F_transpose - float3x3(1.0);
    float3x3 volumeTensor = log(J) * float3x3(1.0);
    
    // Significantly increased stiffness for gravity resistance
    float stiffness_multiplier = 2.0;  // Much stronger to resist gravity
    float3x3 P = stiffness_multiplier * (mu * deformTensor + lambda * volumeTensor);
    
    // Allow higher stress to resist gravity
    const float max_stress = 300.0;  // Increased to allow strong elastic forces
    P[0] = clamp(P[0], float3(-max_stress), float3(max_stress));
    P[1] = clamp(P[1], float3(-max_stress), float3(max_stress));  
    P[2] = clamp(P[2], float3(-max_stress), float3(max_stress));
    
    return P;
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

// Compute boundary normal and distance from domain boundaries
inline void computeBoundaryInfo(float3 position, 
                               constant ComputeShaderUniforms& uniforms,
                               thread float3& boundaryNormal, 
                               thread float& boundaryDistance) {
    float3 boundaryMin = uniforms.boundaryMin;
    float3 boundaryMax = uniforms.boundaryMax;
    
    // Find closest boundary
    float3 distToMin = position - boundaryMin;
    float3 distToMax = boundaryMax - position;
    
    // Find minimum distance to any boundary
    float minDist = min(min(min(distToMin.x, distToMin.y), distToMin.z),
                       min(min(distToMax.x, distToMax.y), distToMax.z));
    
    boundaryDistance = minDist;
    
    // Determine boundary normal based on closest boundary
    if (minDist == distToMin.x) {
        boundaryNormal = float3(-1.0, 0.0, 0.0);  // Left wall
    } else if (minDist == distToMax.x) {
        boundaryNormal = float3(1.0, 0.0, 0.0);   // Right wall
    } else if (minDist == distToMin.y) {
        boundaryNormal = float3(0.0, -1.0, 0.0);  // Bottom wall
    } else if (minDist == distToMax.y) {
        boundaryNormal = float3(0.0, 1.0, 0.0);   // Top wall
    } else if (minDist == distToMin.z) {
        boundaryNormal = float3(0.0, 0.0, -1.0);  // Back wall
    } else {
        boundaryNormal = float3(0.0, 0.0, 1.0);   // Front wall
    }
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

// Rigid Body Particle to Grid Transfer (P2G)
kernel void particlesToGridRigid(
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
    
    // Quadratic B-spline weights
    float3 weights[3];
    bsplineWeights(cell_diff, weights);
    
    // Compute deformation gradient
    float3x3 F = computeDeformationGradient(C, uniforms.deltaTime);
    
    // Apply rigid body constraints (very strong forces to prevent deformation)
    float3x3 P = rigidBodyConstraintForce(F);
    
    // Particle volume (constant for rigid body)
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
                    
                    // Rigid body constraint force: -volume * P * grad_w * dt
                    float3 force = -volume * (P * cell_dist) * uniforms.deltaTime * weight;
                    
                    // Allow very strong forces for rigidity enforcement
                    const float max_force = 1000.0;  // Very high for rigid body constraints
                    force = clamp(force, float3(-max_force), float3(max_force));
                    
                    atomicAddWithUniform(&grid[cell_index].velocity_x, force.x, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_y, force.y, uniforms);
                    atomicAddWithUniform(&grid[cell_index].velocity_z, force.z, uniforms);
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
    
    // Apply gravity (skip for rigid bodies as they handle gravity in Stage 2)
    if (needGravityOnGrid(uniforms.materialMode)) {
        velocity.y += uniforms.gravity * uniforms.deltaTime;
    }
    
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

    // Compute boundary information for this particle
    float3 boundaryNormal;
    float boundaryDistance;
    computeBoundaryInfo(particles[id].position, uniforms, boundaryNormal, boundaryDistance);
    
    // Boundary handling using taichi-mpm approach
    const float boundaryThreshold = 2.0 * uniforms.gridSpacing;  // Distance threshold for near-boundary
    const float friction = 0.1;  // Fluid friction coefficient
    const float pushingForce = 0.0;  // No additional pushing force for fluid
    
    if (boundaryDistance < boundaryThreshold) {
        // Apply friction projection for boundary particles
        float3 projectedVelocity = frictionProject(particles[id].velocity, float3(0.0), boundaryNormal, friction);
        particles[id].velocity = projectedVelocity;
        
        // Add slight pushing force away from boundary if very close
        if (boundaryDistance < uniforms.gridSpacing) {
            particles[id].velocity += boundaryNormal * pushingForce * uniforms.deltaTime;
        }
        
        // Clear affine momentum near boundaries (following taichi-mpm)
        particles[id].C = float3x3(0.0);
    }
    
    // Handle mesh collision detection
    handleCollision(particles[id].position, particles[id].velocity,
                   particles[id].position, sdfTexture, collision);
    
    // Final position clamping as safety net
    float3 safetyMargin = float3(0.5) * uniforms.gridSpacing;
    particles[id].position = clamp(particles[id].position, 
                                  uniforms.boundaryMin + safetyMargin, 
                                  uniforms.boundaryMax - safetyMargin);
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
    
    // Update deformation gradient using stable approach from working sample
    float3x3 velocityGradient = B * 4.0; // Scale factor for grid spacing
    
    // Get current deformation gradient from C matrix  
    float3x3 currentDeform = float3x3(1.0) + particles[id].C * uniforms.deltaTime;
    
    // Update deformation gradient: F_new = F + velocity_gradient * F * dt
    float3x3 newDeform = currentDeform + velocityGradient * currentDeform * uniforms.deltaTime;
    
    // Check jacobian stability like in the working sample
    float newJacobian = determinant(newDeform);
    if (newJacobian < 0.05 || newJacobian > 20.0) {
        // Reset to identity if deformation becomes too extreme - critical for stability
        newJacobian = 1.0;
        newDeform = float3x3(1.0);
        velocityGradient = float3x3(0.0);
    }
    
    // Convert back to C matrix: C = (F - I) / dt
    particles[id].C = (newDeform - float3x3(1.0)) / uniforms.deltaTime;
    
    // Update particle position
    particles[id].position += particles[id].velocity * uniforms.deltaTime;
    
    // Boundary conditions for elastic materials - Essential for stability
    const float k = 3.0;
    const float wall_stiffness = 0.3;
    float3 wall_min = uniforms.boundaryMin + float3(3.0) * uniforms.gridSpacing;
    float3 wall_max = uniforms.boundaryMax - float3(4.0) * uniforms.gridSpacing;
    float3 x_n = particles[id].position + particles[id].velocity * uniforms.deltaTime * k;
    
    // Handle mesh collision detection
    handleCollision(particles[id].position, particles[id].velocity,
                   particles[id].position, sdfTexture, collision);
    
    // Wall collisions - Critical for preventing divergence
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
    
    // Position clamping - Essential safety net like in working sample
    particles[id].position = clamp(particles[id].position, 
                                  uniforms.boundaryMin + uniforms.gridSpacing, 
                                  uniforms.boundaryMax - 2.0 * uniforms.gridSpacing);
    
    // Velocity clamping to prevent extreme values while allowing strong elastic motion
    const float max_velocity = 35.0;  // Higher limit for strong elastic response
    particles[id].velocity = clamp(particles[id].velocity, 
                                  float3(-max_velocity), 
                                  float3(max_velocity));
}

// Rigid Body Grid to Particle Transfer (G2P)
kernel void gridToParticlesRigid1(
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
    float3x3 B = float3x3(0.0); // Velocity gradient for rigid body
    
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
                
                // For rigid body, severely constrain deformation
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
    
    // For rigid body, severely limit C matrix to prevent deformation
    float3x3 newC = B * 4.0; // Scale factor for grid spacing
    
    // Very strong constraints for rigid body - allow minimal deformation
    const float max_C_rigid = 0.1;  // Very small to maintain rigidity
    newC[0] = clamp(newC[0], float3(-max_C_rigid), float3(max_C_rigid));
    newC[1] = clamp(newC[1], float3(-max_C_rigid), float3(max_C_rigid));
    newC[2] = clamp(newC[2], float3(-max_C_rigid), float3(max_C_rigid));
    
    particles[id].C = newC;
    
    // Update particle position
    particles[id].position += particles[id].velocity * uniforms.deltaTime;
    
    // Boundary conditions for rigid body materials
    const float k = 3.0;
    const float wall_stiffness = 0.8;  // Strong wall interaction for rigid body
    float3 wall_min = uniforms.boundaryMin + float3(3.0) * uniforms.gridSpacing;
    float3 wall_max = uniforms.boundaryMax - float3(4.0) * uniforms.gridSpacing;
    float3 x_n = particles[id].position + particles[id].velocity * uniforms.deltaTime * k;
    
    // Handle mesh collision detection
    handleCollision(particles[id].position, particles[id].velocity,
                   particles[id].position, sdfTexture, collision);
    
    // Wall collisions with strong response for rigid body
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
    
    // Position clamping - Essential for rigid body
//    particles[id].position = clamp(particles[id].position, 
//                                  uniforms.boundaryMin + uniforms.gridSpacing, 
//                                  uniforms.boundaryMax - 2.0 * uniforms.gridSpacing);
    particles[id].position = position; //revert position for rigidbody update
    
//    // Velocity clamping for rigid body motion
//    const float max_velocity = 50.0;  // Allow fast rigid body motion
//    particles[id].velocity = clamp(particles[id].velocity, 
//                                  float3(-max_velocity), 
//                                  float3(max_velocity));
}

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

// Stage 1: Accumulate forces and torques from particles to rigid bodies using threadgroup memory
kernel void gridToParticlesRigid2(
    device const MPMParticle* particles [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    device RigidBodyState* rigidBodies [[buffer(2)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    // Use threadgroup memory for reduction
    threadgroup float3 local_forces[64];   // Assuming max 64 threads per group
    threadgroup float3 local_torques[64];
    threadgroup uint local_rigid_ids[64];
    
    // Initialize threadgroup memory
    local_forces[tid] = float3(0);
    local_torques[tid] = float3(0);
    local_rigid_ids[tid] = 0;
    
    if (id < uniforms.particleCount) {
        MPMParticle p = particles[id];
        uint origIdx = p.originalIndex;
        uint rigidId = rigidInfo[origIdx].rigidId;
        
        // Process particles that belong to a rigid body
        if (rigidId > 0) {
            uint rigidBodyIdx = rigidId - 1;
            if (rigidBodyIdx < uniforms.rigidBodyCount) {
                // Calculate forces
                float3 grid_force = p.mass * p.velocity / max(uniforms.deltaTime, 1e-6);
                float3 gravity_force = float3(0, p.mass * uniforms.gravity, 0);
                float3 total_force = grid_force + gravity_force;
                
                // Calculate torque
                float3 r = p.position - rigidBodies[rigidBodyIdx].centerOfMass;
                float3 torque = cross(r, total_force) * 0.05;
                
                // Store in threadgroup memory
                local_forces[tid] = total_force;
                local_torques[tid] = torque;
                local_rigid_ids[tid] = rigidId;
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Only first thread in threadgroup performs atomic operations
    if (tid == 0) {
        // Accumulate all forces and torques for each rigid body
        for (uint i = 0; i < threadgroup_size; i++) {
            if (local_rigid_ids[i] > 0) {
                uint bodyIdx = local_rigid_ids[i] - 1;
                if (bodyIdx < uniforms.rigidBodyCount) {
                    device atomic<float>* forcePtr = (device atomic<float>*)&rigidBodies[bodyIdx].accumulatedForce;
                    atomic_fetch_add_explicit(&forcePtr[0], local_forces[i].x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&forcePtr[1], local_forces[i].y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&forcePtr[2], local_forces[i].z, memory_order_relaxed);
                    
                    device atomic<float>* torquePtr = (device atomic<float>*)&rigidBodies[bodyIdx].accumulatedTorque;
                    atomic_fetch_add_explicit(&torquePtr[0], local_torques[i].x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&torquePtr[1], local_torques[i].y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&torquePtr[2], local_torques[i].z, memory_order_relaxed);
                }
            }
        }
    }
}

// Stage 2: Update rigid body dynamics (one thread per rigid body)
kernel void gridToParticlesRigid3(
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
    
    // Angular dynamics: τ = Iα → α = I⁻¹τ (following taichi-mpm approach)
    // Apply angular damping first (exponential decay)
    rb.angularVelocity *= exp(-rb.angularDamping * dt);
    
    // Integrate angular acceleration
    float3 angularAcceleration = rb.invInertiaTensor * rb.accumulatedTorque;
    
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
kernel void gridToParticlesRigid4(
    device MPMParticle* particles [[buffer(0)]],
    device const RigidBodyState* rigidBodies [[buffer(1)]],
    constant ComputeShaderUniforms& uniforms [[buffer(2)]],
    device const MPMParticleRigidInfo* rigidInfo [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    device MPMParticle& p = particles[id];
    uint origIdx = p.originalIndex;
    uint rigidId = rigidInfo[origIdx].rigidId;
    
    // Skip particles that don't belong to any rigid body
    if (rigidId == 0) return;
    
    uint rigidBodyIdx = rigidId - 1; // Convert to 0-based index
    if (rigidBodyIdx >= uniforms.rigidBodyCount) return;
    
    device const RigidBodyState& rb = rigidBodies[rigidBodyIdx];
    if (!rb.isActive) return;
    
    // Get rotation matrix from quaternion
    float3x3 R = quatToMatrix(rb.orientation);
    
    // Calculate new particle position: x_p = x_cm + R * r_p0
    float3 rotatedOffset = R * rigidInfo[origIdx].initialOffset;
    float3 newPosition = rb.centerOfMass + rotatedOffset;
    
    // Calculate new particle velocity: v_p = v_cm + ω × (R * r_p0)
    float3 angularContribution = cross(rb.angularVelocity, rotatedOffset);
    float3 newVelocity = rb.linearVelocity + angularContribution;
    
    // Update particle state
    p.position = newPosition;
    p.velocity = newVelocity;
    
    // Reset C matrix to prevent deformation in rigid bodies
    p.C = float3x3(0.0);
    
    // Apply boundary constraints
//    float3 boundaryMin = uniforms.boundaryMin + uniforms.gridSpacing;
//    float3 boundaryMax = uniforms.boundaryMax - uniforms.gridSpacing;
//    
//    p.position = clamp(p.position, boundaryMin, boundaryMax);
    
    // Clamp velocity to prevent instabilities
//    const float maxVel = 50.0;
//    p.velocity = clamp(p.velocity, float3(-maxVel), float3(maxVel));
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
        if (rigidInfo[i].rigidId == actualRigidId) {
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
        if (rigidInfo[i].rigidId == actualRigidId) {
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

    // Compute effective mass along normal including angular terms (following Taichi approach)
    float3 rnA = cross(rA, normal);
    float3 rnB = cross(rB, normal);
    float3 angA = cross( (A.invInertiaTensor * rnA), rA );
    float3 angB = cross( (B.invInertiaTensor * rnB), rB );
    float angularDenom = dot(normal, angA + angB);
    float denom = massSum + angularDenom;
    if (denom < 1e-6f) return;
    float J = -(1.0f + restitution) * relVN / denom;
    if (J < 0.0f) return;
    float3 impulse = J * normal;

    // Apply linear & angular impulses
    A.linearVelocity -= impulse * invMassA;
    B.linearVelocity += impulse * invMassB;
    A.angularVelocity -= A.invInertiaTensor * cross(rA, impulse);
    B.angularVelocity += B.invInertiaTensor * cross(rB, impulse);

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
        float3 angTA = cross( (A.invInertiaTensor * rtA), rA );
        float3 angTB = cross( (B.invInertiaTensor * rtB), rB );
        float denomT = massSum + dot(tangent, angTA + angTB);
        if (denomT > 1e-6f) {
            float jt = -dot(relV, tangent) / denomT;
            float maxFriction = friction * J;
            jt = clamp(jt, -maxFriction, maxFriction);
            float3 fImpulse = jt * tangent;
            A.linearVelocity -= fImpulse * invMassA;
            B.linearVelocity += fImpulse * invMassB;
            A.angularVelocity -= A.invInertiaTensor * cross(rA, fImpulse);
            B.angularVelocity += B.invInertiaTensor * cross(rB, fImpulse);
        }
    }
}
