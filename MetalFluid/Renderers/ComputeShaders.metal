#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
#include "ComputeShaders.h"
#include "SDFCollision.h"
using namespace metal;
// MARK: - Neo-Hookean Elastic Material Functions

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

// ===== COLLISION FUNCTIONS (Newton-inspired) =====

// Coulomb friction model - solve for relative velocity in isotropic friction
// Based on Newton's solve_coulomb_isotropic (solve_rheology.py:667-687)
inline float3 solveCoulombIsotropic(float mu, float3 normal, float3 u) {
    float u_n = dot(u, normal);

    if (u_n < 0.0) {  // Contact: normal pointing inward
        u -= u_n * normal;  // Remove normal component
        float tau = dot(u, u);  // Tangential velocity squared
        float alpha = mu * u_n;  // Friction threshold

        if (tau <= alpha * alpha) {
            // Static friction - stick
            u = float3(0.0);
        } else {
            // Dynamic friction - slide
            u *= 1.0 + mu * u_n / sqrt(tau);
        }
    }

    return u;
}

// Improved boundary collision with Coulomb friction
// Based on Newton's project_outside_collider (rasterized_collisions.py:172-238)
inline void projectOutsideBoundary(
    thread float3& position,
    thread float3& velocity,
    thread float3x3& velocityGradient,
    constant ComputeShaderUniforms& uniforms,
    float dt,
    float friction,
    float projectionThreshold
) {
    float3 boundaryMin = uniforms.boundaryMin;
    float3 boundaryMax = uniforms.boundaryMax;

    // Find closest boundary and compute SDF
    float3 distToMin = position - boundaryMin;
    float3 distToMax = boundaryMax - position;

    float minDist = min(min(min(distToMin.x, distToMin.y), distToMin.z),
                       min(min(distToMax.x, distToMax.y), distToMax.z));

    float3 boundaryNormal;
    float sdf;

    // Determine closest boundary and its normal
    if (minDist == distToMin.x) {
        boundaryNormal = float3(-1.0, 0.0, 0.0);
        sdf = distToMin.x;
    } else if (minDist == distToMax.x) {
        boundaryNormal = float3(1.0, 0.0, 0.0);
        sdf = distToMax.x;
    } else if (minDist == distToMin.y) {
        boundaryNormal = float3(0.0, -1.0, 0.0);
        sdf = distToMin.y;
    } else if (minDist == distToMax.y) {
        boundaryNormal = float3(0.0, 1.0, 0.0);
        sdf = distToMax.y;
    } else if (minDist == distToMin.z) {
        boundaryNormal = float3(0.0, 0.0, -1.0);
        sdf = distToMin.z;
    } else {
        boundaryNormal = float3(0.0, 0.0, 1.0);
        sdf = distToMax.z;
    }

    // Check penetration at end of timestep
    float voxelSize = uniforms.gridSpacing;
    float sdf_end = sdf + projectionThreshold * voxelSize;

    if (sdf_end < 0.0) {
        // Collision detected - apply Coulomb friction response
        float3 boundaryVel = float3(0.0);  // Static boundary
        float3 relativeVel = velocity - boundaryVel;

        // Solve for friction-corrected velocity
        float3 deltaVel = solveCoulombIsotropic(friction, boundaryNormal, relativeVel);
        deltaVel += boundaryVel - velocity;

        // Update velocity and position
        velocity += deltaVel;
        position += deltaVel * dt;

        // Project position outside boundary
        position -= min(0.0, sdf_end + dt * dot(deltaVel, boundaryNormal)) * boundaryNormal;

        // Rigidify velocity gradient (keep only antisymmetric part = rotation)
        velocityGradient = 0.5 * (velocityGradient - transpose(velocityGradient));
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
                float3 cell_dist = ((float3(cell_idx) + 0.5f) - position) * uniforms.gridSpacing;
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
                    
                    float3 cell_dist = ((float3(cell_idx) + 0.5) - position) * uniforms.gridSpacing;
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
                    
                    float3 cell_dist = ((float3(cell_idx) + 0.5) - position) * uniforms.gridSpacing;
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
    velocity.y += uniforms.gravity * uniforms.deltaTime * uniforms.gridSpacing;
    
    // Get grid coordinates
    uint3 xyz = gridXYZ(id,uniforms);
    float3 zeroMin = uniforms.boundaryMin + float3(1.0,1.0,1.0) * uniforms.gridSpacing;
    float3 zeroMax = uniforms.boundaryMax - float3(2.0,2.0,2.0) * uniforms.gridSpacing;
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
                            constant SDFSet& sdfSet [[buffer(3)]],
                            device SDFPhysicsState* physicsStates [[buffer(4)]],
                            constant uint& sdfCount [[buffer(5)]],
                            uint id [[thread_position_in_grid]],
                            uint threadgroup_id [[threadgroup_position_in_grid]],
                            uint thread_id [[thread_position_in_threadgroup]],
                            uint threads_per_threadgroup [[threads_per_threadgroup]]
                            ) {
    // Threadgroup memory for accumulating SDF physics
    threadgroup float3 tg_impulse[MAX_COLLISION_SDF];
    threadgroup float3 tg_torque[MAX_COLLISION_SDF];
    
    // Initialize threadgroup memory
    if (thread_id < sdfCount) {
        tg_impulse[thread_id] = float3(0.0);
        tg_torque[thread_id] = float3(0.0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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
                
                float3 cell_dist = ((float3(cell_idx) + 0.5f) - position) * uniforms.gridSpacing;
                
                float3x3 term = float3x3(weighted_velocity * cell_dist.x,weighted_velocity * cell_dist.y,weighted_velocity * cell_dist.z);
                B += term;
            }
        }
    }
    // C (Affine matrix) clamp processing
    // Scale factor for grid spacing - B already includes gridSpacing scaling
    float3x3 unclampedC = B * 4.0f / (uniforms.gridSpacing * uniforms.gridSpacing);
    particles[id].C = unclampedC;
    particles[id].velocity = new_velocity;
    particles[id].position += particles[id].velocity * uniforms.deltaTime;
    
    for (uint i = 0; i < sdfCount; i++) {
        constant CollisionUniforms& collision = *sdfSet.collision[i];
        texture3d<float> sdfTexture = sdfSet.sdf[i];
        float3 collisionImpulse = computeParticleSDFCollisionImpulse(
             particles[id].position,
             particles[id].velocity,
             particles[id].mass,
             sdfTexture,
             uniforms,
             collision,
             uniforms.deltaTime
        );
        float3 J = particles[id].mass * collisionImpulse; // momentum imparted to particle
        if (any(isfinite(J))) {
            // Apply collision impulse to velocity
            particles[id].velocity += collisionImpulse;
            
            float3 negJ = -J; // equal and opposite to SDF object
            
            // Compute Center of Mass in world space using provided mass center
            float3 com = worldSDFMassCenter(collision);
            
            float3 r = particles[id].position - com;
            float3 torque = cross(r, negJ);
            
            // Only accumulate physics if SDF needs physics accumulation
            if (i < MAX_COLLISION_SDF && needsPhysicsAccumulation(collision)) {
                tg_impulse[i] += negJ;
                tg_torque[i] += torque;
            }
        }
        float impulseStrength = length(collisionImpulse);
        if (impulseStrength > 0.1) {
            float3 reactionDirection = -normalize(collisionImpulse);
            
            // Method 2: Velocity field modification to simulate SDF object movement
            // Add a velocity contribution that simulates the SDF object moving away
            float3 sdfVelocityContribution = reactionDirection * impulseStrength * 0.2;
            
            // Apply this as an additional velocity component that decays over time
            // This simulates the SDF object having momentum from the collision
            particles[id].velocity += sdfVelocityContribution * (1.0 - uniforms.deltaTime * 2.0);
        }
    }
    
    // Newton-style boundary collision with Coulomb friction
    const float friction = 0.1;  // Fluid friction coefficient
    const float projectionThreshold = 0.5;  // Projection threshold (fraction of voxel size)

    // Use local variables for thread reference
    float3 tmpPos = particles[id].position;
    float3 tmpVel = particles[id].velocity;
    float3x3 tmpC = particles[id].C;

    projectOutsideBoundary(
        tmpPos,
        tmpVel,
        tmpC,
        uniforms,
        uniforms.deltaTime,
        friction,
        projectionThreshold
    );

    particles[id].position = tmpPos;
    particles[id].velocity = tmpVel;
    particles[id].C = tmpC;

    // Final position clamping as safety net
    float3 safetyMargin = float3(0.5) * uniforms.gridSpacing;
    particles[id].position = clamp(particles[id].position,
                                  uniforms.boundaryMin + safetyMargin,
                                  uniforms.boundaryMax - safetyMargin);
    
    // Flush threadgroup accumulations to device memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id < sdfCount) {
        // Only flush if SDF needs physics accumulation
        constant CollisionUniforms& collision = *sdfSet.collision[thread_id];
        if (needsPhysicsAccumulation(collision)) {
            device SDFPhysicsState* phy = &physicsStates[thread_id];
            atomicAddWithUniform(&((*phy).impulse_x), tg_impulse[thread_id].x, uniforms);
            atomicAddWithUniform(&((*phy).impulse_y), tg_impulse[thread_id].y, uniforms);
            atomicAddWithUniform(&((*phy).impulse_z), tg_impulse[thread_id].z, uniforms);
            atomicAddWithUniform(&((*phy).torque_x), tg_torque[thread_id].x, uniforms);
            atomicAddWithUniform(&((*phy).torque_y), tg_torque[thread_id].y, uniforms);
            atomicAddWithUniform(&((*phy).torque_z), tg_torque[thread_id].z, uniforms);
        }
    }
}

// Neo-Hookean Elastic Grid to Particle Transfer (G2P)
kernel void gridToParticlesElastic(
                                  device MPMParticle* particles [[buffer(0)]],
                                  constant ComputeShaderUniforms& uniforms [[buffer(1)]],
                                  device const NonAtomicMPMGridNode* grid [[buffer(2)]],
                                  constant SDFSet& sdfSet [[buffer(3)]],
                                  device SDFPhysicsState* physicsStates [[buffer(4)]],
                                  constant uint& sdfCount [[buffer(5)]],
                                  uint id [[thread_position_in_grid]],
                                  uint threadgroup_id [[threadgroup_position_in_grid]],
                                  uint thread_id [[thread_position_in_threadgroup]],
                                  uint threads_per_threadgroup [[threads_per_threadgroup]]
                                  ) {
    // Threadgroup memory for accumulating SDF physics
    threadgroup float3 tg_impulse[MAX_COLLISION_SDF];
    threadgroup float3 tg_torque[MAX_COLLISION_SDF];
    
    // Initialize threadgroup memory
    if (thread_id < sdfCount) {
        tg_impulse[thread_id] = float3(0.0);
        tg_torque[thread_id] = float3(0.0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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
                
                float3 cell_dist = ((float3(cell_idx) + 0.5) - position) * uniforms.gridSpacing;
                
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
    // Scale factor for grid spacing - B already includes gridSpacing scaling
    float3x3 velocityGradient = B * 4.0 / (uniforms.gridSpacing * uniforms.gridSpacing);
    
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
    
    for (uint i = 0; i < sdfCount; i++) {
        constant CollisionUniforms& collision = *sdfSet.collision[i];
        texture3d<float> sdfTexture = sdfSet.sdf[i];
        float3 collisionImpulse = computeParticleSDFCollisionImpulse(
             particles[id].position,
             particles[id].velocity,
             particles[id].mass,
             sdfTexture,
             uniforms,
             collision,
             uniforms.deltaTime
        );
        // Apply collision impulse to velocity
        particles[id].velocity += collisionImpulse;
        float3 J = particles[id].mass * collisionImpulse;
        if (any(isfinite(J))) {
            float3 negJ = -J;
            
            // Compute Center of Mass in world space using provided mass center
            float3 com = worldSDFMassCenter(collision);
            
            float3 r = particles[id].position - com;
            float3 torque = cross(r, negJ);
            
            // Only accumulate physics if SDF needs physics accumulation
            if (i < MAX_COLLISION_SDF && needsPhysicsAccumulation(collision)) {
                tg_impulse[i] += negJ;
                tg_torque[i] += torque;
            }
        }
        float impulseStrength = length(collisionImpulse);
        if (impulseStrength > 0.1) {
            float3 reactionDirection = -normalize(collisionImpulse);
            
            // Method 2: Velocity field modification to simulate SDF object movement
            // Add a velocity contribution that simulates the SDF object moving away
            float3 sdfVelocityContribution = reactionDirection * impulseStrength * 0.2;
            
            // Apply this as an additional velocity component that decays over time
            // This simulates the SDF object having momentum from the collision
            particles[id].velocity += sdfVelocityContribution * (1.0 - uniforms.deltaTime * 2.0);
        }
    }

    // Newton-style boundary collision with Coulomb friction for elastic materials
    const float friction = 0.3;  // Higher friction for elastic materials
    const float projectionThreshold = 0.5;  // Projection threshold (fraction of voxel size)

    // Use local variables for thread reference
    float3 tmpPos = particles[id].position;
    float3 tmpVel = particles[id].velocity;
    float3x3 tmpC = particles[id].C;

    projectOutsideBoundary(
        tmpPos,
        tmpVel,
        tmpC,
        uniforms,
        uniforms.deltaTime,
        friction,
        projectionThreshold
    );

    particles[id].position = tmpPos;
    particles[id].velocity = tmpVel;
    particles[id].C = tmpC;

    // Position clamping - Essential safety net
    particles[id].position = clamp(particles[id].position,
                                  uniforms.boundaryMin + uniforms.gridSpacing,
                                  uniforms.boundaryMax - 2.0 * uniforms.gridSpacing);

    // Velocity clamping to prevent extreme values while allowing strong elastic motion
    const float max_velocity = 35.0;  // Higher limit for strong elastic response
    particles[id].velocity = clamp(particles[id].velocity,
                                  float3(-max_velocity),
                                  float3(max_velocity));
    
    // Flush threadgroup accumulations to device memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id < sdfCount) {
        // Only flush if SDF needs physics accumulation
        constant CollisionUniforms& collision = *sdfSet.collision[thread_id];
        if (needsPhysicsAccumulation(collision)) {
            device SDFPhysicsState* phy = &physicsStates[thread_id];
            atomicAddWithUniform(&((*phy).impulse_x), tg_impulse[thread_id].x, uniforms);
            atomicAddWithUniform(&((*phy).impulse_y), tg_impulse[thread_id].y, uniforms);
            atomicAddWithUniform(&((*phy).impulse_z), tg_impulse[thread_id].z, uniforms);
            atomicAddWithUniform(&((*phy).torque_x), tg_torque[thread_id].x, uniforms);
            atomicAddWithUniform(&((*phy).torque_y), tg_torque[thread_id].y, uniforms);
            atomicAddWithUniform(&((*phy).torque_z), tg_torque[thread_id].z, uniforms);
        }
    }
}


// Apply force to grid nodes within radius (with collision detection)
kernel void applyForceToGrid(
    device MPMGridNode* grid [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    constant float3& forcePosition [[buffer(2)]],
    constant float3& forceVector [[buffer(3)]],
    constant float& forceRadius [[buffer(4)]],
    constant SDFSet& sdfSet [[buffer(5)]],
    constant uint& sdfCount [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.gridNodeCount) return;
    
    // Get grid coordinates from linear index
    uint3 gridCoord = gridXYZ(id, uniforms);
    
    // Force position should already be in grid coordinates (converted in Swift)
    float distance = length(float3(gridCoord) - forcePosition);
    
    // Force radius should already be in grid units (converted in Swift)
    if (distance < forceRadius) {
        float mass = atomicLoadWithUniform(&grid[id].mass, uniforms);
        if (mass > 1e-6) {  // Only apply force to grid nodes with mass
            
            // Check for collision at this grid position
            bool isInsideCollider = false;
            for (uint i = 0; i < sdfCount && !isInsideCollider; i++) {
                constant CollisionUniforms& collision = *sdfSet.collision[i];
                texture3d<float> sdfTexture = sdfSet.sdf[i];
                
                // Skip if collision is disabled
                if (!isCollisionEnabled(collision)) continue;
                
                // Sample SDF at grid position (pass grid coordinate, not world position)
                // sampleSDFWithGradient will apply worldTransform internally
                float3 gridPos = float3(gridCoord);
                float4 sdfData = sampleSDFWithGradient(gridPos, sdfTexture, uniforms, collision);
                float phi = sdfData.x;
                
                // If inside SDF (phi < 0), this is a collider region
                if (phi < 0.0) {
                    isInsideCollider = true;
                }
            }
            
            // Only apply force if not inside a collider
            if (!isInsideCollider) {
                float falloff = exp(-distance * 8.0);
                float3 force = forceVector * falloff * uniforms.deltaTime;
                
                // Add force as velocity impulse
                atomicAddWithUniform(&grid[id].velocity_x, force.x, uniforms);
                atomicAddWithUniform(&grid[id].velocity_y, force.y, uniforms);
                atomicAddWithUniform(&grid[id].velocity_z, force.z, uniforms);
            }
        }
    }
}
