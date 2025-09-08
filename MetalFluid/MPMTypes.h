//
//  MPMTypes.h
//  MetalFluid
//
//  Shared types between Swift and Metal for MLS-MPM simulation
//

#ifndef MPMTypes_h
#define MPMTypes_h

#include <simd/simd.h>
#define MPM_ATOMIC_FLOAT metal::atomic<float>
#define MPM_NON_ATOMIC_FLOAT float

// MLS-MPM particle struct (shared between Swift and Metal)
typedef struct {
    float3 position;        // Position
    float3 velocity;        // Velocity
    metal::float3x3 C;            // Affine momentum matrix
    float mass;           // Mass
    uint32_t originalIndex; // Original particle index (used to fetch unsorted rigid info)
    // Rigid-related fields moved to separate struct MPMParticleRigidInfo
} MPMParticle;

// Rigid-related particle info stored separately for compact particle arrays
typedef struct {
    uint32_t rigidId;     // Rigid body ID (0 = no rigid body, >0 = rigid body index)
    float3 initialOffset; // Initial relative position from rigid body center of mass
} MPMParticleRigidInfo;

// MLS-MPM grid node struct (shared between Swift and Metal)
typedef struct {
    MPM_ATOMIC_FLOAT velocity_x;      // atomic<float> velocity_x (Metal) / float (Swift)
    MPM_ATOMIC_FLOAT velocity_y;      // atomic<float> velocity_y (Metal) / float (Swift)
    MPM_ATOMIC_FLOAT velocity_z;      // atomic<float> velocity_z (Metal) / float (Swift)
    MPM_ATOMIC_FLOAT mass;           // atomic<float> mass (Metal) / float (Swift)
} MPMGridNode;
typedef struct {
    MPM_NON_ATOMIC_FLOAT velocity_x;      // atomic<float> velocity_x (Metal) / float (Swift)
    MPM_NON_ATOMIC_FLOAT velocity_y;      // atomic<float> velocity_y (Metal) / float (Swift)
    MPM_NON_ATOMIC_FLOAT velocity_z;      // atomic<float> velocity_z (Metal) / float (Swift)
    MPM_NON_ATOMIC_FLOAT mass;           // atomic<float> mass (Metal) / float (Swift)
} NonAtomicMPMGridNode;

// Accumulator for SDF rigid-body impulses aggregated from particle collisions
// Note: uses atomic floats on GPU; in Swift, treat as plain floats when reading
typedef struct {
    MPM_ATOMIC_FLOAT impulse_x;
    MPM_ATOMIC_FLOAT impulse_y;
    MPM_ATOMIC_FLOAT impulse_z;
    MPM_ATOMIC_FLOAT torque_x;
    MPM_ATOMIC_FLOAT torque_y;
    MPM_ATOMIC_FLOAT torque_z;
} SDFImpulseAccumulator;

// Unified per-SDF physics state (dynamic) to combine accumulator and basic kinematics
typedef struct {
    // Atomic accumulators written by particle collisions (device memory)
    MPM_ATOMIC_FLOAT impulse_x;
    MPM_ATOMIC_FLOAT impulse_y;
    MPM_ATOMIC_FLOAT impulse_z;
    MPM_ATOMIC_FLOAT torque_x;
    MPM_ATOMIC_FLOAT torque_y;
    MPM_ATOMIC_FLOAT torque_z;

    // Dynamic kinematics (read/write by CPU integration; GPU may read if needed)
    simd_float3 linearVelocity;
    simd_float3 angularVelocity;
} SDFPhysicsState;

// Compute shader specific uniforms
typedef struct {
    float deltaTime;
    uint32_t particleCount;
    float gravity;
    float gridSpacing;
    simd_float3 domainOrigin;   // World-space origin of the simulation domain (maps to grid (0,0,0))
    simd_int3 gridResolution;  // Grid resolution in cells (x,y,z)
    uint32_t gridNodeCount;  // Total number of grid nodes (x*y*z)
    simd_float3 boundaryMin; // Simulation boundary min (world coordinates)
    simd_float3 boundaryMax; // Simulation boundary max (world coordinates)
    float stiffness;       // Material stiffness parameter
    float rest_density;    // Rest density for pressure calculation
    float dynamic_viscosity; // Dynamic viscosity parameter
    float massScale;       // Mass scaling factor for particles
    uint32_t timeSalt;
    uint32_t materialMode;     // 0: fluid, 1: neo-hookean elastic, 2: rigid body
    float youngsModulus;       // Young's modulus for elastic material
    float poissonsRatio;       // Poisson's ratio for elastic material
    simd_float4x4 worldTransform; // Transform from fluid space to world space
} ComputeShaderUniforms;

// Collision detection uniforms
typedef struct {
    simd_float3 sdfOrigin;        // SDF volume origin in world space
    simd_float3 sdfSize;          // SDF volume size
    simd_float3 sdfMassCenter;    // SDF mass center in mesh/local space
    simd_int3 sdfResolution;      // SDF texture resolution
    float collisionStiffness;     // Collision response strength
    float collisionDamping;       // Velocity damping on collision
    uint32_t collisionFlags;      // Bit flags: bit 0 = enableCollision, bit 1 = isStaticSDF
    float sdfMass;                // Effective mass of SDF rigid object
    simd_float4x4 collisionTransform;  // Transform matrix (scale, rotation, translation)
    simd_float4x4 collisionInvTransform; // Inverse transform matrix for world->mesh space
} CollisionUniforms;

// Vertex shader specific uniforms
typedef struct {
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewMatrix;
    float gridSpacing;
    simd_float3 physicalDomainOrigin;   // For physics calculations
    simd_int3 gridResolution;
    float rest_density;    // For pressure heatmap calculation
    float particleSizeMultiplier; // For dynamic particle size scaling
} VertexShaderUniforms;

// Fluid rendering uniforms (same as WebGPU-Ocean)
typedef struct {
    simd_float2 texelSize;
    float sphereSize;
    simd_float4x4 invProjectionMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewMatrix;
    simd_float4x4 invViewMatrix;
} FluidRenderUniforms;

// Collision mesh rendering uniforms
typedef struct {
    simd_float4 meshColor;
} CollisionMeshUniforms;

// Triangle structure
struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
};
constant float SPHERE_SIZE_BASE = 0.025;
#endif /* MPMTypes_h */
