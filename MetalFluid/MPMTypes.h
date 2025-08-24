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
//    float volume;         // Volume
//    float Jp;             // Plastic deformation determinant
//    float3 color;         // Rendering color
} MPMParticle;

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
} ComputeShaderUniforms;

// Collision detection uniforms
typedef struct {
    simd_float3 sdfOrigin;        // SDF volume origin in world space
    simd_float3 sdfSize;          // SDF volume size
    simd_int3 sdfResolution;      // SDF texture resolution
    float collisionStiffness;     // Collision response strength
    float collisionDamping;       // Velocity damping on collision
    uint32_t enableCollision;     // Enable/disable collision detection
    uint32_t fillMode;            // 0: surface collision, 1: fill inside
} CollisionUniforms;

// Vertex shader specific uniforms
typedef struct {
    simd_float4x4 mvpMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewMatrix;
    float gridSpacing;
    simd_float3 physicalDomainOrigin;   // For physics calculations
    simd_int3 gridResolution;
    float rest_density;    // For pressure heatmap calculation
    float particleSizeMultiplier; // For dynamic particle size scaling
    float sphere_size;     // Sphere size for billboard rendering (same as WebGPU-Ocean)
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

#endif /* MPMTypes_h */
