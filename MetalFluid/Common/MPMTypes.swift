//
//  MPMTypes.swift
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/08/30.
//
import simd

// MARK: - Collision Types
struct Triangle {
    var v0: SIMD3<Float>
    var v1: SIMD3<Float>
    var v2: SIMD3<Float>
}

struct CollisionVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
}

struct CollisionUniforms {
    var sdfOrigin: SIMD3<Float>
    var sdfSize: SIMD3<Float>
    var sdfMassCenter: SIMD3<Float>
    var sdfResolution: SIMD3<Int32>
    var collisionStiffness: Float
    var collisionDamping: Float
    var collisionFlags: UInt32  // Bit flags: bit 0 = enableCollision, bit 1 = isStaticSDF
    var sdfMass: Float
    var collisionTransform: float4x4
    var collisionInvTransform: float4x4
}

struct CollisionMeshUniforms {
    var meshColor: simd_float4
}

struct ARMeshUniforms {
    var modelViewProjectionMatrix: simd_float4x4
    var modelMatrix: simd_float4x4
    var normalMatrix: simd_float4x4
    var meshColor: SIMD4<Float>
    var opacity: Float
}

struct RaycastUniforms {
    var rayOrigin: SIMD3<Float>
    var rayDirection: SIMD3<Float>
    var triangleCount: UInt32
    var boundingBoxCenter: SIMD3<Float>
    var boundingBoxSize: SIMD3<Float>
}

struct RaycastResult {
    var hitPoint: SIMD3<Float>
    var distance: Float
    var triangleIndex: UInt32
    var hit: UInt32  // Use UInt32 instead of Bool for Metal compatibility
    var _padding: SIMD3<UInt32>  // Ensure proper alignment
}

// MARK: - MPM Core Types
struct MPMParticle {
    var position: SIMD3<Float>  // Position
    var velocity: SIMD3<Float>  // Velocity
    var C: simd_float3x3  // Affine momentum matrix (3D)
    var mass: Float  // Mass
    var originalIndex: UInt32 // Original particle index used to fetch unsorted rigid info
}

struct MPMGridNode {
    var velocity_x: Float  // atomic<float> on GPU
    var velocity_y: Float  // atomic<float> on GPU
    var velocity_z: Float  // atomic<float> on GPU
    var mass: Float  // atomic<float> on GPU
}

// MARK: - Shader Uniforms
struct ComputeShaderUniforms {
    var deltaTime: Float
    var particleCount: UInt32
    var gravity: Float
    var gridSpacing: Float
    var domainOrigin: SIMD3<Float>
    var gridResolution: SIMD3<Int32>
    var gridNodeCount: UInt32
    var boundaryMin: SIMD3<Float>
    var boundaryMax: SIMD3<Float>
    var stiffness: Float
    var rest_density: Float
    var dynamic_viscosity: Float
    var massScale: Float
    var timeSalt: UInt32
    var materialMode: UInt32  // 0: fluid, 1: neo-hookean elastic, 2: rigid body
    var youngsModulus: Float  // Young's modulus for elastic material
    var poissonsRatio: Float  // Poisson's ratio for elastic material
}

struct VertexShaderUniforms {
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var gridSpacing: Float
    var physicalDomainOrigin: SIMD3<Float>   // For physics calculations
    var gridResolution: SIMD3<Int32>
    var rest_density: Float
    var particleSizeMultiplier: Float
    var sphere_size: Float
}

struct FilterUniforms {
    var direction: SIMD2<Float>
    var screenSize: SIMD2<Float>
    var depthThreshold: Float
    var filterRadius: Int32
    var projectedParticleConstant: Float
    var maxFilterSize: Float
}

struct FluidRenderUniforms {
    var texelSize: SIMD2<Float>
    var sphereSize: Float
    var invProjectionMatrix: float4x4
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var invViewMatrix: float4x4
}

struct GaussianUniforms {
    var direction: SIMD2<Float>
    var screenSize: SIMD2<Float>
    var filterRadius: Int32
}

// MARK: - Utility Types
struct SortKey {
    var key: UInt32      // Grid index as sort key
    var value: UInt32    // Original particle index
}

// MARK: - Enums
enum RenderMode {
    case particles
    case water
}

enum MaterialMode {
    case fluid
    case neoHookeanElastic
}

enum ParticleRenderMode {
    case pressureHeatmap
}

enum SortingAlgorithm {
    case none
    case bitonicSort
    case radixSort
}

// --- SDF rigid-body impulse aggregation & state ---
// Swift mirror of SDFPhysicsState (layout-compatible)
struct SDFPhysicsStateSwift {
    var impulse_x: Float
    var impulse_y: Float
    var impulse_z: Float
    var torque_x: Float
    var torque_y: Float
    var torque_z: Float
    var linearVelocity: SIMD3<Float>
    var angularVelocity: SIMD3<Float>
    var collisionFlags: UInt32  // Bit flags: bit 0 = enableCollision, bit 1 = isStaticSDF
    var _padding: SIMD3<Float>  // Ensure proper alignment
}
