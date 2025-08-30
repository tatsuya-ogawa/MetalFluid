//
//  MPMTypes.swift
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/08/30.
//
import simd
struct Triangle {
    var v0: SIMD3<Float>
    var v1: SIMD3<Float>
    var v2: SIMD3<Float>
}
struct CollisionUniforms {
    var sdfOrigin: SIMD3<Float>
    var sdfSize: SIMD3<Float>
    var sdfResolution: SIMD3<Int32>
    var collisionStiffness: Float
    var collisionDamping: Float
    var enableCollision: UInt32
    var sdfMass: Float
    var collisionTransform: float4x4
    var collisionInvTransform: float4x4
}
