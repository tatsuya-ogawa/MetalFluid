import simd

// MARK: - SIMD Helpers
extension float4x4 {
    /// Transforms a 3D point (implicitly w=1) by this matrix and returns the xyz components.
    /// Equivalent to `self * float4(point, 1)`.
    func transformPoint(_ point: SIMD3<Float>) -> SIMD3<Float> {
        let p4 = SIMD4<Float>(point.x, point.y, point.z, 1.0)
        let r = self * p4
        return SIMD3<Float>(r.x, r.y, r.z)
    }
    
    /// Returns the upper-left 3x3 matrix (rotation/scale part)
    var upperLeft3x3: float3x3 {
        return float3x3(
            SIMD3<Float>(columns.0.x, columns.0.y, columns.0.z),
            SIMD3<Float>(columns.1.x, columns.1.y, columns.1.z),
            SIMD3<Float>(columns.2.x, columns.2.y, columns.2.z)
        )
    }
    
    /// Creates a scale matrix from a scale vector
    init(scale: SIMD3<Float>) {
        self = float4x4(
            SIMD4<Float>(scale.x, 0, 0, 0),
            SIMD4<Float>(0, scale.y, 0, 0),
            SIMD4<Float>(0, 0, scale.z, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
}

