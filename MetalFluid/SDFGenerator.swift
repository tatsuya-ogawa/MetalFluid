import Foundation
import Metal
import simd

struct Triangle {
    let v0: SIMD3<Float>
    let v1: SIMD3<Float>
    let v2: SIMD3<Float>
}

class SDFGenerator {
    private let device: MTLDevice
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    func loadOBJ(from url: URL) -> [Triangle] {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            print("Failed to load OBJ file")
            return []
        }
        
        var vertices: [SIMD3<Float>] = []
        var triangles: [Triangle] = []
        
        let lines = content.components(separatedBy: .newlines)
        
        for line in lines {
            let components = line.trimmingCharacters(in: .whitespaces).components(separatedBy: " ")
            
            if components[0] == "v" && components.count >= 4 {
                let x = Float(components[1]) ?? 0
                let y = Float(components[2]) ?? 0
                let z = Float(components[3]) ?? 0
                vertices.append(SIMD3<Float>(x, y, z))
            }
            else if components[0] == "f" && components.count >= 4 {
                // Simple triangulation for quads -> triangles
                let indices = components[1...].compactMap { component -> Int? in
                    let parts = component.components(separatedBy: "/")
                    return Int(parts[0])
                }
                
                if indices.count >= 3 {
                    let v0 = vertices[indices[0] - 1] // OBJ indices are 1-based
                    let v1 = vertices[indices[1] - 1]
                    let v2 = vertices[indices[2] - 1]
                    triangles.append(Triangle(v0: v0, v1: v1, v2: v2))
                    
                    // If quad, add second triangle
                    if indices.count >= 4 {
                        let v3 = vertices[indices[3] - 1]
                        triangles.append(Triangle(v0: v0, v1: v2, v2: v3))
                    }
                }
            }
        }
        
        return triangles
    }
    
    func generateSDF(triangles: [Triangle], resolution: SIMD3<Int32>, boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)) -> MTLTexture? {
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type3D
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = Int(resolution.x)
        textureDescriptor.height = Int(resolution.y)
        textureDescriptor.depth = Int(resolution.z)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let sdfTexture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        // CPU-based SDF generation
        let voxelSize = (boundingBox.max - boundingBox.min) / SIMD3<Float>(resolution)
        var sdfData: [Float] = Array(repeating: Float.greatestFiniteMagnitude, count: Int(resolution.x * resolution.y * resolution.z))
        
        for z in 0..<Int(resolution.z) {
            for y in 0..<Int(resolution.y) {
                for x in 0..<Int(resolution.x) {
                    let worldPos = boundingBox.min + SIMD3<Float>(Float(x), Float(y), Float(z)) * voxelSize
                    let distance = computeDistanceToMesh(point: worldPos, triangles: triangles)
                    
                    let index = x + y * Int(resolution.x) + z * Int(resolution.x) * Int(resolution.y)
                    sdfData[index] = distance
                }
            }
        }
        
        // Upload to texture
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                              size: MTLSize(width: Int(resolution.x), height: Int(resolution.y), depth: Int(resolution.z)))
        
        sdfTexture.replace(region: region,
                          mipmapLevel: 0,
                          slice: 0,
                          withBytes: sdfData,
                          bytesPerRow: Int(resolution.x) * MemoryLayout<Float>.size,
                          bytesPerImage: Int(resolution.x * resolution.y) * MemoryLayout<Float>.size)
        
        return sdfTexture
    }
    
    private func computeDistanceToMesh(point: SIMD3<Float>, triangles: [Triangle]) -> Float {
        var minDistance = Float.greatestFiniteMagnitude
        
        for triangle in triangles {
            let distance = distanceToTriangle(point: point, triangle: triangle)
            minDistance = min(minDistance, distance)
        }
        
        return minDistance
    }
    
    private func distanceToTriangle(point: SIMD3<Float>, triangle: Triangle) -> Float {
        let v0 = triangle.v0
        let v1 = triangle.v1
        let v2 = triangle.v2
        
        let v0v1 = v1 - v0
        let v0v2 = v2 - v0
        let v0p = point - v0
        
        let d00 = dot(v0v1, v0v1)
        let d01 = dot(v0v1, v0v2)
        let d11 = dot(v0v2, v0v2)
        let d20 = dot(v0p, v0v1)
        let d21 = dot(v0p, v0v2)
        
        let denom = d00 * d11 - d01 * d01
        let v = (d11 * d20 - d01 * d21) / denom
        let w = (d00 * d21 - d01 * d20) / denom
        let u = 1.0 - v - w
        
        if u >= 0 && v >= 0 && w >= 0 {
            // Point is inside triangle
            let closestPoint = u * v0 + v * v1 + w * v2
            return distance(point, closestPoint)
        } else {
            // Point is outside triangle, find closest point on edges or vertices
            let dist1 = distanceToLineSegment(point: point, a: v0, b: v1)
            let dist2 = distanceToLineSegment(point: point, a: v1, b: v2)
            let dist3 = distanceToLineSegment(point: point, a: v2, b: v0)
            return min(dist1, min(dist2, dist3))
        }
    }
    
    private func distanceToLineSegment(point: SIMD3<Float>, a: SIMD3<Float>, b: SIMD3<Float>) -> Float {
        let ab = b - a
        let ap = point - a
        let t = simd_clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0)
        let closestPoint = a + t * ab
        return distance(point, closestPoint)
    }
}