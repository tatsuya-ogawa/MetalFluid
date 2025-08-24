import Foundation
import Metal
import MetalKit
import simd

struct Triangle {
    var v0: SIMD3<Float>
    var v1: SIMD3<Float>
    var v2: SIMD3<Float>
}

class MeshLoader {
    private let scaleFactor: Float
    
    init(scaleFactor: Float = 15.0) {
        self.scaleFactor = scaleFactor
    }
    
    /// Load triangles from OBJ file with optional offset
    func loadOBJ(from url: URL, offsetToBottom: SIMD3<Float>?) -> [Triangle] {
        guard let content = try? String(contentsOf: url) else {
            print("Failed to load OBJ file from \(url)")
            return []
        }
        
        var vertices: [SIMD3<Float>] = []
        var triangles: [Triangle] = []
        
        let lines = content.components(separatedBy: .newlines)
        
        for line in lines {
            let components = line.trimmingCharacters(in: .whitespaces).components(separatedBy: " ")
            
            if components.first == "v" && components.count >= 4 {
                // Vertex line: v x y z
                if let x = Float(components[1]),
                   let y = Float(components[2]),
                   let z = Float(components[3]) {
                    vertices.append(SIMD3<Float>(x, y, z))
                }
            } else if components.first == "f" && components.count >= 4 {
                // Face line: f v1 v2 v3 [v4]
                var indices: [Int] = []
                for i in 1..<components.count {
                    let component = components[i]
                    // Handle "v1/vt1/vn1" format by taking only the vertex index
                    let vertexIndexString = component.components(separatedBy: "/")[0]
                    if let index = Int(vertexIndexString), index > 0 && index <= vertices.count {
                        indices.append(index)
                    }
                }
                
                // Convert face to triangles
                if indices.count >= 3 {
                    // Triangle: v1, v2, v3
                    let v0 = vertices[indices[0] - 1]
                    let v1 = vertices[indices[1] - 1]
                    let v2 = vertices[indices[2] - 1]
                    triangles.append(Triangle(v0: v0, v1: v1, v2: v2))
                    
                    // If quad: add second triangle v1, v3, v4
                    if indices.count >= 4 {
                        let v3 = vertices[indices[3] - 1]
                        triangles.append(Triangle(v0: v0, v1: v2, v2: v3))
                    }
                }
            }
        }
        
        // Scale the triangles to a reasonable size for the simulation
        var scaledTriangles = triangles.map { triangle in
            Triangle(
                v0: triangle.v0 * scaleFactor,
                v1: triangle.v1 * scaleFactor,
                v2: triangle.v2 * scaleFactor
            )
        }
        
        // Apply offset if provided
        if let offset = offsetToBottom {
            // Calculate current bounding box
            var minBounds = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
            var maxBounds = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
            
            for triangle in scaledTriangles {
                minBounds = min(minBounds, triangle.v0)
                minBounds = min(minBounds, triangle.v1)
                minBounds = min(minBounds, triangle.v2)
                maxBounds = max(maxBounds, triangle.v0)
                maxBounds = max(maxBounds, triangle.v1)
                maxBounds = max(maxBounds, triangle.v2)
            }
            
            // Calculate translation needed to move mesh bottom to desired position
            let translation = offset - SIMD3<Float>((minBounds.x + maxBounds.x) * 0.5, minBounds.y, (minBounds.z + maxBounds.z) * 0.5)
            
            // Apply translation
            scaledTriangles = scaledTriangles.map { triangle in
                Triangle(
                    v0: triangle.v0 + translation,
                    v1: triangle.v1 + translation,
                    v2: triangle.v2 + translation
                )
            }
        }
        
        print("Loaded \(scaledTriangles.count) triangles from OBJ file")
        return scaledTriangles
    }
    
    /// Calculate bounding box for given triangles
    func calculateBoundingBox(triangles: [Triangle]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard !triangles.isEmpty else {
            return (min: SIMD3<Float>(0, 0, 0), max: SIMD3<Float>(0, 0, 0))
        }
        
        var minBounds = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxBounds = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
        
        for triangle in triangles {
            minBounds = min(minBounds, triangle.v0)
            minBounds = min(minBounds, triangle.v1)
            minBounds = min(minBounds, triangle.v2)
            maxBounds = max(maxBounds, triangle.v0)
            maxBounds = max(maxBounds, triangle.v1)
            maxBounds = max(maxBounds, triangle.v2)
        }
        
        return (min: minBounds, max: maxBounds)
    }
}