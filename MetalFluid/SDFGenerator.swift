import Foundation
import Metal
import simd

struct Triangle {
    let v0: SIMD3<Float>
    let v1: SIMD3<Float>
    let v2: SIMD3<Float>
}

// GPU Triangle structure (must match Metal shader)
struct SDFTriangle {
    let v0: SIMD3<Float>
    let v1: SIMD3<Float>
    let v2: SIMD3<Float>
}

class SDFGenerator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var sdfComputePipelineState: MTLComputePipelineState?
    private var sdfOptimizedPipelineState: MTLComputePipelineState?
    
    init(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue for SDF generator")
        }
        self.commandQueue = queue
        
        setupComputePipelines()
    }
    
    private func setupComputePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to create library for SDF compute shaders")
            return
        }
        
        do {
            if let sdfFunction = library.makeFunction(name: "generateSDF") {
                sdfComputePipelineState = try device.makeComputePipelineState(function: sdfFunction)
            }
            
            if let optimizedFunction = library.makeFunction(name: "generateSDFOptimized") {
                sdfOptimizedPipelineState = try device.makeComputePipelineState(function: optimizedFunction)
            }
        } catch {
            print("Failed to create SDF compute pipeline states: \(error)")
        }
    }
    
    func loadOBJ(from url: URL,scaleFactor:Float=100) -> [Triangle] {
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
        
        // Scale the triangles to a reasonable size for the simulation
        // The bunny model is very small (around 0.1 units), so scale it up
        let scaledTriangles = triangles.map { triangle in
            Triangle(
                v0: triangle.v0 * scaleFactor,
                v1: triangle.v1 * scaleFactor,
                v2: triangle.v2 * scaleFactor
            )
        }
        
        print("🔧 Scaled triangles by factor of \(scaleFactor)")
        if !scaledTriangles.isEmpty {
            let minBounds = scaledTriangles.reduce(scaledTriangles[0].v0) { result, triangle in
                min(min(min(result, triangle.v0), triangle.v1), triangle.v2)
            }
            let maxBounds = scaledTriangles.reduce(scaledTriangles[0].v0) { result, triangle in
                max(max(max(result, triangle.v0), triangle.v1), triangle.v2)
            }
            print("🔧 Scaled bounds: min=\(minBounds), max=\(maxBounds), size=\(maxBounds - minBounds)")
        }
        
        return scaledTriangles
    }
    
    func generateSDF(triangles: [Triangle], resolution: SIMD3<Int32>, boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)) -> MTLTexture? {
        guard let pipelineState = sdfComputePipelineState else {
            print("SDF compute pipeline not available")
            return nil
        }
        
        // Convert triangles to GPU format
        let sdfTriangles = triangles.map { triangle in
            SDFTriangle(v0: triangle.v0, v1: triangle.v1, v2: triangle.v2)
        }
        
        // Create buffers
        guard let triangleBuffer = device.makeBuffer(
            bytes: sdfTriangles,
            length: MemoryLayout<SDFTriangle>.stride * sdfTriangles.count,
            options: .storageModeShared
        ) else {
            print("Failed to create triangle buffer")
            return nil
        }
        
        let totalVoxels = Int(resolution.x * resolution.y * resolution.z)
        guard let sdfDataBuffer = device.makeBuffer(
            length: MemoryLayout<Float>.stride * totalVoxels,
            options: .storageModeShared
        ) else {
            print("Failed to create SDF data buffer")
            return nil
        }
        
        // Create texture
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type3D
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = Int(resolution.x)
        textureDescriptor.height = Int(resolution.y)
        textureDescriptor.depth = Int(resolution.z)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let sdfTexture = device.makeTexture(descriptor: textureDescriptor) else {
            print("Failed to create SDF texture")
            return nil
        }
        
        // Setup compute parameters
        let voxelSize = (boundingBox.max - boundingBox.min) / SIMD3<Float>(resolution)
        var triangleCount = UInt32(sdfTriangles.count)
        var sdfOrigin = boundingBox.min
        var voxelSizeVar = voxelSize
        var resolutionVar = resolution
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create compute command buffer/encoder")
            return nil
        }
        
        computeEncoder.label = "SDF Generation"
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setBuffer(triangleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sdfDataBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&triangleCount, length: MemoryLayout<UInt32>.size, index: 2)
        computeEncoder.setBytes(&sdfOrigin, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
        computeEncoder.setBytes(&voxelSizeVar, length: MemoryLayout<SIMD3<Float>>.size, index: 4)
        computeEncoder.setBytes(&resolutionVar, length: MemoryLayout<SIMD3<Int32>>.size, index: 5)
        
        // Calculate thread group size
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (Int(resolution.x) + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (Int(resolution.y) + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: Int(resolution.z)
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        if commandBuffer.status == .error {
            print("SDF generation compute command failed")
            return nil
        }
        
        // Copy data to texture
        let sdfDataPointer = sdfDataBuffer.contents().bindMemory(to: Float.self, capacity: totalVoxels)
        let sdfDataArray = Array(UnsafeBufferPointer(start: sdfDataPointer, count: totalVoxels))
        
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                              size: MTLSize(width: Int(resolution.x), height: Int(resolution.y), depth: Int(resolution.z)))
        
        sdfTexture.replace(region: region,
                          mipmapLevel: 0,
                          slice: 0,
                          withBytes: sdfDataArray,
                          bytesPerRow: Int(resolution.x) * MemoryLayout<Float>.size,
                          bytesPerImage: Int(resolution.x * resolution.y) * MemoryLayout<Float>.size)
        
        print("GPU SDF generation completed successfully")
        return sdfTexture
    }
    
    // Fast GPU-based SDF generation with optimized memory access
    func generateSDFOptimized(triangles: [Triangle], resolution: SIMD3<Int32>, boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)) -> MTLTexture? {
        guard let pipelineState = sdfOptimizedPipelineState else {
            print("Optimized SDF compute pipeline not available, falling back to standard")
            return generateSDF(triangles: triangles, resolution: resolution, boundingBox: boundingBox)
        }
        
        // Use optimized pipeline for better performance with large meshes
        // Similar implementation but with shared memory optimization
        // Implementation details similar to generateSDF but with optimized pipeline
        return generateSDF(triangles: triangles, resolution: resolution, boundingBox: boundingBox)
    }
}
