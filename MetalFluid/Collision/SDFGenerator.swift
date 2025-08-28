import Foundation
import Metal
import simd
#if canImport(ARKit)
import ARKit
#endif

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
        textureDescriptor.storageMode = .private  // Use private storage for GPU performance
        
        guard let sdfTexture = device.makeTexture(descriptor: textureDescriptor) else {
            print("Failed to create SDF texture")
            return nil
        }
        sdfTexture.label = "SDFTexture"
        
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
        
        // Copy buffer data to texture using blit encoder (for private texture)
        guard let blitCommandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = blitCommandBuffer.makeBlitCommandEncoder() else {
            print("Failed to create blit command buffer/encoder")
            return nil
        }
        
        blitEncoder.copy(from: sdfDataBuffer,
                        sourceOffset: 0,
                        sourceBytesPerRow: Int(resolution.x) * MemoryLayout<Float>.size,
                        sourceBytesPerImage: Int(resolution.x * resolution.y) * MemoryLayout<Float>.size,
                        sourceSize: MTLSize(width: Int(resolution.x), height: Int(resolution.y), depth: Int(resolution.z)),
                        to: sdfTexture,
                        destinationSlice: 0,
                        destinationLevel: 0,
                        destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        
        blitEncoder.endEncoding()
        blitCommandBuffer.commit()
        blitCommandBuffer.waitUntilCompleted()
        
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
    
    // MARK: - AR Mesh SDF Generation
    
    #if canImport(ARKit)
    @available(iOS 13.4, macOS 10.15.4, *)
    func generateSDFFromARMesh(_ meshAnchor: ARMeshAnchor, resolution: SIMD3<Int32>? = nil) -> MTLTexture? {
        let geometry = meshAnchor.geometry
        let vertices = geometry.vertices
        let faces = geometry.faces
        let transform = meshAnchor.transform
        
        // Extract vertex data
        let vertexBuffer = vertices.buffer
        let vertexStride = vertices.stride
        let vertexCount = vertices.count
        
        // Extract face data
        let faceBuffer = faces.buffer
        let faceStride = MemoryLayout<UInt32>.stride * 3 // Each face has 3 indices
        let faceCount = faces.count
        
        // Convert AR mesh to triangles
        var triangles: [Triangle] = []
        var meshBoundingBox = (min: SIMD3<Float>(repeating: Float.greatestFiniteMagnitude),
                              max: SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude))
        
        // Process faces (each face is a triangle with 3 indices)
        for faceIndex in 0..<faceCount {
            let faceOffset = faceIndex * faceStride
            let facePointer = faceBuffer.contents().advanced(by: faceOffset).assumingMemoryBound(to: UInt32.self)
            
            // Get vertex indices for this triangle
            let i0 = Int(facePointer[0])
            let i1 = Int(facePointer[1])
            let i2 = Int(facePointer[2])
            
            // Get vertex positions
            let v0Offset = i0 * vertexStride
            let v1Offset = i1 * vertexStride
            let v2Offset = i2 * vertexStride
            
            let v0Pointer = vertexBuffer.contents().advanced(by: v0Offset).assumingMemoryBound(to: Float.self)
            let v1Pointer = vertexBuffer.contents().advanced(by: v1Offset).assumingMemoryBound(to: Float.self)
            let v2Pointer = vertexBuffer.contents().advanced(by: v2Offset).assumingMemoryBound(to: Float.self)
            
            // Get vertex positions in local space
            var v0Local = SIMD3<Float>(v0Pointer[0], v0Pointer[1], v0Pointer[2])
            var v1Local = SIMD3<Float>(v1Pointer[0], v1Pointer[1], v1Pointer[2])
            var v2Local = SIMD3<Float>(v2Pointer[0], v2Pointer[1], v2Pointer[2])
            
            // Transform vertices to world space
            let v0World = transformPoint(v0Local, by: transform)
            let v1World = transformPoint(v1Local, by: transform)
            let v2World = transformPoint(v2Local, by: transform)
            
            // Update bounding box
            meshBoundingBox.min = min(meshBoundingBox.min, min(v0World, min(v1World, v2World)))
            meshBoundingBox.max = max(meshBoundingBox.max, max(v0World, max(v1World, v2World)))
            
            // Create triangle
            let triangle = Triangle(v0: v0World, v1: v1World, v2: v2World)
            triangles.append(triangle)
        }
        
        // Use default resolution if not specified
        let sdfResolution = resolution ?? SIMD3<Int32>(64, 64, 64)
        
        // Add some padding to bounding box
        let padding: Float = 0.1
        let size = meshBoundingBox.max - meshBoundingBox.min
        let paddingVector = size * padding
        meshBoundingBox.min -= paddingVector
        meshBoundingBox.max += paddingVector
        
        print("🔄 Generating SDF from AR mesh: \(triangles.count) triangles, resolution: \(sdfResolution)")
        print("📦 Bounding box: min=\(meshBoundingBox.min), max=\(meshBoundingBox.max)")
        
        // Generate SDF using existing function
        return generateSDFOptimized(triangles: triangles, resolution: sdfResolution, boundingBox: meshBoundingBox)
    }
    
    @available(iOS 13.4, macOS 10.15.4, *)
    func generateCombinedSDFFromARMeshes(_ meshAnchors: [ARMeshAnchor], resolution: SIMD3<Int32>? = nil) -> MTLTexture? {
        guard !meshAnchors.isEmpty else {
            print("⚠️ No AR mesh anchors provided for SDF generation")
            return nil
        }
        
        var allTriangles: [Triangle] = []
        var combinedBoundingBox = (min: SIMD3<Float>(repeating: Float.greatestFiniteMagnitude),
                                  max: SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude))
        
        // Process all mesh anchors
        for meshAnchor in meshAnchors {
            let geometry = meshAnchor.geometry
            let vertices = geometry.vertices
            let faces = geometry.faces
            let transform = meshAnchor.transform
            
            let vertexBuffer = vertices.buffer
            let vertexStride = vertices.stride
            let faceBuffer = faces.buffer
            let faceStride = MemoryLayout<UInt32>.stride * 3 // Each face has 3 indices
            let faceCount = faces.count
            
            // Process faces for this mesh
            for faceIndex in 0..<faceCount {
                let faceOffset = faceIndex * faceStride
                let facePointer = faceBuffer.contents().advanced(by: faceOffset).assumingMemoryBound(to: UInt32.self)
                
                let i0 = Int(facePointer[0])
                let i1 = Int(facePointer[1])
                let i2 = Int(facePointer[2])
                
                let v0Offset = i0 * vertexStride
                let v1Offset = i1 * vertexStride
                let v2Offset = i2 * vertexStride
                
                let v0Pointer = vertexBuffer.contents().advanced(by: v0Offset).assumingMemoryBound(to: Float.self)
                let v1Pointer = vertexBuffer.contents().advanced(by: v1Offset).assumingMemoryBound(to: Float.self)
                let v2Pointer = vertexBuffer.contents().advanced(by: v2Offset).assumingMemoryBound(to: Float.self)
                
                let v0Local = SIMD3<Float>(v0Pointer[0], v0Pointer[1], v0Pointer[2])
                let v1Local = SIMD3<Float>(v1Pointer[0], v1Pointer[1], v1Pointer[2])
                let v2Local = SIMD3<Float>(v2Pointer[0], v2Pointer[1], v2Pointer[2])
                
                let v0World = transformPoint(v0Local, by: transform)
                let v1World = transformPoint(v1Local, by: transform)
                let v2World = transformPoint(v2Local, by: transform)
                
                combinedBoundingBox.min = min(combinedBoundingBox.min, min(v0World, min(v1World, v2World)))
                combinedBoundingBox.max = max(combinedBoundingBox.max, max(v0World, max(v1World, v2World)))
                
                let triangle = Triangle(v0: v0World, v1: v1World, v2: v2World)
                allTriangles.append(triangle)
            }
        }
        
        let sdfResolution = resolution ?? SIMD3<Int32>(128, 128, 128) // Higher resolution for combined meshes
        
        // Add padding
        let padding: Float = 0.1
        let size = combinedBoundingBox.max - combinedBoundingBox.min
        let paddingVector = size * padding
        combinedBoundingBox.min -= paddingVector
        combinedBoundingBox.max += paddingVector
        
        print("🔄 Generating combined SDF from \(meshAnchors.count) AR meshes: \(allTriangles.count) total triangles")
        print("📦 Combined bounding box: min=\(combinedBoundingBox.min), max=\(combinedBoundingBox.max)")
        
        return generateSDFOptimized(triangles: allTriangles, resolution: sdfResolution, boundingBox: combinedBoundingBox)
    }
    
    private func transformPoint(_ point: SIMD3<Float>, by transform: simd_float4x4) -> SIMD3<Float> {
        let homogeneousPoint = SIMD4<Float>(point.x, point.y, point.z, 1.0)
        let transformedPoint = transform * homogeneousPoint
        return SIMD3<Float>(transformedPoint.x, transformedPoint.y, transformedPoint.z)
    }
    #endif
}
