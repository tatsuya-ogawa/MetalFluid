import Foundation
import Metal
import simd
#if canImport(ARKit)
import ARKit
#endif

public class SDFGenerator {
    private let useOptimized: Bool = false
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var sdfPipelineState: MTLComputePipelineState?
    private var reducePipelineState: MTLComputePipelineState?
    let padding: Float = 0.1
    // Feature toggles (sparse planned, heap-enabled now)
    private let useHeapsForSDF: Bool = true

    public init(device: MTLDevice) {
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
            if useOptimized{
                if let optimizedFunction = library.makeFunction(name: "generateSDFOptimized") {
                    sdfPipelineState = try device.makeComputePipelineState(function: optimizedFunction)
                }
            }else{
                if let function = library.makeFunction(name: "generateSDF") {
                    sdfPipelineState = try device.makeComputePipelineState(function: function)
                }
            }
            if let reduceFn = library.makeFunction(name: "reduceSDFMinMaxPerTile") {
                reducePipelineState = try device.makeComputePipelineState(function: reduceFn)
            }
        } catch {
            print("Failed to create SDF compute pipeline states: \(error)")
        }
    }
    
    
    func generateSDF(triangles: [Triangle], resolution: SIMD3<Int32>, boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)) -> MTLTexture? {
        guard let pipelineState = sdfPipelineState else {
            fatalError("SDF compute pipeline not available, falling back to standard")
        }
        // Convert triangles to GPU format
        let sdfTriangles = triangles.map { triangle in
            Triangle(v0: triangle.v0, v1: triangle.v1, v2: triangle.v2)
        }
        
        // Chunking parameters to prevent GPU memory issues
        let maxTrianglesPerChunk = 1000  // Process in smaller chunks
        let totalTriangles = sdfTriangles.count
        let numChunks = (totalTriangles + maxTrianglesPerChunk - 1) / maxTrianglesPerChunk
        
        print("🔧 Triangle chunking: \(totalTriangles) triangles -> \(numChunks) chunks")
        
        // Create triangle buffer for all triangles (read-only)
        guard let triangleBuffer = device.makeBuffer(
            bytes: sdfTriangles,
            length: MemoryLayout<Triangle>.stride * sdfTriangles.count,
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
//        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.storageMode = .private

        var sdfTexture: MTLTexture?
#if compiler(>=5.9)
        if useHeapsForSDF {
            // Allocate texture from a private heap (future-proof for sparse integration)
            let sizeAndAlign = device.heapTextureSizeAndAlign(descriptor: textureDescriptor)
            let heapDesc = MTLHeapDescriptor()
            heapDesc.storageMode = .private
            heapDesc.type = .sparse
            heapDesc.size = ((sizeAndAlign.size + sizeAndAlign.align - 1) / sizeAndAlign.align) * sizeAndAlign.align
            if let heap = device.makeHeap(descriptor: heapDesc) {
                sdfTexture = heap.makeTexture(descriptor: textureDescriptor)
            }
        }
#endif
        if sdfTexture == nil {
            // Fallback to direct allocation
            sdfTexture = device.makeTexture(descriptor: textureDescriptor)
        }
        guard let sdfTexture else {
            print("Failed to create SDF texture")
            return nil
        }
        sdfTexture.label = "SDFTexture"
        
        // Setup compute parameters
        let voxelSize = (boundingBox.max - boundingBox.min) / SIMD3<Float>(resolution)
        var sdfOrigin = boundingBox.min
        var voxelSizeVar = voxelSize
        var resolutionVar = resolution
        
        // Process triangles in chunks to avoid GPU memory issues
        for chunkIndex in 0..<numChunks {
            let chunkStart = chunkIndex * maxTrianglesPerChunk
            let chunkEnd = min(chunkStart + maxTrianglesPerChunk, totalTriangles)
            let chunkSize = chunkEnd - chunkStart
            
            print("🔧 Processing chunk \(chunkIndex + 1)/\(numChunks): triangles \(chunkStart)..<\(chunkEnd) (\(chunkSize) triangles)")
            
            // Create command buffer for this chunk
            guard let chunkCommandBuffer = commandQueue.makeCommandBuffer() else {
                print("❌ Failed to create command buffer for chunk \(chunkIndex)")
                return nil
            }
            
            // Handle sparse texture mapping for first chunk
            if chunkIndex == 0 && useHeapsForSDF {
                // Sparse: first map full region (tiles). We'll purge empty tiles after SDF is generated.
#if compiler(>=5.9)
                if #available(iOS 17.0, macOS 14.0, *),
                   let rse = chunkCommandBuffer.makeResourceStateCommandEncoder() {
                    let tile = device.sparseTileSize(with: textureDescriptor.textureType,
                                                     pixelFormat: textureDescriptor.pixelFormat,
                                                     sampleCount: 1)
                    let wTiles = max(1, (Int(resolution.x) + tile.width  - 1) / tile.width)
                    let hTiles = max(1, (Int(resolution.y) + tile.height - 1) / tile.height)
                    let dTiles = max(1, (Int(resolution.z) + tile.depth  - 1) / tile.depth)
                    let full = MTLRegionMake3D(0, 0, 0, wTiles, hTiles, dTiles)
#if targetEnvironment(macCatalyst) || os(macOS)
                    rse.updateTextureMapping?(sdfTexture, mode: .map, region: full, mipLevel: 0, slice: 0)
#else
                    rse.updateTextureMapping(sdfTexture, mode: .map, region: full, mipLevel: 0, slice: 0)
#endif
                    rse.endEncoding()
                }
#endif
            }
            
            // Create compute encoder for this chunk
            guard let computeEncoder = chunkCommandBuffer.makeComputeCommandEncoder() else {
                print("❌ Failed to create compute encoder for chunk \(chunkIndex)")
                return nil
            }
            
            computeEncoder.label = "SDF Generation Chunk \(chunkIndex + 1)/\(numChunks)"
            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(triangleBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(sdfDataBuffer, offset: 0, index: 1)
            
            var triangleOffset = UInt32(chunkStart)
            var triangleChunkSize = UInt32(chunkSize)
            computeEncoder.setBytes(&triangleOffset, length: MemoryLayout<UInt32>.size, index: 2)
            computeEncoder.setBytes(&triangleChunkSize, length: MemoryLayout<UInt32>.size, index: 3)
            computeEncoder.setBytes(&sdfOrigin, length: MemoryLayout<SIMD3<Float>>.size, index: 4)
            computeEncoder.setBytes(&voxelSizeVar, length: MemoryLayout<SIMD3<Float>>.size, index: 5)
            computeEncoder.setBytes(&resolutionVar, length: MemoryLayout<SIMD3<Int32>>.size, index: 6)
            
            // Calculate threadgroup size dynamically from pipeline
            let w = sdfPipelineState?.threadExecutionWidth ?? 8
            let maxThreads = sdfPipelineState?.maxTotalThreadsPerThreadgroup ?? 64
            let h = max(1, maxThreads / w)
            let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
            let threadsPerGrid = MTLSize(width: Int(resolution.x), height: Int(resolution.y), depth: Int(resolution.z))
            
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
            
            // Execute this chunk
            chunkCommandBuffer.commit()
            chunkCommandBuffer.waitUntilCompleted()
            
            // Check for errors
            if chunkCommandBuffer.status == .error {
                print("❌ SDF generation chunk \(chunkIndex + 1) failed")
                if let error = chunkCommandBuffer.error {
                    print("  Error: \(error)")
                }
                return nil
            }
            
            print("✅ Chunk \(chunkIndex + 1)/\(numChunks) completed successfully")
        }
        
        print("✅ All \(numChunks) chunks processed successfully")
        
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

        // Narrow-band purge: compute per-tile min/max on GPU and unmap tiles fully outside (min>0)
#if compiler(>=5.9)
        if useHeapsForSDF, #available(iOS 17.0, macOS 14.0, *) {
            let resX = Int(resolution.x), resY = Int(resolution.y), resZ = Int(resolution.z)
            let tile = device.sparseTileSize(with: textureDescriptor.textureType,
                                             pixelFormat: textureDescriptor.pixelFormat,
                                             sampleCount: 1)
            let wTiles = max(1, (resX + tile.width  - 1) / tile.width)
            let hTiles = max(1, (resY + tile.height - 1) / tile.height)
            let dTiles = max(1, (resZ + tile.depth  - 1) / tile.depth)
            let tileCount = wTiles * hTiles * dTiles
            guard let outMinMax = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * tileCount, options: .storageModeShared) else { return sdfTexture }
            outMinMax.label = "SDFTileMinMax"
            if let reduceCB = commandQueue.makeCommandBuffer(),
               let enc = reduceCB.makeComputeCommandEncoder(),
               let reducePSO = reducePipelineState {
                enc.setComputePipelineState(reducePSO)
                enc.setBuffer(sdfDataBuffer, offset: 0, index: 0)
                var resI = SIMD3<Int32>(resolution.x, resolution.y, resolution.z)
                var tileI = SIMD3<Int32>(Int32(tile.width), Int32(tile.height), Int32(tile.depth))
                var tcountI = SIMD3<Int32>(Int32(wTiles), Int32(hTiles), Int32(dTiles))
                enc.setBytes(&resI, length: MemoryLayout<SIMD3<Int32>>.stride, index: 1)
                enc.setBytes(&tileI, length: MemoryLayout<SIMD3<Int32>>.stride, index: 2)
                enc.setBytes(&tcountI, length: MemoryLayout<SIMD3<Int32>>.stride, index: 3)
                enc.setBuffer(outMinMax, offset: 0, index: 4)
                let tg = MTLSize(width: 64, height: 1, depth: 1)
                let tgGrid = MTLSize(width: wTiles, height: hTiles, depth: dTiles)
                enc.dispatchThreadgroups(tgGrid, threadsPerThreadgroup: tg)
                enc.endEncoding()
                reduceCB.commit()
                reduceCB.waitUntilCompleted()
            }
            if let purgeCB = commandQueue.makeCommandBuffer(),
               let rse2 = purgeCB.makeResourceStateCommandEncoder() {
                let ptr = outMinMax.contents().bindMemory(to: SIMD2<Float>.self, capacity: tileCount)
                var idx = 0
                for tz in 0..<dTiles {
                    for ty in 0..<hTiles {
                        for tx in 0..<wTiles {
                            let mm = ptr[idx]; idx += 1
                            if mm.x > 0.0 {
                                let reg = MTLRegionMake3D(tx, ty, tz, 1, 1, 1)
#if targetEnvironment(macCatalyst) || os(macOS)
                                rse2.updateTextureMapping?(sdfTexture, mode: .unmap, region: reg, mipLevel: 0, slice: 0)
#else
                                rse2.updateTextureMapping(sdfTexture, mode: .unmap, region: reg, mipLevel: 0, slice: 0)
#endif
                            }
                        }
                    }
                }
                rse2.endEncoding()
                purgeCB.commit()
                purgeCB.waitUntilCompleted()
            }
        }
#endif
        
        print("GPU SDF generation completed successfully")
        return sdfTexture
    }
}
