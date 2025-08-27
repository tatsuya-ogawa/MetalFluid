//
//  Untitled.swift
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/08/17.
//
import Metal
import MetalKit
import simd
extension MPMFluidRenderer{
    internal func setupBitonicSortPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }
        
        // Extract sort keys pipeline
        guard let extractSortKeysFunction = library.makeFunction(name: "extractSortKeys") else {
            fatalError("Could not find function 'extractSortKeys'")
        }
        do {
            extractSortKeysPipelineState = try device.makeComputePipelineState(
                function: extractSortKeysFunction
            )
        } catch {
            fatalError("Could not create extract sort keys pipeline state: \(error)")
        }
        
        // Bitonic sort pipeline
        guard let bitonicSortFunction = library.makeFunction(name: "bitonicSort") else {
            fatalError("Could not find function 'bitonicSort'")
        }
        do {
            bitonicSortPipelineState = try device.makeComputePipelineState(
                function: bitonicSortFunction
            )
        } catch {
            fatalError("Could not create bitonic sort pipeline state: \(error)")
        }
        
        // Reorder particles pipeline
        guard let reorderParticlesFunction = library.makeFunction(name: "reorderParticles") else {
            fatalError("Could not find function 'reorderParticles'")
        }
        do {
            reorderParticlesPipelineState = try device.makeComputePipelineState(
                function: reorderParticlesFunction
            )
        } catch {
            fatalError("Could not create reorder particles pipeline state: \(error)")
        }
        
        // Verify sort order pipeline
        guard let verifySortFunction = library.makeFunction(name: "verifySortOrder") else {
            fatalError("Could not find function 'verifySortOrder'")
        }
        do {
            verifySortPipelineState = try device.makeComputePipelineState(
                function: verifySortFunction
            )
        } catch {
            fatalError("Could not create verify sort pipeline state: \(error)")
        }
        
        // Update max threads per group
        maxThreadsPerGroup = min(
            extractSortKeysPipelineState.maxTotalThreadsPerThreadgroup,
            256
        )
    }
    
    internal func setupRadixSortPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }
        
        // Extract sort keys for radix sort pipeline
        guard let extractSortKeysRadixFunction = library.makeFunction(name: "extractSortKeysRadix") else {
            fatalError("Could not find function 'extractSortKeysRadix'")
        }
        do {
            extractSortKeysRadixPipelineState = try device.makeComputePipelineState(
                function: extractSortKeysRadixFunction
            )
        } catch {
            fatalError("Could not create extract sort keys radix pipeline state: \(error)")
        }
        
        // One-sweep radix sort pass pipeline
        guard let radixSortPassFunction = library.makeFunction(name: "onesweepRadixSortPass") else {
            fatalError("Could not find function 'onesweepRadixSortPass'")
        }
        do {
            radixSortPassPipelineState = try device.makeComputePipelineState(
                function: radixSortPassFunction
            )
        } catch {
            fatalError("Could not create radix sort pass pipeline state: \(error)")
        }
        
        // Local radix sort pipeline
        guard let radixSortLocalFunction = library.makeFunction(name: "radixSortLocal") else {
            fatalError("Could not find function 'radixSortLocal'")
        }
        do {
            radixSortLocalPipelineState = try device.makeComputePipelineState(
                function: radixSortLocalFunction
            )
        } catch {
            fatalError("Could not create radix sort local pipeline state: \(error)")
        }
        
        // Initialize radix histogram pipeline
        guard let initializeRadixHistogramFunction = library.makeFunction(name: "initializeRadixHistogram") else {
            fatalError("Could not find function 'initializeRadixHistogram'")
        }
        do {
            initializeRadixHistogramPipelineState = try device.makeComputePipelineState(
                function: initializeRadixHistogramFunction
            )
        } catch {
            fatalError("Could not create initialize radix histogram pipeline state: \(error)")
        }
        
        // Compute prefix sums pipeline
        guard let computePrefixSumsFunction = library.makeFunction(name: "computePrefixSums") else {
            fatalError("Could not find function 'computePrefixSums'")
        }
        do {
            computePrefixSumsPipelineState = try device.makeComputePipelineState(
                function: computePrefixSumsFunction
            )
        } catch {
            fatalError("Could not create compute prefix sums pipeline state: \(error)")
        }
        
        // Reorder particles radix pipeline
        guard let reorderParticlesRadixFunction = library.makeFunction(name: "reorderParticlesRadix") else {
            fatalError("Could not find function 'reorderParticlesRadix'")
        }
        do {
            reorderParticlesRadixPipelineState = try device.makeComputePipelineState(
                function: reorderParticlesRadixFunction
            )
        } catch {
            fatalError("Could not create reorder particles radix pipeline state: \(error)")
        }
        
        // Verify radix sort pipeline
        guard let verifyRadixSortFunction = library.makeFunction(name: "verifyRadixSortOrder") else {
            fatalError("Could not find function 'verifyRadixSortOrder'")
        }
        do {
            verifyRadixSortPipelineState = try device.makeComputePipelineState(
                function: verifyRadixSortFunction
            )
        } catch {
            fatalError("Could not create verify radix sort pipeline state: \(error)")
        }
    }
    
    // Sort particles by grid index for better cache locality
    func sortParticlesByGridIndex() {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // 1. Extract sort keys (grid indices)
        extractSortKeys(commandBuffer: commandBuffer)
        
        // 2. Perform bitonic sort
        bitonicSort(commandBuffer: commandBuffer)
        
        // 3. Reorder particles based on sorted keys
        reorderParticles(commandBuffer: commandBuffer)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Swap particle buffers
        swapParticleBufferWith(&sortedParticleBuffer)
    }
    
    internal func bitonicSort(commandBuffer: MTLCommandBuffer) {
        let n = particleCount
        var stage = 1
        
        // Find next power of 2
        let powerOf2 = 1 << Int(ceil(log2(Double(n))))
        
        while stage < powerOf2 {
            var step = stage
            
            while step > 0 {
                guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    return
                }
                
                computeEncoder.setComputePipelineState(bitonicSortPipelineState)
                computeEncoder.setBuffer(sortKeysBuffer, offset: 0, index: 0)
                computeEncoder.setBytes([UInt32(n)], length: MemoryLayout<UInt32>.size, index: 1)
                computeEncoder.setBytes([UInt32(stage)], length: MemoryLayout<UInt32>.size, index: 2)
                computeEncoder.setBytes([UInt32(step)], length: MemoryLayout<UInt32>.size, index: 3)
                
                let threadsPerThreadgroup = MTLSize(
                    width: min(maxThreadsPerGroup, n),
                    height: 1,
                    depth: 1
                )
                let threadgroups = MTLSize(
                    width: (n + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                    height: 1,
                    depth: 1
                )
                
                computeEncoder.dispatchThreadgroups(
                    threadgroups,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
                computeEncoder.endEncoding()
                
                step >>= 1
            }
            
            stage <<= 1
        }
    }
    
    internal func extractSortKeys(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(extractSortKeysPipelineState)
        computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sortKeysBuffer, offset: 0, index: 2)
        
        let threadsPerThreadgroup = MTLSize(
            width: maxThreadsPerGroup,
            height: 1,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (particleCount + maxThreadsPerGroup - 1) / maxThreadsPerGroup,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        computeEncoder.endEncoding()
    }
    internal func setupBitonicSortBuffers() {
        // Sort keys buffer for bitonic sort (only need one)
        let sortKeysBufferSize = MemoryLayout<SortKey>.stride * particleCount
        sortKeysBuffer = device.makeBuffer(
            length: sortKeysBufferSize,
            options: .storageModeShared
        )!
        
        // Sorted particle buffer
        let sortedParticleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        sortedParticleBuffer = device.makeBuffer(
            length: sortedParticleBufferSize,
            options: .storageModeShared
        )!
        
        print("🔍 Bitonic sort buffer sizes:")
        print("   Sort keys buffer: \(sortKeysBufferSize) bytes")
        print("   Sorted particle buffer: \(sortedParticleBufferSize) bytes")

        // Initialize grid buffer (minimal necessary zero clearing only)
        let gridPointer = gridBuffer.contents().bindMemory(
            to: MPMGridNode.self,
            capacity: gridNodes
        )
        for i in 0..<gridNodes {
            gridPointer[i] = MPMGridNode(
                velocity_x: 0.0,
                velocity_y: 0.0,
                velocity_z: 0.0,
                mass: 0.0
            )
        }
    }
    
    internal func setupRadixSortBuffers() {
        // Radix sort keys buffer
        let radixSortKeysBufferSize = MemoryLayout<SortKey>.stride * particleCount
        radixSortKeysBuffer = device.makeBuffer(
            length: radixSortKeysBufferSize,
            options: .storageModeShared
        )!
        
        // Temporary keys buffer for radix sort
        radixTempKeysBuffer = device.makeBuffer(
            length: radixSortKeysBufferSize,
            options: .storageModeShared
        )!
        
        // Histogram buffer for radix sort (16 bins per threadgroup)
        let maxThreadgroups = (particleCount + maxThreadsPerGroup - 1) / maxThreadsPerGroup
        let histogramSize = maxThreadgroups * 16 * MemoryLayout<UInt32>.stride
        radixHistogramBuffer = device.makeBuffer(
            length: histogramSize,
            options: .storageModeShared
        )!
        
        // Sorted particle buffer for radix sort
        let radixSortedParticleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        radixSortedParticleBuffer = device.makeBuffer(
            length: radixSortedParticleBufferSize,
            options: .storageModeShared
        )!
        
        print("🔍 Radix sort buffer sizes:")
        print("   Radix sort keys buffer: \(radixSortKeysBufferSize) bytes")
        print("   Radix temp keys buffer: \(radixSortKeysBufferSize) bytes")
        print("   Radix histogram buffer: \(histogramSize) bytes")
        print("   Radix sorted particle buffer: \(radixSortedParticleBufferSize) bytes")
    }
    internal func sortParticlesByGridIndexSafe() throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "MetalFluid", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"])
        }
        
        switch currentSortingAlgorithm {
        case .bitonicSort:
            // 1. Extract sort keys (grid indices)
            extractSortKeys(commandBuffer: commandBuffer)
            
            // 2. Perform bitonic sort
            bitonicSort(commandBuffer: commandBuffer)
            
            // 3. Reorder particles based on sorted keys
            reorderParticles(commandBuffer: commandBuffer)
            
            // Swap particle buffers
            swapParticleBufferWith(&sortedParticleBuffer)
            
        case .radixSort:
            // 1. Extract sort keys for radix sort
            extractSortKeysRadix(commandBuffer: commandBuffer)
            
            // 2. Perform radix sort
            radixSort(commandBuffer: commandBuffer)
            
            // 3. Reorder particles based on sorted keys
            reorderParticlesRadix(commandBuffer: commandBuffer)
            
            // Swap particle buffers
            swapParticleBufferWith(&radixSortedParticleBuffer)
        case .none: break
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Check for command buffer errors
        if let error = commandBuffer.error {
            throw error
        }
        
        // Optionally validate sort correctness in debug builds
        #if DEBUG
        if false && frameIndex % (sortingFrequency * 20) == 0 {
            validateSortOrder()
        }
        #endif
    }
    
    internal func validateSortOrder() {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // Clear error count
        let errorCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        let errorPointer = errorCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        errorPointer[0] = 0
        
        // Verify sort order
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(verifySortPipelineState)
        computeEncoder.setBuffer(sortKeysBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(errorCountBuffer, offset: 0, index: 1)
        computeEncoder.setBytes([UInt32(particleCount)], length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(
            width: min(maxThreadsPerGroup, particleCount),
            height: 1,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (particleCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let errorCount = errorPointer[0]
        if errorCount > 0 {
            print("⚠️ Sort validation failed: \(errorCount) errors found")
            disableSortingOnError()
        } else {
            print("✅ Sort validation passed")
        }
    }
    // MARK: - Radix Sort Implementation
    
    internal func extractSortKeysRadix(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(extractSortKeysRadixPipelineState)
        computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(radixSortKeysBuffer, offset: 0, index: 2)
        
        let threadsPerThreadgroup = MTLSize(
            width: maxThreadsPerGroup,
            height: 1,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (particleCount + maxThreadsPerGroup - 1) / maxThreadsPerGroup,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        computeEncoder.endEncoding()
    }
    
    internal func radixSort(commandBuffer: MTLCommandBuffer) {
        // Use simplified local radix sort for better performance on GPU
        let numPasses = 8 // For 32-bit keys with 4-bit radix
        
        for pass in 0..<numPasses {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                return
            }
            
            computeEncoder.setComputePipelineState(radixSortLocalPipelineState)
            computeEncoder.setBuffer(radixSortKeysBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(radixTempKeysBuffer, offset: 0, index: 1)
            computeEncoder.setBytes([UInt32(particleCount)], length: MemoryLayout<UInt32>.size, index: 2)
            computeEncoder.setBytes([UInt32(pass)], length: MemoryLayout<UInt32>.size, index: 3)
            
            let threadsPerThreadgroup = MTLSize(
                width: min(maxThreadsPerGroup, 256), // Limit for shared memory
                height: 1,
                depth: 1
            )
            let threadgroups = MTLSize(
                width: (particleCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                height: 1,
                depth: 1
            )
            
            // Set threadgroup memory for local sorting
            let sharedMemorySize = threadsPerThreadgroup.width * MemoryLayout<SortKey>.stride
            computeEncoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)
            
            computeEncoder.dispatchThreadgroups(
                threadgroups,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            computeEncoder.endEncoding()
        }
    }
    
    internal func reorderParticlesRadix(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(reorderParticlesRadixPipelineState)
        computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(radixSortedParticleBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(radixSortKeysBuffer, offset: 0, index: 2)
        computeEncoder.setBytes([UInt32(particleCount)], length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadsPerThreadgroup = MTLSize(
            width: maxThreadsPerGroup,
            height: 1,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (particleCount + maxThreadsPerGroup - 1) / maxThreadsPerGroup,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        computeEncoder.endEncoding()
    }
    
    internal func copyBuffer(
        commandBuffer: MTLCommandBuffer,
        from sourceBuffer: MTLBuffer,
        to destinationBuffer: MTLBuffer
    ) {
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            return
        }
        
        blitEncoder.copy(
            from: sourceBuffer,
            sourceOffset: 0,
            to: destinationBuffer,
            destinationOffset: 0,
            size: min(sourceBuffer.length, destinationBuffer.length)
        )
        blitEncoder.endEncoding()
    }
    
    // Public API for particle sorting configuration
    public func setParticleSorting(enabled: Bool, frequency: Int = 4) {
        enableParticleSorting = enabled
        sortingFrequency = max(1, frequency)
        print("🔄 Particle sorting: \(enabled ? "enabled" : "disabled"), frequency: every \(sortingFrequency) frames")
    }
    
    public func forceSortParticles() {
        let startTime = CACurrentMediaTime()
        sortParticlesByGridIndex()
        let sortTime = CACurrentMediaTime() - startTime
        print("🔄 Manual particle sort took: \(String(format: "%.2f", sortTime * 1000))ms")
    }
    
    // Disable sorting if errors occur
    public func disableSortingOnError() {
        enableParticleSorting = false
        print("⚠️ Particle sorting disabled due to errors")
    }
    
    // MARK: - Sorting Algorithm Selection
    
    public func setSortingAlgorithm(_ algorithm: SortingAlgorithm) {
        currentSortingAlgorithm = algorithm
        print("🔄 Sorting algorithm switched to: \(algorithm)")
    }
    
    public func toggleSortingAlgorithm() {
        switch currentSortingAlgorithm {
        case .bitonicSort:
            setSortingAlgorithm(.radixSort)
        case .radixSort:
            setSortingAlgorithm(.none)
        case .none:
            setSortingAlgorithm(.radixSort)
        }
    }
    
    public func getSortingAlgorithmName() -> String {
        switch currentSortingAlgorithm {
        case .bitonicSort:
            return "Bitonic Sort"
        case .radixSort:
            return "One-sweep Radix Sort"
        case .none:
            return "None"
        }
    }

}
