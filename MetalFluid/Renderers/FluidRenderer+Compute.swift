import Metal
import MetalKit
import simd

// Threading constants matching ComputeShaders.h
private let DEFAULT_THREADGROUP_SIZE = 256
private let GRID_THREADGROUP_SIZE = 64

// MARK: - Compute Extension
extension MPMFluidRenderer {
    
    // MARK: - Setup Functions
    
    private func createComputePipelineState(library: MTLLibrary, functionName: String) -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            fatalError("Could not find function '\(functionName)'")
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            fatalError("Could not create \(functionName) pipeline state: \(error)")
        }
    }
    
    internal func setupComputePipelinesFluid(library: MTLLibrary) {
        particlesToGridFluid1PipelineState = createComputePipelineState(library: library, functionName: "particlesToGridFluid1")
        particlesToGridFluid2PipelineState = createComputePipelineState(library: library, functionName: "particlesToGridFluid2")
        gridToParticlesFluid1PipelineState = createComputePipelineState(library: library, functionName: "gridToParticlesFluid1")
    }
    
    internal func setupComputePipelinesElastic(library: MTLLibrary) {
        particlesToGridElasticPipelineState = createComputePipelineState(library: library, functionName: "particlesToGridElastic")
        gridToParticlesElasticPipelineState = createComputePipelineState(library: library, functionName: "gridToParticlesElastic")
    }
        
    internal func setupComputePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }

        clearGridPipelineState = createComputePipelineState(library: library, functionName: "clearGrid")
        updateGridVelocityPipelineState = createComputePipelineState(library: library, functionName: "updateGridVelocity")

        // Force application pipeline
        applyForceToGridPipelineState = createComputePipelineState(library: library, functionName: "applyForceToGrid")
        
        // SDF collision physics integration pipeline
        applySdfImpulseToTransformPipelineState = createComputePipelineState(library: library, functionName: "applySdfImpulseToTransform")

        setupComputePipelinesFluid(library: library)
        setupComputePipelinesElastic(library: library)
    }
    
    internal func setupParticles() {
        // Setup particles in staging buffer (shared memory for CPU access)
        let stagingParticlePointer = scene.getParticleStagingBuffer().contents().bindMemory(
            to: MPMParticle.self,
            capacity: particleCount
        )
        
        // Get center and range of boundaryMin/boundaryMax
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let center = (boundaryMin + boundaryMax) * 0.5
        let range = (boundaryMax - boundaryMin) * 0.5
        
        if materialParameters.currentMaterialMode == .neoHookeanElastic {
            // Dense cube formation for elastic and rigid body materials
            setupElasticCube(particlePointer: stagingParticlePointer, center: center, range: range)
            
        } else {
            // Original spherical distribution for fluid
            setupFluidSphere(particlePointer: stagingParticlePointer, center: center, range: range, boundaryMin: boundaryMin, boundaryMax: boundaryMax)
        }
        
        // Copy staging data to private GPU buffers using blit encoder
        blitStagingToPrivateBuffers()
        
        print("🔄 Initialized particles and copied to private GPU buffers")
    }
    
    // MARK: - Blit Copy Functions
    
    private func blitStagingToPrivateBuffers() {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            print("⚠️ Failed to create command buffer or blit encoder for staging copy")
            return
        }
        
        let particleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        
        // Copy staging particle buffer to both compute and render private buffers
        blitEncoder.copy(
            from: scene.getParticleStagingBuffer(), sourceOffset: 0,
            to: scene.getComputeParticleBuffer(), destinationOffset: 0,
            size: particleBufferSize
        )
        blitEncoder.copy(
            from: scene.getParticleStagingBuffer(), sourceOffset: 0,
            to: scene.getRenderParticleBuffer(), destinationOffset: 0,
            size: particleBufferSize
        )
        
        blitEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        print("💾 Blit copy completed: staging → private buffers")
    }
    
    private func createParticle(at position: SIMD3<Float>, index: Int) -> MPMParticle {
        return MPMParticle(
            position: position,
            velocity: SIMD3<Float>(0.0, 0.0, 0.0),
            C: simd_float3x3(0.0),
            mass: materialParameters.particleMass,
            originalIndex: UInt32(index)
        )
    }
    
            
    private func setupElasticCube(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>) {
        let particlesPerDim = Int(floor(pow(Float(particleCount), 1.0/3.0)))
        let cubeSize = min(range.x, range.y, range.z) * 0.6
        let spacing = cubeSize / Float(particlesPerDim - 1)
        let cubeOrigin = center - SIMD3<Float>(cubeSize * 0.5, cubeSize * 0.5, cubeSize * 0.5)
        var particleIndex = 0
        
        // Create lattice particles
        for x in 0..<particlesPerDim {
            for y in 0..<particlesPerDim {
                for z in 0..<particlesPerDim {
                    if particleIndex >= particleCount { break }
                    
                    let pos = cubeOrigin + SIMD3<Float>(Float(x) * spacing, Float(y) * spacing, Float(z) * spacing)
                    let randomOffset = SIMD3<Float>(
                        Float.random(in: -0.001...0.001),
                        Float.random(in: -0.001...0.001),
                        Float.random(in: -0.001...0.001)
                    )
                    let finalPos = pos + randomOffset
                    
                    particlePointer[particleIndex] = createParticle(at: finalPos, index: particleIndex)
                    particleIndex += 1
                }
                if particleIndex >= particleCount { break }
            }
            if particleIndex >= particleCount { break }
        }
        
        // Fill remaining particles randomly
        while particleIndex < particleCount {
            let pos = cubeOrigin + SIMD3<Float>(
                Float.random(in: 0...cubeSize),
                Float.random(in: 0...cubeSize),
                Float.random(in: 0...cubeSize)
            )
            
            particlePointer[particleIndex] = createParticle(at: pos, index: particleIndex)
            particleIndex += 1
        }
        
        print("🟦 Created elastic cube: \(particlesPerDim)³ lattice, spacing: \(spacing)")
    }
        
    private func setupFluidSphere(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, boundaryMin: SIMD3<Float>, boundaryMax: SIMD3<Float>) {
        let maxRadius = min(range.x, range.y, range.z)
        
        func randn() -> Float {
            let u1 = Float.random(in: 0..<1)
            let u2 = Float.random(in: 0..<1)
            return sqrt(-2.0 * log(u1 + 1e-7)) * cos(2.0 * .pi * u2)
        }
        
        for i in 0..<particleCount {
            var pos: SIMD3<Float>
            while true {
                let v = SIMD3<Float>(randn(), randn(), randn())
                let r = length(v)
                if r > 3.0 { continue }
                let scaled = v / r * (r / 3.0) * maxRadius
                pos = center + scaled
                if all(pos .>= boundaryMin) && all(pos .<= boundaryMax) {
                    break
                }
            }
            
            let randomOffset = SIMD3<Float>(
                Float.random(in: -0.003...0.003),
                Float.random(in: -0.003...0.003),
                Float.random(in: -0.003...0.003)
            )
            let finalPos = simd_clamp(pos + randomOffset, boundaryMin, boundaryMax)
            
            particlePointer[i] = createParticle(at: finalPos, index: i)
        }
        
        print("🌊 Created fluid sphere: radius \(maxRadius)")
    }
    
    // MARK: - Main Compute Function
    func compute(commandBuffer: MTLCommandBuffer) {
        // Skip if compute is already in progress
        if isComputing {
            print("⚠️ Skipping compute request - compute already in progress")
            return
        }
        
        // Begin compute stage - use compute buffers only
        beginCompute()
        
        // Sort particles periodically for better cache locality (compute buffer only)
        if sortManager.enableParticleSorting && frameIndex % sortManager.sortingFrequency == 0 {
            do {
                let startTime = CACurrentMediaTime()
                var computeBuffer = scene.getComputeParticleBuffer()
                try sortManager.sortParticlesByGridIndexSafe(
                    computeParticleBuffer: &computeBuffer,
                    uniformBuffer: scene.getComputeUniformBuffer()
                )
                let sortTime = CACurrentMediaTime() - startTime
                if frameIndex % (sortManager.sortingFrequency * 10) == 0 {
                    print("🔄 Particle sort took: \(String(format: "%.2f", sortTime * 1000))ms (compute buffer)")
                }
            } catch {
                // Error handling is done inside sortManager
            }
        }
        
        computeSimulation(commandBuffer: commandBuffer)
        
        // Apply SDF impulse integration on GPU (after particle simulation completes)
//        applySdfImpulseToTransformGPU(commandBuffer: commandBuffer)
        
        // Add completion handler to swap buffers when the shared command buffer finishes
        // (This will be called when the render pass commits the command buffer)
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.endComputeAndSwapToRender()
        }
    }
    // Bind SDF resources: textures + uniforms via argument buffer (SDFSet), physics direct
    private func setupSdfArgumentBufferForCompute(computeEncoder: MTLComputeCommandEncoder, collisionManager: CollisionManager) {
        let items = collisionManager.items.filter{ $0.isEnabled() }
        let count = min(items.count, CollisionManager.MAX_COLLISION_SDF)
        // Build argument buffer for SDFSet (textures + uniform pointers) → buffer(3)
        let (argBuf, argEnc) = ensureSdfArgumentBuffer()
        for i in 0..<count {
            if let tex = items[i].getSDFTexture() {
                argEnc.setTexture(tex, index: i)
                computeEncoder.useResource(tex, usage: .read)
            }
            argEnc.setBuffer(items[i].getCollisionUniformBuffer(), offset: 0, index: CollisionManager.MAX_COLLISION_SDF + i)
        }
        computeEncoder.setBuffer(argBuf, offset: 0, index: 3)
        // Physics buffer
        if let phy = scene.getSDFPhysicsBuffer() { computeEncoder.setBuffer(phy, offset: 0, index: 4) }
        // Count
        var c32 = UInt32(count)
        computeEncoder.setBytes(&c32, length: MemoryLayout<UInt32>.size, index: 5)
    }
    
    // Lazy init & reuse of SDF argument buffer (avoid per-dispatch creation cost)    
    internal func ensureSdfArgumentBuffer() -> (MTLBuffer, MTLArgumentEncoder) {
        if let sdfArgumentEncoder, let sdfArgumentBuffer{
            return (sdfArgumentBuffer, sdfArgumentEncoder)
        }else{
            // SDFSet: textures + uniform pointers
            let texDesc = MTLArgumentDescriptor()
            texDesc.dataType = .texture
            texDesc.textureType = .type3D
            texDesc.index = 0
            texDesc.access = .readOnly
            texDesc.arrayLength = CollisionManager.MAX_COLLISION_SDF
            let uniDesc = MTLArgumentDescriptor()
            uniDesc.dataType = .pointer
            uniDesc.index = CollisionManager.MAX_COLLISION_SDF
            uniDesc.access = .readOnly
            uniDesc.arrayLength = CollisionManager.MAX_COLLISION_SDF
            let enc = device.makeArgumentEncoder(arguments: [texDesc, uniDesc])!
            let buf = device.makeBuffer(length: enc.encodedLength, options: [])!
            enc.setArgumentBuffer(buf, offset: 0)
            sdfArgumentEncoder = enc
            sdfArgumentBuffer = buf
            return (buf, enc)
        }
    }


    // GPU-based SDF impulse integration and collision transform update
    internal func applySdfImpulseToTransformGPU(commandBuffer: MTLCommandBuffer) {
        guard let collisionManager = collisionManager,
              let phyBuf = scene.getSDFPhysicsBuffer() else { return }
        
        // Create compute encoder for SDF physics integration
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("⚠️ Failed to create compute encoder for SDF physics integration")
            return
        }
        
        computeEncoder.setComputePipelineState(applySdfImpulseToTransformPipelineState)
        
        // Set buffers
        computeEncoder.setBuffer(collisionManager.representativeItem.getCollisionUniformBuffer(), offset: 0, index: 0)
        computeEncoder.setBuffer(phyBuf, offset: 0, index: 1)
        computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 2)
        
        // Set parameters
        var sdfIndex: UInt32 = 0
        var enableGravity: Bool = true  // TODO: Get from SDF settings
        var isDynamic: Bool = true      // TODO: Get from SDF settings
        
        computeEncoder.setBytes(&sdfIndex, length: MemoryLayout<UInt32>.size, index: 3)
        computeEncoder.setBytes(&enableGravity, length: MemoryLayout<Bool>.size, index: 4)  
        computeEncoder.setBytes(&isDynamic, length: MemoryLayout<Bool>.size, index: 5)
        
        // Dispatch single thread since we're updating one SDF
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
    }

    // MARK: - MPM Simulation Pipeline
    internal func computeSimulation(commandBuffer: MTLCommandBuffer) {
        // Decide threadgroup sizes based on each pipeline's reported limits
        let gridThreadgroupSize = min(
            clearGridPipelineState.maxTotalThreadsPerThreadgroup,
            DEFAULT_THREADGROUP_SIZE
        )
        let particleThreadgroupSize = min(
            particlesToGridFluid1PipelineState.maxTotalThreadsPerThreadgroup,
            DEFAULT_THREADGROUP_SIZE
        )

        // Precompute dispatch sizes
        let gridThreadsPerThreadgroup = MTLSize(
            width: gridThreadgroupSize,
            height: 1,
            depth: 1
        )
        let gridThreadgroups = MTLSize(
            width: (gridNodes + gridThreadgroupSize - 1) / gridThreadgroupSize,
            height: 1,
            depth: 1
        )

        let particleThreadsPerThreadgroup = MTLSize(
            width: particleThreadgroupSize,
            height: 1,
            depth: 1
        )
        let particleThreadgroups = MTLSize(
            width: (particleCount + particleThreadgroupSize - 1)
                / particleThreadgroupSize,
            height: 1,
            depth: 1
        )

        // Use multiple simulation substeps per frame for better stability.
        let substeps = max(1, materialParameters.simulationSubsteps)

        // SDF physics accumulators are cleared after integration; do not clear here to preserve velocities

        for _ in 0..<substeps {
            // 1. Clear grid
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(clearGridPipelineState)
                computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 0)
                computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                computeEncoder.dispatchThreadgroups(
                    gridThreadgroups,
                    threadsPerThreadgroup: gridThreadsPerThreadgroup
                )
                computeEncoder.endEncoding()
            }

            // 2. Particle to Grid (P2G) - Material-dependent transfer
            if materialParameters.currentMaterialMode == .fluid {
                // Fluid P2G Phase 1: Transfer mass and momentum from particles to grid
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        particlesToGridFluid1PipelineState
                    )
                    computeEncoder.setBuffer(scene.getComputeParticleBuffer(), offset: 0, index: 0)
                    computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                    computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 2)
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
                // Fluid P2G Phase 2: Add volume and stress-based momentum from particles to grid
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        particlesToGridFluid2PipelineState
                    )
                    computeEncoder.setBuffer(scene.getComputeParticleBuffer(), offset: 0, index: 0)
                    computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                    computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 2)
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            } else if materialParameters.currentMaterialMode == .neoHookeanElastic {
                // Elastic P2G: Neo-Hookean elastic material transfer
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        particlesToGridElasticPipelineState
                    )
                    computeEncoder.setBuffer(scene.getComputeParticleBuffer(), offset: 0, index: 0)
                    computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                    computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 2)
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            } else { // .rigidBody
                // Rigid body mode: P2G handled separately via dedicated rigid body simulation
                print("🔶 Rigid body simulation mode - P2G skipped, using projection-based dynamics")
            }

            // Apply queued forces to grid (after P2G, before grid velocity update)
            applyQueuedForcesToGrid(commandBuffer: commandBuffer)

            // 3. Update grid velocity - Grid velocity update and boundary condition application
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(
                    updateGridVelocityPipelineState
                )
                computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 0)
                computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                computeEncoder.dispatchThreadgroups(
                    gridThreadgroups,
                    threadsPerThreadgroup: gridThreadsPerThreadgroup
                )
                computeEncoder.endEncoding()
            }

            // 4. Grid to Particles (G2P) - Material-dependent transfer
            if materialParameters.currentMaterialMode == .fluid {
                // Fluid G2P: Transfer velocity and affine momentum from grid to particles
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        gridToParticlesFluid1PipelineState
                    )
                    computeEncoder.setBuffer(scene.getComputeParticleBuffer(), offset: 0, index: 0)
                    computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                    computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 2)
                    
                    // Set collision resources if available
                    if let collisionManager {
                        setupSdfArgumentBufferForCompute(computeEncoder: computeEncoder, collisionManager: collisionManager)
                        // No direct accumulator binding; provided via argument buffer
                    }
                    
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            } else if materialParameters.currentMaterialMode == .neoHookeanElastic {
                // Elastic G2P: Neo-Hookean elastic material transfer
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        gridToParticlesElasticPipelineState
                    )
                    computeEncoder.setBuffer(scene.getComputeParticleBuffer(), offset: 0, index: 0)
                    computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
                    computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 2)
                    
                    // Set collision resources if available
                    if let collisionManager {
                        setupSdfArgumentBufferForCompute(computeEncoder: computeEncoder, collisionManager: collisionManager)
                        // No direct accumulator binding
                    }
                    
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            } else { // .rigidBody
                // Rigid body mode: handled separately via dedicated rigid body simulation
                // G2P is skipped for rigid bodies as they use projection-based dynamics
                print("🔶 Rigid body simulation mode - G2P skipped, using projection-based dynamics")
            }
        }
    }
        
    // MARK: - Buffer Copy Helper
    private func copyBuffersWithBlit(from sourceBuffer: MTLBuffer, to destinationBuffer: MTLBuffer, size: Int, label: String) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            print("⚠️ Failed to create command buffer or blit encoder for \(label)")
            return
        }
        
        blitEncoder.copy(
            from: sourceBuffer,
            sourceOffset: 0,
            to: destinationBuffer,
            destinationOffset: 0,
            size: size
        )
        blitEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // 2-stage pipeline management
    public func beginCompute() {
        // Copy render buffers to compute buffers before starting computation
        copyRenderBuffersToCompute()
        isComputing = true
    }
    
    public func endComputeAndSwapToRender() {
        guard isComputing else {
            print("⚠️ endComputeAndSwapToRender called but not computing")
            return
        }
        
        // Swap compute and render buffers instead of copying
        swapComputeAndRenderBuffers()
        isComputing = false
    }
    
    private func copyRenderBuffersToCompute() {
        let particleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        // Copy particle data from render to compute
        copyBuffersWithBlit(
            from: scene.getRenderParticleBuffer(),
            to: scene.getComputeParticleBuffer(),
            size: particleBufferSize,
            label: "render to compute particles"
        )
    }
    
    private func swapComputeAndRenderBuffers() {
        scene.swapComputeRenderBuffers()
    }
    
    

}
