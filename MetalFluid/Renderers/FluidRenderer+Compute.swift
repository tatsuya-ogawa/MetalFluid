import Metal
import MetalKit
import simd

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
    
    internal func setupComputePipelinesRigid(library: MTLLibrary) {
        particlesToGridRigidPipelineState = createComputePipelineState(library: library, functionName: "particlesToGridRigid")
        gridToParticlesRigid1PipelineState = createComputePipelineState(library: library, functionName: "gridToParticlesRigid1")
        gridToParticlesRigid2PipelineState = createComputePipelineState(library: library, functionName: "gridToParticlesRigid2")
        gridToParticlesRigid3PipelineState = createComputePipelineState(library: library, functionName: "gridToParticlesRigid3")
        gridToParticlesRigid4PipelineState = createComputePipelineState(library: library, functionName: "gridToParticlesRigid4")
        
        // Optional collision solver (only used if multiple rigid bodies)
        if library.makeFunction(name: "solveRigidBodyCollisions") != nil {
            solveRigidBodyCollisionsPipelineState = createComputePipelineState(library: library, functionName: "solveRigidBodyCollisions")
        }
    }
    internal func setupComputePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }

        clearGridPipelineState = createComputePipelineState(library: library, functionName: "clearGrid")
        updateGridVelocityPipelineState = createComputePipelineState(library: library, functionName: "updateGridVelocity")
        
        setupComputePipelinesFluid(library: library)
        setupComputePipelinesElastic(library: library)
        setupComputePipelinesRigid(library: library)
    }
    
    internal func setupParticles() {
        // Setup particles in staging buffer (shared memory for CPU access)
        let stagingParticlePointer = particleStagingBuffer.contents().bindMemory(
            to: MPMParticle.self,
            capacity: particleCount
        )
        
        // Get center and range of boundaryMin/boundaryMax
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let center = (boundaryMin + boundaryMax) * 0.5
        let range = (boundaryMax - boundaryMin) * 0.5
        
        // Rigid info staging pointer
        let stagingRigidInfoPointer = rigidInfoStagingBuffer.contents().bindMemory(
            to: MPMParticleRigidInfo.self,
            capacity: particleCount
        )

        if materialParameters.currentMaterialMode == .neoHookeanElastic {
            // Dense cube formation for elastic and rigid body materials
            setupElasticCube(particlePointer: stagingParticlePointer, center: center, range: range)
            
        } else if materialParameters.currentMaterialMode == .rigidBody{
            setupElasticCube(particlePointer: stagingParticlePointer, center: center, range: range, rigidInfoPointer: stagingRigidInfoPointer)
            // Initialize rigid body states if in rigid body mode
            initializeRigidBodyStatesCPU(particlePointer: stagingParticlePointer, rigidInfoPointer: stagingRigidInfoPointer, center: center)
        } else {
            // Original spherical distribution for fluid
            setupFluidSphere(particlePointer: stagingParticlePointer, center: center, range: range, boundaryMin: boundaryMin, boundaryMax: boundaryMax, rigidInfoPointer: stagingRigidInfoPointer)
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
        let rigidInfoBufferSize = MemoryLayout<MPMParticleRigidInfo>.stride * particleCount
        
        // Copy staging particle buffer to both compute and render private buffers
        blitEncoder.copy(
            from: particleStagingBuffer, sourceOffset: 0,
            to: computeParticleBuffer, destinationOffset: 0,
            size: particleBufferSize
        )
        blitEncoder.copy(
            from: particleStagingBuffer, sourceOffset: 0,
            to: renderParticleBuffer, destinationOffset: 0,
            size: particleBufferSize
        )
        
        // Copy staging rigid info to compute rigid info private buffer
        blitEncoder.copy(
            from: rigidInfoStagingBuffer, sourceOffset: 0,
            to: computeRigidInfoBuffer, destinationOffset: 0,
            size: rigidInfoBufferSize
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
    
    private func createRigidInfo(rigidId: UInt32, initialOffset: SIMD3<Float>) -> MPMParticleRigidInfo {
        return MPMParticleRigidInfo(rigidId: rigidId, initialOffset: initialOffset)
    }
    
    private func setupElasticCube(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, rigidInfoPointer: UnsafeMutablePointer<MPMParticleRigidInfo>? = nil) {
        let particlesPerDim = Int(floor(pow(Float(particleCount), 1.0/3.0)))
        let cubeSize = min(range.x, range.y, range.z) * 0.6
        let spacing = cubeSize / Float(particlesPerDim - 1)
        let cubeOrigin = center - SIMD3<Float>(cubeSize * 0.5, cubeSize * 0.5, cubeSize * 0.5)
        let isRigidBody = materialParameters.currentMaterialMode == .rigidBody
        
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
                    rigidInfoPointer?[particleIndex] = createRigidInfo(
                        rigidId: isRigidBody ? 1 : 0,
                        initialOffset: isRigidBody ? (finalPos - center) : SIMD3<Float>(0, 0, 0)
                    )
                    
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
            rigidInfoPointer?[particleIndex] = createRigidInfo(
                rigidId: 0,
                initialOffset: isRigidBody ? (pos - center) : SIMD3<Float>(0, 0, 0)
            )
            
            particleIndex += 1
        }
        
        print("🟦 Created elastic cube: \(particlesPerDim)³ lattice, spacing: \(spacing)")
    }
    
    private func setupFluidSphere(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, boundaryMin: SIMD3<Float>, boundaryMax: SIMD3<Float>, rigidInfoPointer: UnsafeMutablePointer<MPMParticleRigidInfo>? = nil) {
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
            rigidInfoPointer?[i] = createRigidInfo(rigidId: 0, initialOffset: randomOffset)
        }
        
        print("🌊 Created fluid sphere: radius \(maxRadius)")
    }
    
    private func initializeRigidBodyStates() {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // Set up pipeline for rigid body initialization
        guard let library = device.makeDefaultLibrary(),
              let initFunction = library.makeFunction(name: "initializeRigidBodies") else {
            print("⚠️ Could not find initializeRigidBodies function")
            return
        }
        
        do {
            let initPipelineState = try device.makeComputePipelineState(function: initFunction)
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(initPipelineState)
                computeEncoder.setBuffer(rigidBodyStateBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 2)
                // Rigid info buffer now required by initializeRigidBodies kernel
                computeEncoder.setBuffer(computeRigidInfoBuffer, offset: 0, index: 3)
                
                let rigidBodyCount = materialParameters.currentMaterialMode == .rigidBody ? 1 : 0
                let threadgroupSize = min(initPipelineState.maxTotalThreadsPerThreadgroup, 256)
                let threadgroups = MTLSize(
                    width: (rigidBodyCount + threadgroupSize - 1) / threadgroupSize,
                    height: 1,
                    depth: 1
                )
                let threadsPerThreadgroup = MTLSize(
                    width: threadgroupSize,
                    height: 1,
                    depth: 1
                )
                
                computeEncoder.dispatchThreadgroups(
                    threadgroups,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
                computeEncoder.endEncoding()
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            print("🔶 Initialized rigid body states")
            
        } catch {
            print("⚠️ Could not create rigid body initialization pipeline: \(error)")
        }
    }
    // CPU-side rigid body initialization
    private func initializeRigidBodyStatesCPU(particlePointer: UnsafeMutablePointer<MPMParticle>, rigidInfoPointer: UnsafeMutablePointer<MPMParticleRigidInfo>, center: SIMD3<Float>) {
        let rigidBodyStatePointer = rigidBodyStateBuffer.contents().bindMemory(to: RigidBodyState.self, capacity: 1)
        let rigidBodyId: UInt32 = 1 // We're initializing rigid body with ID 1
        
        // Calculate center of mass and total mass
        var centerOfMass = SIMD3<Float>(0, 0, 0)
        var totalMass: Float = 0.0
        var particleCount: UInt32 = 0
        
        for i in 0..<self.particleCount {
            if rigidInfoPointer[i].rigidId == rigidBodyId {
                let particle = particlePointer[i]
                centerOfMass += particle.position * particle.mass
                totalMass += particle.mass
                particleCount += 1
            }
        }
        
        if totalMass > 0.0 {
            centerOfMass /= totalMass
        }
        
        // Initialize rigid body state
        let rigidBodyState = RigidBodyState(
            centerOfMass: centerOfMass,
            linearVelocity: SIMD3<Float>(0, 0, 0),
            angularVelocity: SIMD3<Float>(0, 0, 0),
            orientation: SIMD4<Float>(0, 0, 0, 1), // Identity quaternion
            totalMass: totalMass,
            invInertiaTensor: simd_float3x3(1.0), // Identity matrix for now
            accumulatedForce: SIMD3<Float>(0, 0, 0),
            accumulatedTorque: SIMD3<Float>(0, 0, 0),
            particleCount: particleCount,
            isActive: (particleCount > 0) ? 1 : 0,
            linearDamping: 0.99,
            angularDamping: 0.99,
            restitution: 0.5,
            friction: 0.3,
            halfExtents: SIMD3<Float>(0, 0, 0),
            boundingRadius: 0.5
        )
        // Write to buffer
        rigidBodyStatePointer[0] = rigidBodyState
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
                try sortManager.sortParticlesByGridIndexSafe(
                    computeParticleBuffer: &computeParticleBuffer,
                    uniformBuffer: computeUniformBuffer
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
        
        // Add completion handler to swap buffers when the shared command buffer finishes
        // (This will be called when the render pass commits the command buffer)
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.endComputeAndSwapToRender()
        }
    }
    
    // MARK: - MPM Simulation Pipeline
    
    internal func computeSimulation(commandBuffer: MTLCommandBuffer) {
        // Decide threadgroup sizes based on each pipeline's reported limits
        let gridThreadgroupSize = min(
            clearGridPipelineState.maxTotalThreadsPerThreadgroup,
            256
        )
        let particleThreadgroupSize = min(
            particlesToGridFluid1PipelineState.maxTotalThreadsPerThreadgroup,
            256
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

        for _ in 0..<substeps {
            // 1. Clear grid
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(clearGridPipelineState)
                computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
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
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
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
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
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
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            } else { // .rigidBody
                // Rigid Body P2G: Rigid body material transfer
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        particlesToGridRigidPipelineState
                    )
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            }

            // 3. Update grid velocity - Grid velocity update and boundary condition application
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(
                    updateGridVelocityPipelineState
                )
                computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
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
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
                    
                    // Set collision resources if available
                    if let collisionManager {
                        computeEncoder.setBuffer(collisionManager.getCollisionUniformBuffer(), offset: 0, index: 3)
                        
                        if let sdfTexture = collisionManager.getSDFTexture() {
                            computeEncoder.setTexture(sdfTexture, index: 0)
                        }
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
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
                    
                    // Set collision resources if available
                    if let collisionManager {
                        computeEncoder.setBuffer(collisionManager.getCollisionUniformBuffer(), offset: 0, index: 3)
                        
                        if let sdfTexture = collisionManager.getSDFTexture() {
                            computeEncoder.setTexture(sdfTexture, index: 0)
                        }
                    }
                    
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
            } else { // .rigidBody
                // Rigid Body G2P Stage 1: Basic G2P transfer (gridToParticlesRigid1)
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        gridToParticlesRigid1PipelineState
                    )
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
                    
                    // Set collision resources if available
                    if let collisionManager {
                        computeEncoder.setBuffer(collisionManager.getCollisionUniformBuffer(), offset: 0, index: 3)
                        
                        if let sdfTexture = collisionManager.getSDFTexture() {
                            computeEncoder.setTexture(sdfTexture, index: 0)
                        }
                    }
                    
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
                
                // Rigid Body G2P Stage 2: Accumulate forces and torques (gridToParticlesRigid2)
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        gridToParticlesRigid2PipelineState
                    )
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(rigidBodyStateBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(computeRigidInfoBuffer, offset: 0, index: 3)
                    
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
                
                // Rigid Body G2P Stage 3: Update rigid body dynamics (gridToParticlesRigid3)
                let rigidBodyCount = materialParameters.currentMaterialMode == .rigidBody ? 1 : 0
                if rigidBodyCount > 0 {
                    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                        computeEncoder.setComputePipelineState(
                            gridToParticlesRigid3PipelineState
                        )
                        computeEncoder.setBuffer(rigidBodyStateBuffer, offset: 0, index: 0)
                        computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                        
                        let rigidBodyThreadgroups = MTLSize(
                            width: (rigidBodyCount + particleThreadgroupSize - 1) / particleThreadgroupSize,
                            height: 1,
                            depth: 1
                        )
                        
                        computeEncoder.dispatchThreadgroups(
                            rigidBodyThreadgroups,
                            threadsPerThreadgroup: particleThreadsPerThreadgroup
                        )
                        computeEncoder.endEncoding()
                    }

                    // Collision solve between rigid bodies (only if >1 bodies)
                    if rigidBodyCount > 1, let collisionPSO = solveRigidBodyCollisionsPipelineState {
                        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                            computeEncoder.setComputePipelineState(collisionPSO)
                            computeEncoder.setBuffer(rigidBodyStateBuffer, offset: 0, index: 0)
                            computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                            let pairCount = rigidBodyCount * (rigidBodyCount - 1) / 2
                            let tgSize = 64
                            let tgCount = (pairCount + tgSize - 1) / tgSize
                            let threadgroups = MTLSize(width: tgCount, height: 1, depth: 1)
                            let threadsPerGroup = MTLSize(width: tgSize, height: 1, depth: 1)
                            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
                            computeEncoder.endEncoding()
                        }
                    }
                }
                
                // Rigid Body G2P Stage 4: Project particles to maintain rigid body constraints (gridToParticlesRigid4)
                if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(
                        gridToParticlesRigid4PipelineState
                    )
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(rigidBodyStateBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 2)
                    // gridToParticlesRigid4 uses the per-particle rigidInfo for initial offsets
                    computeEncoder.setBuffer(computeRigidInfoBuffer, offset: 0, index: 3)
                    
                    computeEncoder.dispatchThreadgroups(
                        particleThreadgroups,
                        threadsPerThreadgroup: particleThreadsPerThreadgroup
                    )
                    computeEncoder.endEncoding()
                }
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
            from: renderParticleBuffer,
            to: computeParticleBuffer,
            size: particleBufferSize,
            label: "render to compute particles"
        )
    }
    
    private func swapComputeAndRenderBuffers() {
        // Swap particle buffers using Swift's swap function
        swap(&computeParticleBuffer, &renderParticleBuffer)
        
        // Swap grid buffers (no initial copy needed like particles)
        swap(&computeGridBuffer, &renderGridBuffer)
                
        // Update buffer labels for debugging
        computeParticleBuffer.label = "ComputeParticleBuffer"
        renderParticleBuffer.label = "RenderParticleBuffer"
        computeGridBuffer.label = "ComputeGridBuffer"
        renderGridBuffer.label = "RenderGridBuffer"
        computeRigidInfoBuffer.label = "ComputeRigidInfoBuffer"
    }
    
    
    public func getCurrentBufferInfo() -> (stage: String, computeBuffer: String, renderBuffer: String) {
        return (
            stage: isComputing ? "Computing" : "Rendering",
            computeBuffer: computeParticleBuffer?.label ?? "nil",
            renderBuffer: renderParticleBuffer?.label ?? "nil"
        )
    }

}
