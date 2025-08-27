import Metal
import MetalKit
import simd

// MARK: - Compute Extension
extension MPMFluidRenderer {
    
    // MARK: - Setup Functions
    internal func setupComputePipelinesFluid(library:MTLLibrary){
        // Particles to grid 1 pipeline
        guard
            let particlesToGridFluid1Function = library.makeFunction(
                name: "particlesToGridFluid1"
            )
        else {
            fatalError("Could not find function 'particlesToGridFluid1'")
        }
        do {
            particlesToGridFluid1PipelineState = try device.makeComputePipelineState(
                function: particlesToGridFluid1Function
            )
        } catch {
            fatalError(
                "Could not create particlesToGridFluid1 pipeline state: \(error)"
            )
        }
        // Particles to grid 2 pipeline
        guard
            let particlesToGridFluid2Function = library.makeFunction(
                name: "particlesToGridFluid2"
            )
        else {
            fatalError("Could not find function 'particlesToGridFluid2'")
        }
        do {
            particlesToGridFluid2PipelineState = try device.makeComputePipelineState(
                function: particlesToGridFluid2Function
            )
        } catch {
            fatalError(
                "Could not create particlesToGridFluid2 pipeline state: \(error)"
            )
        }
        // Grid to particles pipeline
        guard
            let gridToParticlesFluid1Function = library.makeFunction(
                name: "gridToParticlesFluid1"
            )
        else {
            fatalError("Could not find function 'gridToParticlesFluid1'")
        }
        do {
            gridToParticlesFluid1PipelineState = try device.makeComputePipelineState(
                function: gridToParticlesFluid1Function
            )
        } catch {
            fatalError(
                "Could not create gridToParticlesFluid1 pipeline state: \(error)"
            )
        }
    }
    internal func setupComputePipelinesElastic(library: MTLLibrary) {
        // Elastic material pipeline states
        guard let particlesToGridElasticFunction = library.makeFunction(name: "particlesToGridElastic") else {
            fatalError("Could not find function 'particlesToGridElastic'")
        }
        do {
            particlesToGridElasticPipelineState = try device.makeComputePipelineState(function: particlesToGridElasticFunction)
        } catch {
            fatalError("Could not create particlesToGridElastic pipeline state: \(error)")
        }
        guard let gridToParticlesElasticFunction = library.makeFunction(name: "gridToParticlesElastic") else {
            fatalError("Could not find function 'gridToParticlesElastic'")
        }
        do {
            gridToParticlesElasticPipelineState = try device.makeComputePipelineState(function: gridToParticlesElasticFunction)
        } catch {
            fatalError("Could not create gridToParticlesElastic pipeline state: \(error)")
        }
    }
    internal func setupComputePipelinesRigid(library: MTLLibrary) {
        // Rigid body material pipeline states
        guard let particlesToGridRigidFunction = library.makeFunction(name: "particlesToGridRigid") else {
            fatalError("Could not find function 'particlesToGridRigid'")
        }
        do {
            particlesToGridRigidPipelineState = try device.makeComputePipelineState(function: particlesToGridRigidFunction)
        } catch {
            fatalError("Could not create particlesToGridRigid pipeline state: \(error)")
        }
        guard let gridToParticlesRigid1Function = library.makeFunction(name: "gridToParticlesRigid1") else {
            fatalError("Could not find function 'gridToParticlesRigid1'")
        }
        do {
            gridToParticlesRigid1PipelineState = try device.makeComputePipelineState(function: gridToParticlesRigid1Function)
        } catch {
            fatalError("Could not create gridToParticlesRigid1 pipeline state: \(error)")
        }
        guard let gridToParticlesRigid2Function = library.makeFunction(name: "gridToParticlesRigid2") else {
            fatalError("Could not find function 'gridToParticlesRigid2'")
        }
        do {
            gridToParticlesRigid2PipelineState = try device.makeComputePipelineState(function: gridToParticlesRigid2Function)
        } catch {
            fatalError("Could not create gridToParticlesRigid2 pipeline state: \(error)")
        }
        guard let gridToParticlesRigid3Function = library.makeFunction(name: "gridToParticlesRigid3") else {
            fatalError("Could not find function 'gridToParticlesRigid3'")
        }
        do {
            gridToParticlesRigid3PipelineState = try device.makeComputePipelineState(function: gridToParticlesRigid3Function)
        } catch {
            fatalError("Could not create gridToParticlesRigid3 pipeline state: \(error)")
        }
        guard let gridToParticlesRigid4Function = library.makeFunction(name: "gridToParticlesRigid4") else {
            fatalError("Could not find function 'gridToParticlesRigid4'")
        }
        do {
            gridToParticlesRigid4PipelineState = try device.makeComputePipelineState(function: gridToParticlesRigid4Function)
        } catch {
            fatalError("Could not create gridToParticlesRigid4 pipeline state: \(error)")
        }
    }
    internal func setupComputePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }

        // Clear grid pipeline
        guard let clearGridFunction = library.makeFunction(name: "clearGrid")
        else {
            fatalError("Could not find function 'clearGrid'")
        }
        do {
            clearGridPipelineState = try device.makeComputePipelineState(
                function: clearGridFunction
            )
        } catch {
            fatalError("Could not create clear grid pipeline state: \(error)")
        }

        
        // Update grid velocity pipeline
        guard
            let updateGridVelocityFunction = library.makeFunction(
                name: "updateGridVelocity"
            )
        else {
            fatalError("Could not find function 'updateGridVelocity'")
        }
        do {
            updateGridVelocityPipelineState =
                try device.makeComputePipelineState(
                    function: updateGridVelocityFunction
                )
        } catch {
            fatalError(
                "Could not create update grid velocity pipeline state: \(error)"
            )
        }
        setupComputePipelinesFluid(library: library)
        setupComputePipelinesElastic(library: library)
        setupComputePipelinesRigid(library: library)
    }
    
    internal func setupParticles() {
        // Setup particles in compute buffer (primary)
        let computeParticlePointer = computeParticleBuffer.contents().bindMemory(
            to: MPMParticle.self,
            capacity: particleCount
        )
        
        // Get center and range of boundaryMin/boundaryMax
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let center = (boundaryMin + boundaryMax) * 0.5
        let range = (boundaryMax - boundaryMin) * 0.5
        
        // Rigid info pointer
        let computeRigidInfoPointer = computeRigidInfoBuffer.contents().bindMemory(
            to: MPMParticleRigidInfo.self,
            capacity: particleCount
        )

        if materialParameters.currentMaterialMode == .neoHookeanElastic {
            // Dense cube formation for elastic and rigid body materials
            setupElasticCube(particlePointer: computeParticlePointer, center: center, range: range)
            
        } else if materialParameters.currentMaterialMode == .rigidBody{
            setupElasticCube(particlePointer: computeParticlePointer,center: center, range: range, rigidInfoPointer: computeRigidInfoPointer)
            // Initialize rigid body states if in rigid body mode
            initializeRigidBodyStates()
        } else {
            // Original spherical distribution for fluid
            setupFluidSphere(particlePointer: computeParticlePointer, center: center, range: range, boundaryMin: boundaryMin, boundaryMax: boundaryMax)
        }
        
        // Copy initial data from compute buffer to render buffer
        let particleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        let renderParticlePointer = renderParticleBuffer.contents()
        memcpy(renderParticlePointer, computeParticlePointer, particleBufferSize)

        // Copy rigid info to render rigid info buffer
        let rigidInfoSize = MemoryLayout<MPMParticleRigidInfo>.stride * particleCount
        let computeRigidPtr = computeRigidInfoBuffer.contents()
        let renderRigidPtr = renderRigidInfoBuffer.contents()
        memcpy(renderRigidPtr, computeRigidPtr, rigidInfoSize)
        
        print("🔄 Initialized particles in both compute and render buffers")
    }
    
    private func setupElasticCube(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>,rigidInfoPointer:UnsafeMutablePointer<MPMParticleRigidInfo>?=nil) {
        // Calculate cube dimensions based on particle count
        let particlesPerDim = Int(ceil(pow(Float(particleCount), 1.0/3.0)))
        
        // Cube size - make it smaller than the boundary to leave space
        let cubeSize = min(range.x, range.y, range.z) * 0.6  // 60% of available space
        let spacing = cubeSize / Float(particlesPerDim - 1)
        
        // Calculate cube origin (bottom-left-back corner)
        let cubeOrigin = center - SIMD3<Float>(cubeSize * 0.5, cubeSize * 0.5, cubeSize * 0.5)
        
        var particleIndex = 0
        
        for x in 0..<particlesPerDim {
            for y in 0..<particlesPerDim {
                for z in 0..<particlesPerDim {
                    if particleIndex >= particleCount { break }
                    
                    // Calculate position in the dense cube
                    let pos = cubeOrigin + SIMD3<Float>(
                        Float(x) * spacing,
                        Float(y) * spacing,
                        Float(z) * spacing
                    )
                    
                    // Add very small random offset for numerical stability
                    let randomOffset = SIMD3<Float>(
                        Float.random(in: -0.001...0.001),
                        Float.random(in: -0.001...0.001),
                        Float.random(in: -0.001...0.001)
                    )
                    
                    particlePointer[particleIndex] = MPMParticle(
                        position: pos + randomOffset,
                        velocity: SIMD3<Float>(0.0, 0.0, 0.0),
                        C: simd_float3x3(0.0),  // Affine momentum matrix initialization
                        mass: materialParameters.particleMass,
                        originalIndex: UInt32(particleIndex),
                    )
                    if let rigidInfoPointer{
                        rigidInfoPointer[particleIndex] = MPMParticleRigidInfo(
                            rigidId: materialParameters.currentMaterialMode == .rigidBody ? 1 : 0,  // Assign to rigid body 1 if rigid body mode
                            initialOffset: materialParameters.currentMaterialMode == .rigidBody ? (pos + randomOffset - center) : SIMD3<Float>(0, 0, 0)
                        )
                    }
                    
                    particleIndex += 1
                }
                if particleIndex >= particleCount { break }
            }
            if particleIndex >= particleCount { break }
        }
        
        // Fill remaining particles with random positions within cube if needed
        while particleIndex < particleCount {
            let pos = cubeOrigin + SIMD3<Float>(
                Float.random(in: 0...cubeSize),
                Float.random(in: 0...cubeSize),
                Float.random(in: 0...cubeSize)
            )
            
            particlePointer[particleIndex] = MPMParticle(
                position: pos,
                velocity: SIMD3<Float>(0.0, 0.0, 0.0),
                C: simd_float3x3(0.0),
                mass: materialParameters.particleMass,
                originalIndex: UInt32(particleIndex),
            )
            if let rigidInfoPointer{
                rigidInfoPointer[particleIndex] = MPMParticleRigidInfo(
                    rigidId: materialParameters.currentMaterialMode == .rigidBody ? 1 : 0,  // Assign to rigid body 1 if rigid body mode
                    initialOffset: materialParameters.currentMaterialMode == .rigidBody ? (pos - center) : SIMD3<Float>(0, 0, 0)
                )
            }
            
            particleIndex += 1
        }
        
        print("🟦 Created elastic cube: \(particlesPerDim)³ lattice, spacing: \(spacing)")
    }
    
    private func setupFluidSphere(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, boundaryMin: SIMD3<Float>, boundaryMax: SIMD3<Float>,rigidInfoPointer:UnsafeMutablePointer<MPMParticleRigidInfo>?=nil) {
        let maxRadius = min(range.x, range.y, range.z)
        
        func randn() -> Float {
            // Standard normal distribution using Box-Muller method
            let u1 = Float.random(in: 0..<1)
            let u2 = Float.random(in: 0..<1)
            return sqrt(-2.0 * log(u1 + 1e-7)) * cos(2.0 * .pi * u2)
        }
        
        for i in 0..<particleCount {
            // Spherical normal distribution sampling
            var pos: SIMD3<Float>
            while true {
                // 3D normal distribution
                let x = randn()
                let y = randn()
                let z = randn()
                let v = SIMD3<Float>(x, y, z)
                // Constrain to sphere (within maxRadius from center)
                let r = length(v)
                if r > 3.0 { continue } // Discard outside 3σ
                // Normalize sphere radius to maxRadius
                let scaled = v / r * (r / 3.0) * maxRadius
                pos = center + scaled
                // Accept if within boundaries
                if all(pos .>= boundaryMin) && all(pos .<= boundaryMax) {
                    break
                }
            }
            // Small random noise
            let randomOffset = SIMD3<Float>(
                Float.random(in: -0.003...0.003),
                Float.random(in: -0.003...0.003),
                Float.random(in: -0.003...0.003)
            )
            let finalPos = simd_clamp(pos + randomOffset, boundaryMin, boundaryMax)
            
            particlePointer[i] = MPMParticle(
                position: finalPos,
                velocity: SIMD3<Float>(0.0, 0.0, 0.0),  // Initial velocity is 0
                C: simd_float3x3(0.0),  // Affine momentum matrix initialization
                mass: materialParameters.particleMass,
                originalIndex: UInt32(i),
            )
            if let rigidInfoPointer{
                rigidInfoPointer[i] = MPMParticleRigidInfo(
                    rigidId: 0,  // Fluid particles don't belong to rigid bodies
                    initialOffset: randomOffset
                )
            }
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
        
        // End compute and copy results to render buffers
        endComputeAndSwapToRender()
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
                computeEncoder.setBuffer(gridBuffer, offset: 0, index: 0)
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
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
                computeEncoder.setBuffer(gridBuffer, offset: 0, index: 0)
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
                    
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
                    
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
                    computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
                    
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
        let uniformBufferSize = MemoryLayout<ComputeShaderUniforms>.stride
        let rigidInfoSize = MemoryLayout<MPMParticleRigidInfo>.stride * particleCount
        
        // Copy particle data from render to compute
        copyBuffersWithBlit(
            from: renderParticleBuffer,
            to: computeParticleBuffer,
            size: particleBufferSize,
            label: "render to compute particles"
        )
        
        // Copy rigid info from render to compute
        copyBuffersWithBlit(
            from: renderRigidInfoBuffer,
            to: computeRigidInfoBuffer,
            size: rigidInfoSize,
            label: "render to compute rigid info"
        )
        print("🔄 Copied render buffers to compute buffers (\(particleBufferSize + uniformBufferSize + rigidInfoSize) bytes)")
    }
    
    private func swapComputeAndRenderBuffers() {
        // Swap particle buffers using Swift's swap function
        swap(&computeParticleBuffer, &renderParticleBuffer)
        
        // Swap rigid info buffers using Swift's swap function
        swap(&computeRigidInfoBuffer, &renderRigidInfoBuffer)
        
        // Update buffer labels for debugging
        computeParticleBuffer.label = "ComputeParticleBuffer"
        renderParticleBuffer.label = "RenderParticleBuffer"
        computeRigidInfoBuffer.label = "ComputeRigidInfoBuffer"
        renderRigidInfoBuffer.label = "RenderRigidInfoBuffer"
    }
    
    
    public func getCurrentBufferInfo() -> (stage: String, computeBuffer: String, renderBuffer: String) {
        return (
            stage: isComputing ? "Computing" : "Rendering",
            computeBuffer: computeParticleBuffer?.label ?? "nil",
            renderBuffer: renderParticleBuffer?.label ?? "nil"
        )
    }

}
