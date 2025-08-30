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
        accumulateRigidBodyForcesPipelineState = createComputePipelineState(library: library, functionName: "accumulateRigidBodyForces")
        updateRigidBodyDynamicsPipelineState = createComputePipelineState(library: library, functionName: "updateRigidBodyDynamics")
        projectRigidBodyParticlesPipelineState = createComputePipelineState(library: library, functionName: "projectRigidBodyParticles")
        
        // Optional collision solver (only used if multiple rigid bodies)
        if library.makeFunction(name: "solveRigidBodyCollisions") != nil {
            solveRigidBodyCollisionsPipelineState = createComputePipelineState(library: library, functionName: "solveRigidBodyCollisions")
        }
    }
    
    internal func setupSDFCollisionPipelines(library: MTLLibrary) {
        // SDF Collision (Projection-Based Dynamics) pipelines
        if library.makeFunction(name: "processParticleSDFCollisions") != nil {
            processParticleSDFCollisionsPipelineState = createComputePipelineState(library: library, functionName: "processParticleSDFCollisions")
        }
        
        if library.makeFunction(name: "solveParticleConstraintsIterative") != nil {
            solveParticleConstraintsIterativePipelineState = createComputePipelineState(library: library, functionName: "solveParticleConstraintsIterative")
        }
        
        if library.makeFunction(name: "processRigidBodySDFCollisions") != nil {
            processRigidBodySDFCollisionsPipelineState = createComputePipelineState(library: library, functionName: "processRigidBodySDFCollisions")
        }
        
        if library.makeFunction(name: "solveRigidBodyConstraintsIterative") != nil {
            solveRigidBodyConstraintsIterativePipelineState = createComputePipelineState(library: library, functionName: "solveRigidBodyConstraintsIterative")
        }
        
        if library.makeFunction(name: "solveRigidBodyToRigidBodyCollisions") != nil {
            solveRigidBodyToRigidBodyCollisionsPipelineState = createComputePipelineState(library: library, functionName: "solveRigidBodyToRigidBodyCollisions")
        }
    }
    internal func setupComputePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }

        clearGridPipelineState = createComputePipelineState(library: library, functionName: "clearGrid")
        updateGridVelocityPipelineState = createComputePipelineState(library: library, functionName: "updateGridVelocity")

        // Force application pipeline
        applyForceToGridPipelineState = createComputePipelineState(library: library, functionName: "applyForceToGrid")

        setupComputePipelinesFluid(library: library)
        setupComputePipelinesElastic(library: library)
        setupComputePipelinesRigid(library: library)
        setupSDFCollisionPipelines(library: library)
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
        let isRigidBody = materialParameters.currentMaterialMode == .rigidBody
        
        if isRigidBody {
            // SDF-based rigid body initialization
            setupSDFBasedRigidBody(particlePointer: particlePointer, center: center, range: range, rigidInfoPointer: rigidInfoPointer)
        } else {
            // Original elastic cube initialization
            setupOriginalElasticCube(particlePointer: particlePointer, center: center, range: range, rigidInfoPointer: rigidInfoPointer)
        }
    }
    
    private func setupSDFBasedRigidBody(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, rigidInfoPointer: UnsafeMutablePointer<MPMParticleRigidInfo>?) {
        guard let collisionManager = collisionManager,
              collisionManager.isEnabled(),
              let sdfTexture = collisionManager.getSDFTexture() else {
            print("⚠️ SDF not available, falling back to cube initialization")
            setupOriginalElasticCube(particlePointer: particlePointer, center: center, range: range, rigidInfoPointer: rigidInfoPointer)
            return
        }
        
        let collisionUniformPointer = collisionManager.getCollisionUniformBuffer().contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        let sdfOrigin = collisionUniformPointer[0].sdfOrigin
        let sdfSize = collisionUniformPointer[0].sdfSize
        let sdfResolution = collisionUniformPointer[0].sdfResolution
        let transform = collisionUniformPointer[0].collisionTransform
        
        // Calculate SDF center in world space
        let sdfLocalCenter = sdfOrigin + sdfSize * 0.5
        let sdfWorldCenter4 = transform * SIMD4<Float>(sdfLocalCenter.x, sdfLocalCenter.y, sdfLocalCenter.z, 1.0)
        let sdfWorldCenter = SIMD3<Float>(sdfWorldCenter4.x, sdfWorldCenter4.y, sdfWorldCenter4.z)
        
        var particleIndex = 0
        let maxAttempts = particleCount * 10 // Prevent infinite loop
        var attempts = 0
        
        print("🔶 Initializing SDF-based rigid body with \(particleCount) particles")
        print("   SDF origin: \(sdfOrigin), size: \(sdfSize)")
        print("   SDF world center: \(sdfWorldCenter)")
        
        // Generate particles inside the SDF volume
        while particleIndex < particleCount && attempts < maxAttempts {
            attempts += 1
            
            // Generate random position within expanded boundary range
            let expandedRange = range * 1.2 // Slightly expand search area
            let pos = center + SIMD3<Float>(
                Float.random(in: -expandedRange.x...expandedRange.x),
                Float.random(in: -expandedRange.y...expandedRange.y),
                Float.random(in: -expandedRange.z...expandedRange.z)
            )
            
            // Check if this position is inside the SDF (negative SDF value = inside)
            if isPositionInsideSDF(position: pos, sdfTexture: sdfTexture, 
                                  sdfOrigin: sdfOrigin, sdfSize: sdfSize, 
                                  sdfResolution: sdfResolution, 
                                  transform: collisionUniformPointer[0].collisionInvTransform) {
                
                let randomOffset = SIMD3<Float>(
                    Float.random(in: -0.001...0.001),
                    Float.random(in: -0.001...0.001),
                    Float.random(in: -0.001...0.001)
                )
                let finalPos = pos + randomOffset
                
                particlePointer[particleIndex] = createParticle(at: finalPos, index: particleIndex)
                rigidInfoPointer?[particleIndex] = createRigidInfo(
                    rigidId: 1,
                    initialOffset: finalPos - sdfWorldCenter
                )
                
                particleIndex += 1
            }
        }
        
        // Fill any remaining particles with fallback method if SDF sampling didn't generate enough
        if particleIndex < particleCount {
            print("⚠️ SDF sampling only generated \(particleIndex)/\(particleCount) particles, filling remainder randomly")
            while particleIndex < particleCount {
                let pos = center + SIMD3<Float>(
                    Float.random(in: -range.x * 0.3...range.x * 0.3),
                    Float.random(in: -range.y * 0.3...range.y * 0.3),
                    Float.random(in: -range.z * 0.3...range.z * 0.3)
                )
                
                particlePointer[particleIndex] = createParticle(at: pos, index: particleIndex)
                rigidInfoPointer?[particleIndex] = createRigidInfo(
                    rigidId: 1,
                    initialOffset: pos - sdfWorldCenter
                )
                
                particleIndex += 1
            }
        }
        
        print("🟦 Created SDF-based rigid body: \(particleIndex) particles in \(attempts) attempts")
    }
    
    private func setupOriginalElasticCube(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, rigidInfoPointer: UnsafeMutablePointer<MPMParticleRigidInfo>?) {
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
    
    // Helper function to check if position is inside SDF (simplified CPU-based check)
    private func isPositionInsideSDF(position: SIMD3<Float>, sdfTexture: MTLTexture, 
                                   sdfOrigin: SIMD3<Float>, sdfSize: SIMD3<Float>, 
                                   sdfResolution: SIMD3<Int32>, transform: float4x4) -> Bool {
        // Transform world position to SDF local space
        let worldPos4 = SIMD4<Float>(position.x, position.y, position.z, 1.0)
        let localPos4 = transform * worldPos4
        let localPos = SIMD3<Float>(localPos4.x, localPos4.y, localPos4.z)
        
        // Check if position is within SDF bounds
        if any(localPos .< sdfOrigin) || any(localPos .> (sdfOrigin + sdfSize)) {
            return false
        }
        
        // Convert to SDF texture coordinates [0, 1]
        let relativePos = (localPos - sdfOrigin) / sdfSize
        let texCoord = SIMD3<Int32>(
            min(Int32(relativePos.x * Float(sdfResolution.x)), sdfResolution.x - 1),
            min(Int32(relativePos.y * Float(sdfResolution.y)), sdfResolution.y - 1),
            min(Int32(relativePos.z * Float(sdfResolution.z)), sdfResolution.z - 1)
        )
        
        // For CPU-based initialization, we'll use a simple heuristic
        // In practice, you might want to read the actual SDF texture data
        // For now, we'll assume positions closer to the center are more likely to be inside
        let centerDist = length(relativePos - SIMD3<Float>(0.5, 0.5, 0.5))
        return centerDist < 0.4 // Rough approximation - particles within 40% of center radius
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
        
        // Get SDF-based properties if available
        var halfExtents = SIMD3<Float>(0.5, 0.5, 0.5) // Default
        var boundingRadius: Float = 0.5 // Default
        var inertiaTensor = simd_float3x3(1.0) // Default identity
        
        if let collisionManager = collisionManager, collisionManager.isEnabled() {
            let collisionUniformPointer = collisionManager.getCollisionUniformBuffer().contents().bindMemory(
                to: CollisionUniforms.self,
                capacity: 1
            )
            let sdfSize = collisionUniformPointer[0].sdfSize
            
            // Use SDF size to calculate more accurate physical properties
            halfExtents = sdfSize * 0.5
            boundingRadius = length(halfExtents)
            
            // Calculate inertia tensor for a box with SDF dimensions
            let mass = max(totalMass, 1.0)
            let w = sdfSize.x, h = sdfSize.y, d = sdfSize.z
            inertiaTensor = simd_float3x3(
                SIMD3<Float>((mass/12.0) * (h*h + d*d), 0, 0),
                SIMD3<Float>(0, (mass/12.0) * (w*w + d*d), 0),
                SIMD3<Float>(0, 0, (mass/12.0) * (w*w + h*h))
            )
            
            print("🔶 Using SDF-based rigid body properties:")
            print("   Half extents: \(halfExtents)")
            print("   Bounding radius: \(boundingRadius)")
            print("   Inertia tensor diagonal: \(inertiaTensor.columns.0.x), \(inertiaTensor.columns.1.y), \(inertiaTensor.columns.2.z)")
        }
        
        // Calculate inverse inertia tensor
        let invInertiaTensor = simd_float3x3(
            SIMD3<Float>(1.0 / max(inertiaTensor.columns.0.x, 1e-6), 0, 0),
            SIMD3<Float>(0, 1.0 / max(inertiaTensor.columns.1.y, 1e-6), 0),
            SIMD3<Float>(0, 0, 1.0 / max(inertiaTensor.columns.2.z, 1e-6))
        )
        
        // Initialize rigid body state with SDF-based properties
        let rigidBodyState = RigidBodyState(
            centerOfMass: centerOfMass,
            linearVelocity: SIMD3<Float>(0, 0, 0),
            angularVelocity: SIMD3<Float>(0, 0, 0),
            orientation: SIMD4<Float>(0, 0, 0, 1), // Identity quaternion
            totalMass: totalMass,
            invInertiaTensor: invInertiaTensor,
            accumulatedForce: SIMD3<Float>(0, 0, 0),
            accumulatedTorque: SIMD3<Float>(0, 0, 0),
            particleCount: particleCount,
            isActive: (particleCount > 0) ? 1 : 0,
            linearDamping: 0.99,
            angularDamping: 0.99,
            restitution: 0.5,
            friction: 0.3,
            halfExtents: halfExtents,
            boundingRadius: boundingRadius
        )
        
        // Write to buffer
        rigidBodyStatePointer[0] = rigidBodyState
        
        print("🔶 Initialized SDF-based rigid body state:")
        print("   Center of mass: \(centerOfMass)")
        print("   Total mass: \(totalMass)")
        print("   Particle count: \(particleCount)")
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
        
        // Run rigid body simulation if in rigid body mode
        if materialParameters.currentMaterialMode == .rigidBody {
            computeRigidBodySimulation(commandBuffer: commandBuffer)
        }
        
        // Add completion handler to swap buffers when the shared command buffer finishes
        // (This will be called when the render pass commits the command buffer)
        commandBuffer.addCompletedHandler { [weak self] _ in
            // Integrate aggregated SDF impulses into collision transform for next frame
            self?.applySDFImpulseAggregationToCollisionTransform()
            self?.endComputeAndSwapToRender()
        }
    }
    // Lazy init & reuse of SDF argument buffer (avoid per-dispatch creation cost)    
    internal func ensureSdfArgumentBuffer() -> (MTLBuffer, MTLArgumentEncoder) {
        if let sdfArgumentEncoder, let sdfArgumentBuffer{
            return (sdfArgumentBuffer, sdfArgumentEncoder)
        }else{
            let desc = MTLArgumentDescriptor()
            desc.dataType = .texture
            desc.textureType = .type3D
            desc.index = 0
            desc.access = .readOnly
            desc.arrayLength = CollisionManager.MAX_RIGIDS
            let enc = device.makeArgumentEncoder(arguments: [desc])!
            let buf = device.makeBuffer(length: enc.encodedLength, options: [])!
            enc.setArgumentBuffer(buf, offset: 0)
            sdfArgumentEncoder = enc
            sdfArgumentBuffer = buf
            return (buf, enc)
        }
    }

    // Integrate accumulated SDF impulses (from particles) and update collision transform
    internal func applySDFImpulseAggregationToCollisionTransform() {
        guard let accBuf = sdfImpulseAccumulatorBuffer,
              let collisionManager = collisionManager else { return }
        // Read accumulator (index 0)
        let accPtr = accBuf.contents().bindMemory(to: SDFImpulseAccumulator.self, capacity: CollisionManager.MAX_RIGIDS)
        let acc = accPtr[0]
        let J = SIMD3<Float>(acc.impulse_x, acc.impulse_y, acc.impulse_z)
        let Tau = SIMD3<Float>(acc.torque_x, acc.torque_y, acc.torque_z)
        if simd_length(J) < 1e-6 && simd_length(Tau) < 1e-6 { return }

        // Read SDF size to approximate inertia as a solid box
        let cuPtr = collisionManager.getCollisionUniformBuffer().contents().bindMemory(to: CollisionUniforms.self, capacity: 1)
        let size = cuPtr[0].sdfSize
        let hx = max(1e-4, size.x * 0.5)
        let hy = max(1e-4, size.y * 0.5)
        let hz = max(1e-4, size.z * 0.5)

        // Simple physical params
        let dt = timeStep
        let mass: Float = max(1e-4, cuPtr[0].sdfMass)
        let I = SIMD3<Float>(
            (mass / 12.0) * (hy*hy + hz*hz),
            (mass / 12.0) * (hx*hx + hz*hz),
            (mass / 12.0) * (hx*hx + hy*hy)
        )

        // Damped velocity integration
        let linearDamping: Float = 0.2
        let angularDamping: Float = 0.1
        sdfRigidLinearVelocity *= exp(-linearDamping * dt)
        sdfRigidAngularVelocity *= exp(-angularDamping * dt)
        sdfRigidLinearVelocity += J / mass
        sdfRigidAngularVelocity += SIMD3<Float>(
            Tau.x / max(I.x, 1e-6),
            Tau.y / max(I.y, 1e-6),
            Tau.z / max(I.z, 1e-6)
        )

        // Clamp velocities for stability
        let maxLin: Float = 5.0
        let maxAng: Float = 5.0
        sdfRigidLinearVelocity = simd_clamp(sdfRigidLinearVelocity, -SIMD3<Float>(repeating: maxLin), SIMD3<Float>(repeating: maxLin))
        sdfRigidAngularVelocity = simd_clamp(sdfRigidAngularVelocity, -SIMD3<Float>(repeating: maxAng), SIMD3<Float>(repeating: maxAng))

        // Build delta transform (translation scaled by dt; rotation around COM)
        let dTrans = float4x4(translation: sdfRigidLinearVelocity * dt)
        let omegaDt = sdfRigidAngularVelocity * dt
        let angle = simd_length(omegaDt)
        var dRot = matrix_identity_float4x4
        if angle > 1e-6 {
            let axis = simd_normalize(omegaDt)
            let c = cos(angle)
            let s = sin(angle)
            let t: Float = 1 - c
            let x = axis.x, y = axis.y, z = axis.z
            let r00 = t*x*x + c
            let r01 = t*x*y - s*z
            let r02 = t*x*z + s*y
            let r10 = t*x*y + s*z
            let r11 = t*y*y + c
            let r12 = t*y*z - s*x
            let r20 = t*x*z - s*y
            let r21 = t*y*z + s*x
            let r22 = t*z*z + c
            let rot3 = float3x3(SIMD3<Float>(r00, r01, r02),
                                 SIMD3<Float>(r10, r11, r12),
                                 SIMD3<Float>(r20, r21, r22))
            let c0 = SIMD4<Float>(rot3.columns.0.x, rot3.columns.0.y, rot3.columns.0.z, 0)
            let c1 = SIMD4<Float>(rot3.columns.1.x, rot3.columns.1.y, rot3.columns.1.z, 0)
            let c2 = SIMD4<Float>(rot3.columns.2.x, rot3.columns.2.y, rot3.columns.2.z, 0)
            let c3 = SIMD4<Float>(0, 0, 0, 1)
            dRot = float4x4(c0, c1, c2, c3)
        }

        // Compute world-space COM from local SDF center
        let comLocal = cuPtr[0].sdfOrigin + 0.5 * cuPtr[0].sdfSize
        let comWorld4 = cuPtr[0].collisionTransform * SIMD4<Float>(comLocal.x, comLocal.y, comLocal.z, 1.0)
        let comWorld = SIMD3<Float>(comWorld4.x, comWorld4.y, comWorld4.z)

        let Tcom = float4x4(translation: comWorld)
        let TcomInv = float4x4(translation: -comWorld)

        // Apply on top of current transform: translate, then rotate about COM
        var T = cuPtr[0].collisionTransform
        T = dTrans * (Tcom * dRot * TcomInv) * T
        cuPtr[0].collisionTransform = T
        cuPtr[0].collisionInvTransform = T.inverse

        // Optionally reset accumulator (we also clear it before next frame)
        accPtr[0] = SDFImpulseAccumulator(
            impulse_x: 0, impulse_y: 0, impulse_z: 0,
            torque_x: 0, torque_y: 0, torque_z: 0
        )
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

        // Clear SDF impulse accumulator once per frame before substeps
        if let accBuf = sdfImpulseAccumulatorBuffer, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: accBuf, range: 0..<accBuf.length, value: 0)
            blit.endEncoding()
        }

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
                        let (argumentBuffer, argumentEncoder) = ensureSdfArgumentBuffer()
                        
                        if let tex = collisionManager.getSDFTexture() {
                            argumentEncoder.setTexture(tex, index: 0)
                            computeEncoder.useResource(tex, usage: .read)
                        }
                        computeEncoder.setBuffer(argumentBuffer, offset: 0, index: 4)
                        // Bind SDF impulse accumulator for aggregation
                        if let accBuf = sdfImpulseAccumulatorBuffer {
                            computeEncoder.setBuffer(accBuf, offset: 0, index: 5)
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
                        let (argumentBuffer, argumentEncoder) = ensureSdfArgumentBuffer()
                        
                        if let tex = collisionManager.getSDFTexture() {
                            argumentEncoder.setTexture(tex, index: 0)
                            computeEncoder.useResource(tex, usage: .read)
                        }
                        computeEncoder.setBuffer(argumentBuffer, offset: 0, index: 4)
                        // Bind SDF impulse accumulator for aggregation
                        if let accBuf = sdfImpulseAccumulatorBuffer {
                            computeEncoder.setBuffer(accBuf, offset: 0, index: 5)
                        }
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
    
    // MARK: - Rigid Body Simulation
    
    internal func computeRigidBodySimulation(commandBuffer: MTLCommandBuffer) {
        guard materialParameters.currentMaterialMode == .rigidBody else { return }
        
        let particleThreadgroupSize = min(256, accumulateRigidBodyForcesPipelineState.maxTotalThreadsPerThreadgroup)
        let particleThreadgroups = MTLSize(
            width: (particleCount + particleThreadgroupSize - 1) / particleThreadgroupSize,
            height: 1,
            depth: 1
        )
        let particleThreadsPerThreadgroup = MTLSize(
            width: particleThreadgroupSize,
            height: 1,
            depth: 1
        )
        
        let rigidBodyCount = 1 // Currently supporting one rigid body
        
        // Stage 1: Accumulate forces and torques from particles to rigid bodies
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(accumulateRigidBodyForcesPipelineState)
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
        
        // Stage 2: Update rigid body dynamics
        if rigidBodyCount > 0 {
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(updateRigidBodyDynamicsPipelineState)
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
        
        // Stage 3: Project particles to maintain rigid body constraints
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(projectRigidBodyParticlesPipelineState)
            computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(rigidBodyStateBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(computeRigidInfoBuffer, offset: 0, index: 3)
            
            computeEncoder.dispatchThreadgroups(
                particleThreadgroups,
                threadsPerThreadgroup: particleThreadsPerThreadgroup
            )
            computeEncoder.endEncoding()
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
