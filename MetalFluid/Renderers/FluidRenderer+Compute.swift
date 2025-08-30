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
            setupElasticCube(particlePointer: stagingParticlePointer, center: center, range: range, rigidInfoPointer: stagingRigidInfoPointer)
            
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
            
    private func setupElasticCube(particlePointer: UnsafeMutablePointer<MPMParticle>, center: SIMD3<Float>, range: SIMD3<Float>, rigidInfoPointer: UnsafeMutablePointer<MPMParticleRigidInfo>?) {
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
            // Integrate GPU wall collision velocities + aggregated particle impulses into SDF transform
            self?.applySDFImpulseAggregationToCollisionTransform(useGPUVelocities: true)
            self?.endComputeAndSwapToRender()
        }
    }
    // Set up SDF argument buffer with collision resources
    private func setupSdfArgumentBufferForCompute(computeEncoder: MTLComputeCommandEncoder, collisionManager: CollisionManager) {
        let (argumentBuffer, argumentEncoder) = ensureSdfArgumentBuffer()
        for i in 0..<collisionManager.items.count {
            let tex = collisionManager.items[i].getSDFTexture()!
            argumentEncoder.setTexture(tex, index: i)
            computeEncoder.useResource(tex, usage: .read)
            // Set collision uniforms in argument buffer at index 1
            argumentEncoder.setBuffer(collisionManager.items[i].getCollisionUniformBuffer(), offset: 0, index: CollisionManager.MAX_COLLISION_SDF + i)
            // Set accumulator buffer at index 2*MAX + i (use shared accumulator 0 for now)
            if let accBuf = sdfImpulseAccumulatorBuffer {
                argumentEncoder.setBuffer(accBuf, offset: 0, index: CollisionManager.MAX_COLLISION_SDF * 2 + i)
            }
        }
        computeEncoder.setBuffer(argumentBuffer, offset: 0, index: 3)
    }
    
    // Lazy init & reuse of SDF argument buffer (avoid per-dispatch creation cost)    
    internal func ensureSdfArgumentBuffer() -> (MTLBuffer, MTLArgumentEncoder) {
        if let sdfArgumentEncoder, let sdfArgumentBuffer{
            return (sdfArgumentBuffer, sdfArgumentEncoder)
        }else{
            // SDF texture array occupies indices [0 .. MAX-1]
            let sdfTextureDesc = MTLArgumentDescriptor()
            sdfTextureDesc.dataType = .texture
            sdfTextureDesc.textureType = .type3D
            sdfTextureDesc.index = 0
            sdfTextureDesc.access = .readOnly
            sdfTextureDesc.arrayLength = CollisionManager.MAX_COLLISION_SDF

            // CollisionUniforms buffer array occupies indices [MAX .. 2*MAX-1]
            let collisionBufferDesc = MTLArgumentDescriptor()
            collisionBufferDesc.dataType = .pointer
            collisionBufferDesc.index = CollisionManager.MAX_COLLISION_SDF
            collisionBufferDesc.access = .readOnly
            collisionBufferDesc.arrayLength = CollisionManager.MAX_COLLISION_SDF
            // Accumulator array occupies indices [2*MAX .. 3*MAX-1]
            let accumBufferDesc = MTLArgumentDescriptor()
            accumBufferDesc.dataType = .pointer
            accumBufferDesc.index = CollisionManager.MAX_COLLISION_SDF * 2
            accumBufferDesc.access = .readOnly
            accumBufferDesc.arrayLength = CollisionManager.MAX_COLLISION_SDF

            let enc = device.makeArgumentEncoder(arguments: [sdfTextureDesc, collisionBufferDesc, accumBufferDesc])!
            let buf = device.makeBuffer(length: enc.encodedLength, options: [])!
            enc.setArgumentBuffer(buf, offset: 0)
            sdfArgumentEncoder = enc
            sdfArgumentBuffer = buf
            return (buf, enc)
        }
    }

    private func getCurrentComputeUniforms() -> ComputeShaderUniforms {
        let p = computeUniformBuffer.contents().bindMemory(to: ComputeShaderUniforms.self, capacity: 1)
        return p[0]
    }

    // Integrate accumulated SDF impulses (from particles) and update collision transform
    internal func applySDFImpulseAggregationToCollisionTransform(useGPUVelocities: Bool = false) {
        guard let accBuf = sdfImpulseAccumulatorBuffer,
              let collisionManager = collisionManager else { return }
        // Static mode from per-SDF settings - simplified for now
        // let primaryMoves = collisionManager.getSDFSettings(index: 0)?.moves ?? true
        let primaryMoves = true  // Assume dynamic for now
        if !primaryMoves {
            let accPtr = accBuf.contents().bindMemory(to: SDFImpulseAccumulator.self, capacity: CollisionManager.MAX_COLLISION_SDF)
            accPtr[0] = SDFImpulseAccumulator(impulse_x: 0, impulse_y: 0, impulse_z: 0, torque_x: 0, torque_y: 0, torque_z: 0)
            return
        }
        // Read accumulator (index 0)
        let accPtr = accBuf.contents().bindMemory(to: SDFImpulseAccumulator.self, capacity: CollisionManager.MAX_COLLISION_SDF)
        let acc = accPtr[0]
        let J = SIMD3<Float>(acc.impulse_x, acc.impulse_y, acc.impulse_z)
        let Tau = SIMD3<Float>(acc.torque_x, acc.torque_y, acc.torque_z)
        if simd_length(J) < 1e-6 && simd_length(Tau) < 1e-6 && !useGPUVelocities { return }

        // Read SDF size to approximate inertia as a solid box
        let cuPtr = collisionManager.bunnyItem.getCollisionUniformBuffer().contents().bindMemory(to: CollisionUniforms.self, capacity: 1)
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

        // Optionally fetch GPU-updated velocities from rigidBodyStateBuffer[0]
        if useGPUVelocities {
            let rbPtr = rigidBodyStateBuffer.contents().bindMemory(to: RigidBodyState.self, capacity: 2)
            let vGPU = rbPtr[0].linearVelocity
            let wGPU = rbPtr[0].angularVelocity
            sdfRigidLinearVelocity = vGPU
            sdfRigidAngularVelocity = wGPU
        }

        // Optional gravity for SDF rigid body from per-SDF settings - simplified for now
        // let primaryGravity = collisionManager.getSDFSettings(index: 0)?.useGravity ?? false
        let primaryGravity = true  // Assume gravity enabled for now
        if primaryGravity {
            sdfRigidLinearVelocity.y += materialParameters.gravity * dt
        }

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
        let TcomInv = float4x4(translation: SIMD3<Float>(-comWorld.x, -comWorld.y, -comWorld.z))

        // Apply on top of current transform: translate, then rotate about COM
        var T = cuPtr[0].collisionTransform
        T = dTrans * (Tcom * dRot * TcomInv) * T
        // --- World-boundary collision handling for SDF OBB ---
        // Compute world AABB of transformed SDF (OBB -> AABB using |M| * h)
        let h = 0.5 * cuPtr[0].sdfSize
        let centerLocal = cuPtr[0].sdfOrigin + h
        let centerWorld4 = T * SIMD4<Float>(centerLocal.x, centerLocal.y, centerLocal.z, 1)
        var centerWorld = SIMD3<Float>(centerWorld4.x, centerWorld4.y, centerWorld4.z)

        // Upper-left 3x3 of T (rotation-scale)
        let m = float3x3(
            SIMD3<Float>(T.columns.0.x, T.columns.0.y, T.columns.0.z),
            SIMD3<Float>(T.columns.1.x, T.columns.1.y, T.columns.1.z),
            SIMD3<Float>(T.columns.2.x, T.columns.2.y, T.columns.2.z)
        )
        let absM = float3x3(
            SIMD3<Float>(abs(m.columns.0.x), abs(m.columns.0.y), abs(m.columns.0.z)),
            SIMD3<Float>(abs(m.columns.1.x), abs(m.columns.1.y), abs(m.columns.1.z)),
            SIMD3<Float>(abs(m.columns.2.x), abs(m.columns.2.y), abs(m.columns.2.z))
        )
        let extents = absM * h
        var worldMin = centerWorld - extents
        var worldMax = centerWorld + extents

        // Read simulation boundaries
        let uPtr = computeUniformBuffer.contents().bindMemory(to: ComputeShaderUniforms.self, capacity: 1)
        let bmin = uPtr[0].boundaryMin
        let bmax = uPtr[0].boundaryMax
        let wallThickness: Float = uPtr[0].gridSpacing
        let innerMin = bmin + SIMD3<Float>(repeating: wallThickness)
        let innerMax = bmax - SIMD3<Float>(repeating: wallThickness)

        // Compute correction to keep inside [bmin, bmax]
        var correction = SIMD3<Float>(repeating: 0)
        var hitNormal = SIMD3<Float>(repeating: 0)
        // Thin-wall SDF style: allow motion within [innerMin, innerMax],
        // treat the bands [bmin, innerMin] and [innerMax, bmax] as walls of thickness 1 cell
        if worldMin.x < innerMin.x { correction.x += (innerMin.x - worldMin.x); hitNormal.x += 1 }
        if worldMax.x > innerMax.x { correction.x -= (worldMax.x - innerMax.x); hitNormal.x -= 1 }
        if worldMin.y < innerMin.y { correction.y += (innerMin.y - worldMin.y); hitNormal.y += 1 }
        if worldMax.y > innerMax.y { correction.y -= (worldMax.y - innerMax.y); hitNormal.y -= 1 }
        if worldMin.z < innerMin.z { correction.z += (innerMin.z - worldMin.z); hitNormal.z += 1 }
        if worldMax.z > innerMax.z { correction.z -= (worldMax.z - innerMax.z); hitNormal.z -= 1 }

        if any(correction .!= 0) {
            // Apply correction to translation
            T.columns.3.x += correction.x
            T.columns.3.y += correction.y
            T.columns.3.z += correction.z
            centerWorld += correction
            worldMin += correction
            worldMax += correction

            // Reflect linear velocity on hit axes with restitution
            let restitution: Float = 0.2
            if hitNormal.x != 0 { sdfRigidLinearVelocity.x = -sdfRigidLinearVelocity.x * restitution }
            if hitNormal.y != 0 { sdfRigidLinearVelocity.y = -sdfRigidLinearVelocity.y * restitution }
            if hitNormal.z != 0 { sdfRigidLinearVelocity.z = -sdfRigidLinearVelocity.z * restitution }

            // Damp angular velocity slightly on collision to reduce tunneling
            sdfRigidAngularVelocity *= 0.8
        }

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
                    computeEncoder.setBuffer(computeParticleBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 2)
                    
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
