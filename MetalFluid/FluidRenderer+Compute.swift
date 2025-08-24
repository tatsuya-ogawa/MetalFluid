import Metal
import MetalKit
import simd

// MARK: - Compute Extension
extension MPMFluidRenderer {
    
    // MARK: - Setup Functions
    
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

        // Particles to grid 1 pipeline
        guard
            let particlesToGrid1Function = library.makeFunction(
                name: "particlesToGrid1"
            )
        else {
            fatalError("Could not find function 'particlesToGrid1'")
        }
        do {
            particlesToGrid1PipelineState = try device.makeComputePipelineState(
                function: particlesToGrid1Function
            )
        } catch {
            fatalError(
                "Could not create particlesToGrid1 pipeline state: \(error)"
            )
        }
        // Particles to grid 2 pipeline
        guard
            let particlesToGrid2Function = library.makeFunction(
                name: "particlesToGrid2"
            )
        else {
            fatalError("Could not find function 'particlesToGrid2'")
        }
        do {
            particlesToGrid2PipelineState = try device.makeComputePipelineState(
                function: particlesToGrid2Function
            )
        } catch {
            fatalError(
                "Could not create particlesToGrid2 pipeline state: \(error)"
            )
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

        // Grid to particles pipeline
        guard
            let gridToParticlesFunction = library.makeFunction(
                name: "gridToParticles"
            )
        else {
            fatalError("Could not find function 'gridToParticles'")
        }
        do {
            gridToParticlesPipelineState = try device.makeComputePipelineState(
                function: gridToParticlesFunction
            )
        } catch {
            fatalError(
                "Could not create grid to particles pipeline state: \(error)"
            )
        }
    }
    
    internal func setupParticles() {
        let particlePointer = particleBuffer.contents().bindMemory(
            to: MPMParticle.self,
            capacity: particleCount
        )
        
        // Get center and range of boundaryMin/boundaryMax
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let center = (boundaryMin + boundaryMax) * 0.5
        let range = (boundaryMax - boundaryMin) * 0.5
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
            
            // Note: Particle color is now handled by shaders
            
            particlePointer[i] = MPMParticle(
                position: finalPos,
                velocity: SIMD3<Float>(0.0, 0.0, 0.0),  // Initial velocity is 0
                C: simd_float3x3(0.0),  // Affine momentum matrix initialization
                mass: particleMass,
                //                volume: particleMass / restDensity,
                //                Jp: 1.0,  // Plastic deformation initial value
                //                color: particleColor
            )
        }
    }
    
    // MARK: - Main Compute Function
    
    func compute(commandBuffer: MTLCommandBuffer) {
        // Sort particles periodically for better cache locality
        if enableParticleSorting && frameIndex % sortingFrequency == 0 {
            do {
                let startTime = CACurrentMediaTime()
                try sortParticlesByGridIndexSafe()
                let sortTime = CACurrentMediaTime() - startTime
                if frameIndex % (sortingFrequency * 10) == 0 {
                    print("🔄 Particle sort took: \(String(format: "%.2f", sortTime * 1000))ms")
                }
            } catch {
                print("⚠️ Sorting error: \(error), disabling particle sorting")
                disableSortingOnError()
            }
        }
        
        computeSimulation(commandBuffer: commandBuffer)
    }
    
    // MARK: - MPM Simulation Pipeline
    
    internal func computeSimulation(commandBuffer: MTLCommandBuffer) {
        // Decide threadgroup sizes based on each pipeline's reported limits
        let gridThreadgroupSize = min(
            clearGridPipelineState.maxTotalThreadsPerThreadgroup,
            256
        )
        let particleThreadgroupSize = min(
            particlesToGrid1PipelineState.maxTotalThreadsPerThreadgroup,
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
        let substeps = max(1, simulationSubsteps)

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

            // 2. Particle to Grid 1 (P2G1) - Transfer mass and momentum from particles to grid
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(
                    particlesToGrid1PipelineState
                )
                computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
                computeEncoder.dispatchThreadgroups(
                    particleThreadgroups,
                    threadsPerThreadgroup: particleThreadsPerThreadgroup
                )
                computeEncoder.endEncoding()
            }
            // 2.5. Particle to Grid 2 (P2G2) - Add volume and stress-based momentum from particles to grid
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(
                    particlesToGrid2PipelineState
                )
                computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(gridBuffer, offset: 0, index: 2)
                computeEncoder.dispatchThreadgroups(
                    particleThreadgroups,
                    threadsPerThreadgroup: particleThreadsPerThreadgroup
                )
                computeEncoder.endEncoding()
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

            // 4. Grid to Particles (G2P) - Transfer velocity and affine momentum from grid to particles
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(
                    gridToParticlesPipelineState
                )
                computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
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
        }
    }
    
    // MARK: - Particle Sorting Support
    
    internal func reorderParticles(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(reorderParticlesPipelineState)
        computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedParticleBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sortKeysBuffer, offset: 0, index: 2) // Use bitonic sorted keys
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

}
