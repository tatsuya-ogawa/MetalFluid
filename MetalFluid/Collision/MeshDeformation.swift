import Foundation
import Metal
import MetalKit
import simd

// Deformation parameters for GPU
struct DeformationUniforms {
    var forceCount: UInt32
    var vertexCount: UInt32
    var influenceRadius: Float
    var deformationStrength: Float
    var currentTime: Float
}

// GPU-friendly collision force structure (matching CollisionForce but with 16-byte alignment)
struct GPUCollisionForce {
    var position: SIMD3<Float>
    var force: SIMD3<Float>
    var intensity: Float
    var timestamp: Float
}

class MeshDeformation {
    private let device: MTLDevice
    
    // GPU buffers for deformation
    private var deformationUniformBuffer: MTLBuffer
    private var collisionForceBuffer: MTLBuffer
    private var maxForces: Int = 100
    
    // Deformation parameters
    public var influenceRadius: Float = 0.3
    public var deformationStrength: Float = 0.1
    
    // Compute pipeline for vertex deformation
    private var deformationComputePipeline: MTLComputePipelineState?
    
    init(device: MTLDevice) {
        self.device = device
        
        // Create uniform buffer
        let uniformSize = MemoryLayout<DeformationUniforms>.stride
        guard let uniformBuffer = device.makeBuffer(length: uniformSize, options: .storageModeShared) else {
            fatalError("Failed to create deformation uniform buffer")
        }
        self.deformationUniformBuffer = uniformBuffer
        
        // Create collision force buffer
        let forceBufferSize = MemoryLayout<GPUCollisionForce>.stride * maxForces
        guard let forceBuffer = device.makeBuffer(length: forceBufferSize, options: .storageModeShared) else {
            fatalError("Failed to create collision force buffer")
        }
        self.collisionForceBuffer = forceBuffer
        
        setupComputePipeline()
    }
    
    private func setupComputePipeline() {
        guard let library = device.makeDefaultLibrary(),
              let computeFunction = library.makeFunction(name: "deformMeshVertices") else {
            print("⚠️ Failed to create mesh deformation compute function")
            return
        }
        
        do {
            deformationComputePipeline = try device.makeComputePipelineState(function: computeFunction)
        } catch {
            print("⚠️ Failed to create deformation compute pipeline: \(error)")
        }
    }
    
    /// Apply deformation to mesh vertices using CPU calculation
    func applyDeformationCPU(originalVertices: [SIMD3<Float>], 
                            collisionForces: [CollisionForce]) -> [SIMD3<Float>] {
        guard !collisionForces.isEmpty else { return originalVertices }
        
        return originalVertices.map { vertex in
            var deformedVertex = vertex
            
            for force in collisionForces {
                let distance = length(vertex - force.position)
                
                // Skip if vertex is outside influence radius
                if distance > influenceRadius { continue }
                
                // Calculate Gaussian influence based on distance
                let influence = exp(-(distance * distance) / (2.0 * influenceRadius * influenceRadius))
                
                // Apply force-based deformation
                let forceDirection = normalize(force.force)
                let deformationAmount = force.intensity * influence * deformationStrength
                
                deformedVertex += forceDirection * deformationAmount
            }
            
            return deformedVertex
        }
    }
    
    /// Apply deformation using GPU compute shader
    func applyDeformationGPU(commandBuffer: MTLCommandBuffer,
                            originalVertexBuffer: MTLBuffer,
                            deformedVertexBuffer: MTLBuffer,
                            vertexCount: Int,
                            collisionForces: [CollisionForce]) {
        
        guard let computePipeline = deformationComputePipeline,
              !collisionForces.isEmpty else {
            // Copy original to deformed if no deformation needed
            let blitEncoder = commandBuffer.makeBlitCommandEncoder()
            let bufferSize = MemoryLayout<SIMD3<Float>>.stride * vertexCount
            blitEncoder?.copy(from: originalVertexBuffer, sourceOffset: 0,
                             to: deformedVertexBuffer, destinationOffset: 0,
                             size: bufferSize)
            blitEncoder?.endEncoding()
            return
        }
        
        // Update collision force buffer
        updateCollisionForceBuffer(collisionForces: collisionForces)
        
        // Update uniforms
        let uniformPointer = deformationUniformBuffer.contents().bindMemory(
            to: DeformationUniforms.self, capacity: 1
        )
        
        uniformPointer[0] = DeformationUniforms(
            forceCount: UInt32(min(collisionForces.count, maxForces)),
            vertexCount: UInt32(vertexCount),
            influenceRadius: influenceRadius,
            deformationStrength: deformationStrength,
            currentTime: Float(CACurrentMediaTime())
        )
        
        // Dispatch compute shader
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(originalVertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(deformedVertexBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(collisionForceBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(deformationUniformBuffer, offset: 0, index: 3)
        
        let threadsPerGroup = MTLSize(width: min(computePipeline.threadExecutionWidth, vertexCount), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (vertexCount + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
    }
    
    private func updateCollisionForceBuffer(collisionForces: [CollisionForce]) {
        let forcePointer = collisionForceBuffer.contents().bindMemory(
            to: GPUCollisionForce.self, capacity: maxForces
        )
        
        let forceCount = min(collisionForces.count, maxForces)
        
        for i in 0..<forceCount {
            let force = collisionForces[i]
            forcePointer[i] = GPUCollisionForce(
                position: force.position,
                force: force.force,
                intensity: force.intensity,
                timestamp: force.timestamp
            )
        }
        
        // Clear remaining slots if needed
        for i in forceCount..<maxForces {
            forcePointer[i] = GPUCollisionForce(
                position: SIMD3<Float>(0, 0, 0),
                force: SIMD3<Float>(0, 0, 0),
                intensity: 0,
                timestamp: 0
            )
        }
    }
    
    /// Set deformation parameters
    func setParameters(influenceRadius: Float, deformationStrength: Float) {
        self.influenceRadius = influenceRadius
        self.deformationStrength = deformationStrength
    }
}
