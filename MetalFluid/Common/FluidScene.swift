import Metal
import MetalKit
import simd
import UIKit

// MARK: - Fluid Scene Class
class FluidScene {
    // MARK: - Properties
    let device: MTLDevice
    var particleCount: Int
    var gridSize: Int
    var gridHeightMultiplier: Float

    
    var gridNodes: Int { gridSize * Int(Float(gridSize) * gridHeightMultiplier) * gridSize }
    
    // MARK: - Buffer Management
    // Compute buffers (private for GPU performance)
    internal var computeParticleBuffer: MTLBuffer!
    internal var computeGridBuffer: MTLBuffer!
    internal var computeRigidInfoBuffer: MTLBuffer!
    internal var computeUniformBuffer: MTLBuffer!
    
    // Rendering buffers (private for GPU performance)
    internal var renderParticleBuffer: MTLBuffer!
    internal var renderGridBuffer: MTLBuffer!
    
    // Shared buffers for CPU/GPU access
    internal var particleStagingBuffer: MTLBuffer!
    internal var vertexUniformBuffer: MTLBuffer!
    
    // Render uniform buffers
    internal var filterUniformBuffer: MTLBuffer!
    internal var fluidRenderUniformBuffer: MTLBuffer!
    internal var gaussianUniformBuffer: MTLBuffer!
    
    // SDF and physics buffers
    internal var sdfPhysicsBuffer: MTLBuffer!
    internal var sdfCollisionsArrayBuffer: MTLBuffer?
    
    // Optional buffers
    internal var cubeIndexBuffer: MTLBuffer?
    
    // MARK: - Initialization
    init(device: MTLDevice, particleCount: Int, gridSize: Int, gridHeightMultiplier: Float) {
        self.device = device
        self.particleCount = particleCount
        self.gridSize = gridSize
        self.gridHeightMultiplier = gridHeightMultiplier
        
        setupBuffers()
    }
    
    // MARK: - Buffer Setup
    private func setupBuffers() {
        let particleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        let gridBufferSize = MemoryLayout<MPMGridNode>.stride * gridNodes
        let computeUniformBufferSize = MemoryLayout<ComputeShaderUniforms>.stride
        let vertexUniformBufferSize = MemoryLayout<VertexShaderUniforms>.stride
        
        // Create staging buffers for initial data upload (shared for CPU access)
        guard let particleStaging = device.makeBuffer(
            length: particleBufferSize,
            options: .storageModeShared
        ) else {
            fatalError("Failed to create particle staging buffer")
        }
        particleStaging.label = "ParticleStagingBuffer"
        particleStagingBuffer = particleStaging
        
        // Create compute buffers (private for GPU performance)
        guard let computeParticles = device.makeBuffer(
            length: particleBufferSize,
            options: .storageModePrivate
        ) else {
            fatalError("Failed to create compute particle buffer")
        }
        computeParticles.label = "ComputeParticleBuffer"
        computeParticleBuffer = computeParticles
        
        guard let computeUniforms = device.makeBuffer(
            length: computeUniformBufferSize,
            options: .storageModeShared
        ) else {
            fatalError("Failed to create compute uniform buffer")
        }
        computeUniforms.label = "ComputeUniformBuffer"
        computeUniformBuffer = computeUniforms
        
        // Create rendering buffers (private for GPU performance)
        guard let renderParticles = device.makeBuffer(
            length: particleBufferSize,
            options: .storageModePrivate
        ) else {
            fatalError("Failed to create render particle buffer")
        }
        renderParticles.label = "RenderParticleBuffer"
        renderParticleBuffer = renderParticles
        
        // Grid double buffering - Using private storage for GPU performance  
        computeGridBuffer = device.makeBuffer(
            length: gridBufferSize,
            options: .storageModePrivate
        )!
        computeGridBuffer.label = "ComputeGridBuffer"
        
        renderGridBuffer = device.makeBuffer(
            length: gridBufferSize,
            options: .storageModePrivate
        )!
        renderGridBuffer.label = "RenderGridBuffer"
        
        // Vertex shader uniform buffer
        vertexUniformBuffer = device.makeBuffer(
            length: vertexUniformBufferSize,
            options: .storageModeShared
        )!
        vertexUniformBuffer.label = "VertexUniformBuffer"
        
        // Filter uniform buffer
        let filterUniformSize = MemoryLayout<FilterUniforms>.stride
        filterUniformBuffer = device.makeBuffer(
            length: filterUniformSize,
            options: .storageModeShared
        )!
        filterUniformBuffer.label = "FilterUniformBuffer"
        
        // Fluid render uniform buffer
        let fluidRenderUniformSize = MemoryLayout<FluidRenderUniforms>.stride
        fluidRenderUniformBuffer = device.makeBuffer(
            length: fluidRenderUniformSize,
            options: .storageModeShared
        )!
        fluidRenderUniformBuffer.label = "FluidRenderUniformBuffer"
        
        // Gaussian uniform buffer
        let gaussianUniformSize = MemoryLayout<GaussianUniforms>.stride
        gaussianUniformBuffer = device.makeBuffer(
            length: gaussianUniformSize,
            options: .storageModeShared
        )!
        gaussianUniformBuffer.label = "GaussianUniformBuffer"
        
        // SDF impulse accumulator buffer
        let physicsSize = MemoryLayout<SDFPhysicsStateSwift>.stride * CollisionManager.MAX_COLLISION_SDF
        sdfPhysicsBuffer = device.makeBuffer(length: physicsSize, options: .storageModeShared)
        sdfPhysicsBuffer?.label = "SDFPhysicsBuffer"        
    }
    
    // MARK: - Buffer Access Methods
    func getComputeParticleBuffer() -> MTLBuffer { computeParticleBuffer }
    func getRenderParticleBuffer() -> MTLBuffer { renderParticleBuffer }
    func getComputeGridBuffer() -> MTLBuffer { computeGridBuffer }
    func getRenderGridBuffer() -> MTLBuffer { renderGridBuffer }
    func getComputeRigidInfoBuffer() -> MTLBuffer { computeRigidInfoBuffer }
    func getComputeUniformBuffer() -> MTLBuffer { computeUniformBuffer }
    func getVertexUniformBuffer() -> MTLBuffer { vertexUniformBuffer }
    func getParticleStagingBuffer() -> MTLBuffer { particleStagingBuffer }
    func getFilterUniformBuffer() -> MTLBuffer { filterUniformBuffer }
    func getFluidRenderUniformBuffer() -> MTLBuffer { fluidRenderUniformBuffer }
    func getGaussianUniformBuffer() -> MTLBuffer { gaussianUniformBuffer }
    func getSDFPhysicsBuffer() -> MTLBuffer? { sdfPhysicsBuffer }
    func getSDFCollisionsArrayBuffer() -> MTLBuffer? { sdfCollisionsArrayBuffer }
    func getCubeIndexBuffer() -> MTLBuffer? { cubeIndexBuffer }
    
    // MARK: - Buffer Management Operations
    func swapComputeRenderBuffers() {
        swap(&computeParticleBuffer, &renderParticleBuffer)
        swap(&computeGridBuffer, &renderGridBuffer)
    }
    
    func copyFromStagingToCompute(commandBuffer: MTLCommandBuffer) {
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
        
        // Copy staging data to compute buffers
        blitEncoder.copy(
            from: particleStagingBuffer, 
            sourceOffset: 0,
            to: computeParticleBuffer, 
            destinationOffset: 0,
            size: MemoryLayout<MPMParticle>.stride * particleCount
        )
        
        blitEncoder.endEncoding()
    }
    
    func copyFromComputeToRender(commandBuffer: MTLCommandBuffer) {
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
        
        // Copy compute results to render buffers
        blitEncoder.copy(
            from: computeParticleBuffer,
            sourceOffset: 0,
            to: renderParticleBuffer,
            destinationOffset: 0,
            size: MemoryLayout<MPMParticle>.stride * particleCount
        )
        
        blitEncoder.copy(
            from: computeGridBuffer,
            sourceOffset: 0,
            to: renderGridBuffer,
            destinationOffset: 0,
            size: MemoryLayout<MPMGridNode>.stride * gridNodes
        )
        
        blitEncoder.endEncoding()
    }
    
    // MARK: - Dynamic Buffer Management
    func updateParticleCount(_ newCount: Int) {
        if newCount != particleCount {
            print("🔄 Updating particle count from \(particleCount) to \(newCount)")
            particleCount = newCount
            setupBuffers() // Recreate buffers with new size
        }
    }
    
    func updateGridSize(_ newGridSize: Int, _ newGridHeightMultiplier: Float) {
        if newGridSize != gridSize || newGridHeightMultiplier != gridHeightMultiplier {
            print("🔄 Updating grid size from \(gridSize) to \(newGridSize), height multiplier from \(gridHeightMultiplier) to \(newGridHeightMultiplier)")
            gridSize = newGridSize
            gridHeightMultiplier = newGridHeightMultiplier
            setupBuffers() // Recreate buffers with new size
        }
    }
    
    // MARK: - Grid Helper Methods
    func getGridRes() -> SIMD3<Int32> {
        return SIMD3<Int32>(
            Int32(gridSize),
            Int32(Float(gridSize) * gridHeightMultiplier),
            Int32(gridSize)
        )
    }
    
    func getGridNodes() -> Int {
        return gridNodes
    }
}
