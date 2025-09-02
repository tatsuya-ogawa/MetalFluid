import Metal
import MetalKit
import simd
import UIKit

// MARK: - Texture Bundle Struct
internal struct FluidRenderTextures {
    let depthTexture: MTLTexture
    let tempDepthTexture: MTLTexture
    let filteredDepthTexture: MTLTexture
    let thicknessTexture: MTLTexture
    let tempThicknessTexture: MTLTexture
    let filteredThicknessTexture: MTLTexture
    let environmentTexture: MTLTexture
    let offscreenColorTexture: MTLTexture  // Offscreen render target
    let offscreenDepthTexture: MTLTexture  // Offscreen depth buffer
    let screenSize: SIMD2<Float>
    let bufferIndex: Int  // Buffer index for double buffering
}

// MARK: - Renderer Protocol
protocol ModeRenderer {
    func render(
        renderPassDescriptor: MTLRenderPassDescriptor,
        performCompute: Bool,
        projectionMatrix: float4x4,
        viewMatrix: float4x4
    )
}

// MARK: - Particle Renderer
class ParticleRenderer: ModeRenderer {
    internal weak var fluidRenderer: MPMFluidRenderer?
    
    init(fluidRenderer: MPMFluidRenderer) {
        self.fluidRenderer = fluidRenderer
    }
    
    func render(
        renderPassDescriptor: MTLRenderPassDescriptor,
        performCompute: Bool,
        projectionMatrix: float4x4,
        viewMatrix: float4x4
    ) {
        guard let renderer = fluidRenderer,
              let commandBuffer = renderer.commandQueue.makeCommandBuffer() else {
            return
        }
        
        guard let particleBuffer = self.fluidRenderer?.scene.getRenderParticleBuffer() else{
            return
        }
        
        if performCompute {
            renderer.compute(commandBuffer: commandBuffer)
        }
        
        // Update matrices for rendering
        let screenSize = SIMD2<Float>(
            Float(renderPassDescriptor.colorAttachments[0].texture!.width),
            Float(renderPassDescriptor.colorAttachments[0].texture!.height)
        )
        
        renderer.ensureDepthBuffer(for: renderPassDescriptor, screenSize: screenSize, label: "ParticleRenderDepthBuffer")
        // Render background (same command buffer)
        guard let backgroundRenderer = renderer.backgroundRenderer,let colorAttachmentTexture = renderPassDescriptor.colorAttachments[0].texture else{
                return
        }
        backgroundRenderer.renderBackground(commandBuffer: commandBuffer, targetTexture: colorAttachmentTexture)
        backgroundRenderer.updateCollisionSDFIfNeeded()
        renderPassDescriptor.colorAttachments[0].loadAction = .load
        
        // Render mesh and particles with new ordering
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(
            descriptor: renderPassDescriptor
        ) {
            // Step 1: Render collision mesh with depth writing enabled
            renderer.renderCollisionMeshInEncoder(renderEncoder: renderEncoder)
            
            // Step 2: Create a new depth stencil state for particles that tests depth but doesn't write
            let particleDepthStencilDescriptor = MTLDepthStencilDescriptor()
            particleDepthStencilDescriptor.depthCompareFunction = .less
            particleDepthStencilDescriptor.isDepthWriteEnabled = false  // Don't write depth for transparent particles
            let particleDepthStencilState = renderer.device.makeDepthStencilState(descriptor: particleDepthStencilDescriptor)!
            
            // Step 3: Render particles with depth testing but transparent blending
            guard let pipelineState = renderer.pressureHeatmapPipelineState else{
                fatalError("Unable to select pipeline state")
            }
            
            renderEncoder.setRenderPipelineState(pipelineState)
            renderEncoder.setDepthStencilState(particleDepthStencilState)
            renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(renderer.scene.getVertexUniformBuffer(), offset: 0, index: 1)
            
            // For pressure heatmap mode, also bind the render grid buffer
            if renderer.currentParticleRenderMode == .pressureHeatmap {
                renderEncoder.setVertexBuffer(renderer.scene.getRenderGridBuffer(), offset: 0, index: 2)
            }
            
            renderEncoder.drawPrimitives(
                type: .point,
                vertexStart: 0,
                vertexCount: renderer.particleCount
            )
            
            // Overlay AR mesh wireframe (if AR background is active)
            backgroundRenderer.renderOverlay(renderEncoder: renderEncoder, targetTexture: colorAttachmentTexture)
            renderEncoder.endEncoding()
        }
        
        commandBuffer.commit()
    }
}

// MARK: - Water Renderer
class WaterRenderer: ModeRenderer {
    internal weak var fluidRenderer: MPMFluidRenderer?
    
    init(fluidRenderer: MPMFluidRenderer) {
        self.fluidRenderer = fluidRenderer
    }
    
    func render(
        renderPassDescriptor: MTLRenderPassDescriptor,
        performCompute: Bool,
        projectionMatrix: float4x4,
        viewMatrix: float4x4
    ) {
        guard let renderer = fluidRenderer,
              let commandBuffer = renderer.commandQueue.makeCommandBuffer() else {
            return
        }
        guard let particleBuffer = self.fluidRenderer?.scene.getRenderParticleBuffer() else{
            return
        }
        if performCompute {
            renderer.compute(commandBuffer: commandBuffer)
        }
        
        // Get screen size from render pass descriptor
        guard let colorTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return
        }
        let screenSize = SIMD2<Float>(Float(colorTexture.width), Float(colorTexture.height))
        
        guard let (textures, release) = renderer.acquireTextures(for: screenSize) else {
            return
        }
        // Release textures after GPU has finished using them
        commandBuffer.addCompletedHandler { _ in release() }
        
        // Step 1: Render depth map
        renderer.renderDepthMap(commandBuffer: commandBuffer, particleBuffer: particleBuffer, textures: textures)
        
        // Step 2: Apply bilateral filter to depth (4 iterations)
        for _ in 0..<4 {
            renderer.applyDepthFilter(
                commandBuffer: commandBuffer,
                textures: textures,
                depthThreshold: 0.01,
                filterRadius: 3
            )
        }
        
        // Step 3: Render thickness map
        renderer.renderThicknessMap(commandBuffer: commandBuffer,particleBuffer: particleBuffer, textures: textures)
        
        // Step 4: Apply Gaussian filter to thickness (1 iteration)
        renderer.applyThicknessFilter(commandBuffer: commandBuffer, textures: textures, filterRadius: 4)
        
        // Step 5: Render final fluid surface
        let fluidUniformPointer = renderer.scene.getFluidRenderUniformBuffer().contents().bindMemory(
            to: FluidRenderUniforms.self,
            capacity: 1
        )
        
        let texelSize = SIMD2<Float>(1.0 / textures.screenSize.x, 1.0 / textures.screenSize.y)
        let invProjectionMatrix = projectionMatrix.inverse
        let invViewMatrix = viewMatrix.inverse
        
        fluidUniformPointer[0] = FluidRenderUniforms(
            texelSize: texelSize,
            sphereSize: 1.0,
            invProjectionMatrix: invProjectionMatrix,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            invViewMatrix: invViewMatrix
        )
        
        // Draw background into the same render target, then overlay fluid using framebuffer fetch in shader
        renderer.ensureDepthBuffer(for: renderPassDescriptor, screenSize: screenSize, label: "WaterRenderDepthBuffer")
        guard let backgroundRenderer = renderer.backgroundRenderer,let colorAttachmentTexture = renderPassDescriptor.colorAttachments[0].texture else{
                return
        }
        backgroundRenderer.renderBackground(commandBuffer: commandBuffer, targetTexture: colorAttachmentTexture)
        backgroundRenderer.updateCollisionSDFIfNeeded()
        renderPassDescriptor.colorAttachments[0].loadAction = .load

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        renderer.renderCollisionMeshInEncoder(renderEncoder: renderEncoder)
        
        let fluidDepthStencilDescriptor = MTLDepthStencilDescriptor()
        fluidDepthStencilDescriptor.depthCompareFunction = .less
        fluidDepthStencilDescriptor.isDepthWriteEnabled = true
        let fluidDepthStencilState = renderer.device.makeDepthStencilState(descriptor: fluidDepthStencilDescriptor)!
        renderEncoder.setRenderPipelineState(renderer.fluidSurfacePipelineState)
        renderEncoder.setDepthStencilState(fluidDepthStencilState)
        renderEncoder.setVertexBuffer(renderer.scene.getFluidRenderUniformBuffer(), offset: 0, index: 0)
        renderEncoder.setFragmentTexture(textures.depthTexture, index: 0)
        renderEncoder.setFragmentTexture(textures.filteredThicknessTexture, index: 1)
        renderEncoder.setFragmentTexture(textures.environmentTexture, index: 2)
        renderEncoder.setFragmentBuffer(renderer.scene.getFluidRenderUniformBuffer(), offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        // Overlay AR mesh wireframe (if AR background is active)
        if let colorAttachmentTexture = renderPassDescriptor.colorAttachments[0].texture,
           let backgroundRenderer = renderer.backgroundRenderer {
            backgroundRenderer.renderOverlay(renderEncoder: renderEncoder, targetTexture: colorAttachmentTexture)
        }
        renderEncoder.endEncoding()
        commandBuffer.commit()
    }
}

// MARK: - Material Parameters
class MaterialParameters {
    
    // Number of simulation substeps per frame
    public var simulationSubsteps: Int {
        get{
            switch currentMaterialMode {
            case .fluid:
                return 2
            case .neoHookeanElastic:
                return 2
            }
        }
    }
    // Material mode state
    public var currentMaterialMode: MaterialMode = .fluid
    public var youngsModulus: Float = 2e7  // Young's modulus (Pa) - Increased significantly for gravity resistance
    public var poissonsRatio: Float = 0.15  // Poisson's ratio - Lower for stiffer response
    public let particleMass: Float = 1
    public let restDensity: Float = 4.0
    let stiffness: Float = 3.0
    let dynamic_viscosity: Float = 0.1
    public var gravity: Float {
        get{
            switch currentMaterialMode{
            case .fluid:
                return  -2.5 //-9.81
            case .neoHookeanElastic:
                return -1.5
            }
        }
    }
    public var materialMode: UInt32{
        get{
            return (currentMaterialMode == .fluid ? 0 : (currentMaterialMode == .neoHookeanElastic ? 1 : 2))
        }
    }
}

// MARK: - MPM Fluid Renderer
class MPMFluidRenderer: NSObject {
    let timeStep: Float = 0.1
    let materialParameters = MaterialParameters()
    
    public var device: MTLDevice!
    public var commandQueue: MTLCommandQueue!
    public var scene: FluidScene!
    internal var pressureHeatmapPipelineState: MTLRenderPipelineState!
    
    // MARK: - Pipeline States
    internal var depthRenderPipelineState: MTLRenderPipelineState!
    internal var depthFilterPipelineState: MTLRenderPipelineState!
    internal var fluidSurfacePipelineState: MTLRenderPipelineState!
    internal var thicknessRenderPipelineState: MTLRenderPipelineState!
    internal var gaussianFilterPipelineState: MTLRenderPipelineState!
    internal var depthStencilState: MTLDepthStencilState!
    
    public var clearGridPipelineState: MTLComputePipelineState!
    public var particlesToGridFluid1PipelineState: MTLComputePipelineState!
    public var particlesToGridFluid2PipelineState: MTLComputePipelineState!
    public var updateGridVelocityPipelineState: MTLComputePipelineState!
    public var gridToParticlesFluid1PipelineState: MTLComputePipelineState!
    public var particlesToGridElasticPipelineState: MTLComputePipelineState!
    public var gridToParticlesElasticPipelineState: MTLComputePipelineState!
    public var applyForceToGridPipelineState: MTLComputePipelineState!
    public var applySdfImpulseToTransformPipelineState: MTLComputePipelineState!
    
    // MARK: - Force System
    struct QueuedForce {
        let position: SIMD3<Float>
        let vector: SIMD3<Float>
        let radius: Float
    }
    private var forceQueue: [QueuedForce] = []
    private let forceQueueLock = NSLock()
    
    // MARK: - System Components
    internal var sortManager: SortManager!
    internal var isComputing: Bool = false
    public var collisionManager: CollisionManager?
    
    // MARK: - Texture Ring Buffer
    internal let maxTextureBuffers: Int = 3
    internal var textureRing: [FluidRenderTextures] = []
    internal var textureRingInUse: [Bool] = []
    internal var textureRingIndex: Int = 0
    internal var textureRingSize: SIMD2<Int> = SIMD2<Int>(0, 0)
    internal var screenSize: SIMD2<Float> = SIMD2<Float>(800, 600)
    
    // MARK: - SDF System
    internal var sdfArgumentEncoder: MTLArgumentEncoder?
    internal var sdfArgumentBuffer: MTLBuffer?
    internal var sdfRigidLinearVelocity: SIMD3<Float> = .zero
    internal var sdfRigidAngularVelocity: SIMD3<Float> = .zero
    
    // MARK: - Simulation Parameters
    public var particleCount: Int
    public var gridSize: Int
    private let gridHeightMultiplier: Float
    public var gridNodes: Int { gridSize * Int(Float(gridSize) * gridHeightMultiplier) * gridSize }
    internal var frameIndex: Int = 0
    public var particleSizeMultiplier: Float = 1.0
    public var massScale: Float = 1.0
    
    // MARK: - Rendering
    public var currentRenderMode: RenderMode = .particles
    public var currentParticleRenderMode: ParticleRenderMode = .pressureHeatmap
    internal var particleRenderer: ParticleRenderer!
    internal var waterRenderer: WaterRenderer!
    public var backgroundRenderer: BackgroundRenderer?
    weak var viewController: UIViewController?
    
    // MARK: - Grid and Transform
    public var gridSpacing: Float{
        get{
            return 1 / Float(gridSize)
        }
    }
    public var worldTransform: float4x4 = matrix_identity_float4x4
    
    public func getGridRes()->SIMD3<Int32>{
        return SIMD3<Int32>(
            Int32(gridSize),
            Int32(Float(gridSize) * gridHeightMultiplier),
            Int32(gridSize)
        )
    }
    
    func getRenderScale() -> Float{
        return 1.0 / gridSpacing / Float(gridSize) * 2
    }
    
    internal var domainOrigin: SIMD3<Float> {
        let gridRes = getGridRes()
        let domainExtentX: Float = Float(gridRes.x) * gridSpacing
        let domainExtentY: Float = Float(gridRes.y) * gridSpacing
        let domainExtentZ: Float = Float(gridRes.z) * gridSpacing
        let originOffset:Float = 0.0
        return SIMD3<Float>(
            originOffset * domainExtentX,
            originOffset * domainExtentY,
            originOffset * domainExtentZ
        )
    }

    private func getDomainCenteringTransform() -> float4x4 {
        let gridRes = getGridRes()
        let domainExtentX: Float = Float(gridRes.x) * gridSpacing
        let domainExtentY: Float = Float(gridRes.y) * gridSpacing
        let domainExtentZ: Float = Float(gridRes.z) * gridSpacing
        let originOffsetX: Float = -0.5
        let originOffsetZ: Float = -0.5
        let originOffsetY: Float = -0.5 / gridHeightMultiplier
        let renderOrigin = SIMD3<Float>(
            originOffsetX * domainExtentX,
            originOffsetY * domainExtentY,
            originOffsetZ * domainExtentZ
        )
        let centerVec = renderOrigin - domainOrigin
        return float4x4(translation: centerVec)
    }
    public func getGridToLocalTransform() -> float4x4 {
        let scaleFactor: Float = self.getRenderScale()
        let scaleMatrix = float4x4(
            SIMD4<Float>(scaleFactor, 0, 0, 0),
            SIMD4<Float>(0, scaleFactor, 0, 0),
            SIMD4<Float>(0, 0, scaleFactor, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
        let centeringMatrix = self.getDomainCenteringTransform()
        return scaleMatrix * centeringMatrix
    }

    
    let pad: Float = 5.0
    public func getBoundaryMinMax()->(SIMD3<Float>,SIMD3<Float>) {
        let gridRes = getGridRes()
        let boundaryMin = domainOrigin + SIMD3<Float>(pad, pad, pad) * gridSpacing
        let boundaryMax = domainOrigin + SIMD3<Float>(
            Float(gridRes.x) - pad,
            Float(gridRes.y) - pad,
            Float(gridRes.z) - pad
        ) * gridSpacing
        return (boundaryMin, boundaryMax)
    }
    
    // MARK: - Initialization
    
    init(particleCount: Int, gridSize: Int, gridHeightMultiplier: Float) {
        self.particleCount = particleCount
        self.gridSize = gridSize
        self.gridHeightMultiplier = gridHeightMultiplier
        super.init()
        setupMetal()
        setupCollisionManager()  // Initialize collision manager once
        setupScene()  // Create scene after Metal setup
        setupParticles()
        frameIndex = 0
        setupModeRenderers()
    }
    
    deinit {
        print("🗑️ MPMFluidRenderer deinitialized")
    }
    
    internal func setupModeRenderers() {
        particleRenderer = ParticleRenderer(fluidRenderer: self)
        waterRenderer = WaterRenderer(fluidRenderer: self)
    }

    internal func setupMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        self.commandQueue = commandQueue
        
        setupComputePipelines()
        setupRenderPipeline()
        setupDepthPipelines()
        setupFluidSurfacePipeline()
        setupThicknessPipelines()
        setupSortManager()
    }
    
    private func setupCollisionManager() {
        guard let device = device else { return }
        // Initialize collision manager (only once)
        collisionManager = CollisionManager(device: device)
    }
    
    private func setupSortManager() {
        guard let device = device,
              let commandQueue = commandQueue else { return }
        sortManager = SortManager(device: device, commandQueue: commandQueue, particleCount: particleCount)
    }
    
    private func setupScene() {
        guard let device = device else {
            fatalError("Metal device not initialized")
        }
        
        scene = FluidScene(
            device: device,
            particleCount: particleCount,
            gridSize: gridSize,
            gridHeightMultiplier: gridHeightMultiplier
        )
    }
            
    func update(deltaTime: Float, screenSize: SIMD2<Float>, projectionMatrix: float4x4, viewMatrix: float4x4)
    {
        // Update compute shader uniforms using the helper method
        updateComputeUniforms()
        
        // Update vertex shader uniforms
        let vertexUniformPointer = scene.getVertexUniformBuffer().contents().bindMemory(
            to: VertexShaderUniforms.self,
            capacity: 1
        )
        let gridRes = getGridRes()
        vertexUniformPointer[0] = VertexShaderUniforms(
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            gridSpacing: gridSpacing,
            physicalDomainOrigin: domainOrigin,
            gridResolution: gridRes,
            rest_density: materialParameters.restDensity,
            particleSizeMultiplier: particleSizeMultiplier,
            sphere_size: 0.025 * particleSizeMultiplier  // Same calculation as WebGPU
        )
        
        frameIndex += 1
    }
    
    // Apply all queued forces to grid (called after P2G phase)
    internal func applyQueuedForcesToGrid(commandBuffer: MTLCommandBuffer) {
        forceQueueLock.lock()
        let currentForces = forceQueue
        forceQueue.removeAll()
        forceQueueLock.unlock()
        
        guard !currentForces.isEmpty,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        for force in currentForces {
            // Create temporary buffers for force parameters
            let forcePositionBuffer = device.makeBuffer(bytes: [force.position], length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
            let forceVectorBuffer = device.makeBuffer(bytes: [force.vector], length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
            let forceRadiusBuffer = device.makeBuffer(bytes: [force.radius], length: MemoryLayout<Float>.stride, options: .storageModeShared)!
            
            // Dispatch force application to grid
            computeEncoder.setComputePipelineState(applyForceToGridPipelineState)
            computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 0)
            computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
            computeEncoder.setBuffer(forcePositionBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(forceVectorBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(forceRadiusBuffer, offset: 0, index: 4)
            
            let gridThreadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)  // GRID_THREADGROUP_SIZE
            let gridThreadgroups = MTLSize(
                width: (gridNodes + gridThreadsPerThreadgroup.width - 1) / gridThreadsPerThreadgroup.width,
                height: 1,
                depth: 1
            )
            
            computeEncoder.dispatchThreadgroups(gridThreadgroups, threadsPerThreadgroup: gridThreadsPerThreadgroup)
        }
        
        computeEncoder.endEncoding()
    }
    
    // For interaction (add force on tap)
    func addForce(at position: SIMD2<Float>, force: SIMD2<Float>) {
        // Convert 2D screen position to 3D world position (assume z=0)  
        let worldPos = SIMD3<Float>(position.x, position.y, 0.0)
        let force3D = SIMD3<Float>(force.x, force.y, 0.0)
        let forceRadius: Float = 0.15
        
        // Add force to queue (thread-safe)
        forceQueueLock.lock()
        forceQueue.append(QueuedForce(position: worldPos, vector: force3D, radius: forceRadius))
        forceQueueLock.unlock()
    }
    
    // Reset simulation
    func reset() {
        setupParticles()
        frameIndex = 0
        
        // Reset SDF rigid body velocities to zero
        sdfRigidLinearVelocity = .zero
        sdfRigidAngularVelocity = .zero
        
        // Reset SDF collision transform to initial position
        collisionManager?.resetSDFTransforms()
        
        // Clear SDF physics accumulators and velocities
        if let phyBuf = scene.getSDFPhysicsBuffer() {
            let p = phyBuf.contents().bindMemory(to: SDFPhysicsStateSwift.self, capacity: CollisionManager.MAX_COLLISION_SDF)
            for i in 0..<CollisionManager.MAX_COLLISION_SDF {
                p[i].impulse_x = 0; p[i].impulse_y = 0; p[i].impulse_z = 0
                p[i].torque_x = 0; p[i].torque_y = 0; p[i].torque_z = 0
                p[i].linearVelocity = .zero
                p[i].angularVelocity = .zero
            }
        }
        
        print("🔄 Simulation reset - particles and SDF state restored")
    }
            
    // Helper method to render collision mesh within an existing render encoder
    internal func renderCollisionMeshInEncoder(renderEncoder: MTLRenderCommandEncoder) {
        guard let collisionManager = collisionManager,
              collisionManager.isMeshVisible() else {
            return
        }
        
        // Render collision mesh using the existing encoder
        collisionManager.renderMeshesInEncoder(
            renderEncoder: renderEncoder,
            vertexUniformBuffer: scene.getVertexUniformBuffer()
        )
    }
    
    // --- Add grid debug buffer and methods ---
//    public var debugGridBuffer: MTLBuffer!
    
    public func setParticleSizeMultiplier(_ multiplier: Float) {
        particleSizeMultiplier = multiplier
        print("🎛️ Particle size multiplier set to: \(multiplier)")
    }
    
    public func setMassScale(_ scale: Float) {
        massScale = scale
        print("⚖️ Mass scale set to: \(scale)")
    }
    
    public func setParticleCount(_ count: Int) {
        if count != particleCount {
            particleCount = count
            print("🔢 Particle count set to: \(particleCount)")
            // Update scene buffers with new particle count
            scene.updateParticleCount(count)
            // Reinitialize particles with new count
            setupParticles()
            sortManager?.updateParticleCount(count)
        }
    }
    
    private func updateComputeUniforms() {
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let currentTime = CACurrentMediaTime()
        let timeSalt = UInt32(currentTime*1000)%UInt32.max
        let nodeCount = UInt32(gridNodes)
        
        // Update compute shader uniforms
        let computeUniformPointer = scene.getComputeUniformBuffer().contents().bindMemory(
            to: ComputeShaderUniforms.self,
            capacity: 1
        )
        let gridRes = getGridRes()
        computeUniformPointer[0] = ComputeShaderUniforms(
            deltaTime: timeStep,
            particleCount: UInt32(particleCount),
            gravity: materialParameters.gravity,
            gridSpacing: gridSpacing,
            domainOrigin: domainOrigin,
            gridResolution: gridRes,
            gridNodeCount: nodeCount,
            boundaryMin: boundaryMin,
            boundaryMax: boundaryMax,
            stiffness: materialParameters.stiffness,
            rest_density: materialParameters.restDensity,
            dynamic_viscosity: materialParameters.dynamic_viscosity,
            massScale: massScale,
            timeSalt: timeSalt,
            materialMode: materialParameters.materialMode,
            youngsModulus: materialParameters.youngsModulus,
            poissonsRatio: materialParameters.poissonsRatio,
        )
    }
    
    public func setGridSize(_ size: Int) {
        if size != gridSize {
            gridSize = size
            print("📐 Grid size set to: \(gridSize)")
            // Update scene buffers with new grid size
            scene.updateGridSize(size, gridHeightMultiplier)
            // Reinitialize particles with new grid
            setupParticles()
            
            // Update compute uniforms immediately for new grid size
            updateComputeUniforms()
            
            // Update collision boundaries for new grid size
            let (boundaryMin, boundaryMax) = getBoundaryMinMax()
            collisionManager?.updateGridBoundaries(
                gridBoundaryMin: boundaryMin,
                gridBoundaryMax: boundaryMax
            )
            
            sortManager?.updateParticleCount(particleCount)
        }
    }
    
    // Set world transform (translation + rotation); physics uses translation part
    public func setWorldTransform(_ M: float4x4) {
        worldTransform = M
    }

    internal func ensureDepthBuffer(for renderPassDescriptor: MTLRenderPassDescriptor, screenSize: SIMD2<Float>, label: String) {
        guard renderPassDescriptor.depthAttachment.texture == nil else { return }
        
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: Int(screenSize.x),
            height: Int(screenSize.y),
            mipmapped: false
        )
        depthTextureDescriptor.usage = .renderTarget
        depthTextureDescriptor.storageMode = .private
        
        let depthBuffer = device.makeTexture(descriptor: depthTextureDescriptor)!
        depthBuffer.label = label
        
        renderPassDescriptor.depthAttachment.texture = depthBuffer
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.clearDepth = 1.0
        renderPassDescriptor.depthAttachment.storeAction = .store
    }

    // MARK: - Texture ring buffer management
    private func ensureTextureRing(for newSize: SIMD2<Float>) {
        let w = Int(newSize.x.rounded())
        let h = Int(newSize.y.rounded())
        if textureRing.isEmpty || textureRingSize.x != w || textureRingSize.y != h {
            textureRing.removeAll()
            textureRingInUse.removeAll()
            textureRingSize = SIMD2<Int>(w, h)
            for _ in 0..<maxTextureBuffers {
                let tex = createTexturesForSize(newSize)
                textureRing.append(tex)
                textureRingInUse.append(false)
            }
            textureRingIndex = 0
        }
    }

    /// Acquire a textures bundle from the ring. Returns textures and a release closure.
    public func acquireTextures(for newSize: SIMD2<Float>) -> (FluidRenderTextures, () -> Void)? {
        ensureTextureRing(for: newSize)
        // Find a free slot starting from textureRingIndex
        var chosenIndex: Int? = nil
        for i in 0..<maxTextureBuffers {
            let idx = (textureRingIndex + i) % maxTextureBuffers
            if textureRingInUse[idx] == false {
                chosenIndex = idx
                break
            }
        }
        if chosenIndex == nil {
            // No free slot; reuse the next index and warn
            let idx = textureRingIndex
//            print("⚠️ Texture ring saturated; reusing slot \(idx). Consider increasing ring size.")
            chosenIndex = idx
        }
        let idx = chosenIndex!
        textureRingInUse[idx] = true
        textureRingIndex = (idx + 1) % maxTextureBuffers
        let textures = textureRing[idx]
        // Release closure marks the slot as free
        let release: () -> Void = { [weak self] in
            self?.textureRingInUse[idx] = false
        }
        return (textures, release)
    }
}

extension MPMFluidRenderer{
    // MARK: - Public API
    
    public func setRenderMode(_ mode: RenderMode) {
        currentRenderMode = mode
        print("🎨 Render mode switched to: \(mode)")
    }
    
    public func toggleRenderMode() {
        currentRenderMode = (currentRenderMode == .particles) ? .water : .particles
    }
}
