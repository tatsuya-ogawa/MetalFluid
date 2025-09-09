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

// MARK: - Legacy renderer classes - moved to IntegratedRenderer

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
    
    // AR mode detection removed - handled by IntegratedRenderer
    
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
    public var elasticCubeScale: Float = 0.8
    
    // MARK: - Rendering
    public var currentRenderMode: RenderMode = .particles
    public var currentParticleRenderMode: ParticleRenderMode = .pressureHeatmap
    public var backgroundRenderer: BackgroundRenderer?
    weak var viewController: UIViewController?
    
    
    // MARK: - Grid and Transform
    private var gridSpacingMultiplier: Float = 1.0  // Multiplier for base grid spacing
    
    public var gridSpacing: Float{
        get{
            return (1.0 / Float(gridSize)) * gridSpacingMultiplier
        }
    }
    
    public func setGridSpacingMultiplier(_ multiplier: Float) {
        gridSpacingMultiplier = multiplier
    }
    
    public func getGridSpacingMultiplier() -> Float {
        return gridSpacingMultiplier
    }
    
    public func getGridRes()->SIMD3<Int32>{
        return SIMD3<Int32>(
            Int32(gridSize),
            Int32(Float(gridSize) * gridHeightMultiplier),
            Int32(gridSize)
        )
    }
    
    func getRenderScale() -> Float{
        return 1.0
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

    // Get domain center for world transform positioning
    public func getDomainBasePosition() -> SIMD3<Float> {
        let gridRes = getGridRes()
        let domainExtentX: Float = Float(gridRes.x) * gridSpacing
        let domainExtentY: Float = Float(gridRes.y) * gridSpacing / gridHeightMultiplier
        let domainExtentZ: Float = Float(gridRes.z) * gridSpacing
        
        // Center of the domain (domain origin is 0,0,0 so center is extent/2)
        return SIMD3<Float>(
            domainExtentX * 0.5,
            domainExtentY * 0.1,
            domainExtentZ * 0.5
        )
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
        setupCollisionManager()
        setupScene()
        setupParticles()
        frameIndex = 0
    }
    
    deinit {
        print("🗑️ MPMFluidRenderer deinitialized")
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
    func updateVertexUniforms(deltaTime: Float, screenSize: SIMD2<Float>, projectionMatrix: float4x4, viewMatrix: float4x4, worldTransform: float4x4) {
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
        )
    }
        
    // Apply all queued forces to grid (called after P2G phase)
    internal func applyQueuedForcesToGrid(commandBuffer: MTLCommandBuffer) {
        forceQueueLock.lock()
        let currentForces = forceQueue
        forceQueue.removeAll()
        forceQueueLock.unlock()
        
        guard !currentForces.isEmpty,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        // Prepare collision detection data
        let enabledItems = collisionManager?.items.filter({ $0.isEnabled() }) ?? []
        
        for force in currentForces {
            // Convert world coordinates to grid coordinates
            // Get world transform from compute uniforms
            let computeUniformsPointer = scene.getComputeUniformBuffer().contents().bindMemory(to: ComputeShaderUniforms.self, capacity: 1)
            let worldTransform = computeUniformsPointer[0].worldTransform
            let invWorldTransform = worldTransform.inverse
            let forcePositionFluid = invWorldTransform * SIMD4<Float>(force.position, 1.0)
            let forcePositionGrid = (SIMD3<Float>(forcePositionFluid.x,forcePositionFluid.y,forcePositionFluid.z) - domainOrigin) / gridSpacing
            let forceRadiusGrid = force.radius / gridSpacing
            
            let forcePositionBuffer = device.makeBuffer(bytes: [forcePositionGrid], length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
            let forceVectorBuffer = device.makeBuffer(bytes: [force.vector], length: MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
            let forceRadiusBuffer = device.makeBuffer(bytes: [forceRadiusGrid], length: MemoryLayout<Float>.stride, options: .storageModeShared)!
            
            // Apply force with collision detection
            computeEncoder.setComputePipelineState(applyForceToGridPipelineState)
            computeEncoder.setBuffer(scene.getComputeGridBuffer(), offset: 0, index: 0)
            computeEncoder.setBuffer(scene.getComputeUniformBuffer(), offset: 0, index: 1)
            computeEncoder.setBuffer(forcePositionBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(forceVectorBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(forceRadiusBuffer, offset: 0, index: 4)
            
            // Set up SDF argument buffer (textures and uniforms) at index 5
            if let collisionManager = collisionManager {
                setupSdfArgumentBufferForCompute(computeEncoder: computeEncoder, collisionManager: collisionManager)
            }
            
            // Set SDF count at index 6
            let sdfCount = UInt32(min(enabledItems.count, CollisionManager.MAX_COLLISION_SDF))
            let sdfCountBuffer = device.makeBuffer(bytes: [sdfCount], length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            computeEncoder.setBuffer(sdfCountBuffer, offset: 0, index: 6)
            
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
    internal func renderCollisionMeshInEncoder(renderEncoder: MTLRenderCommandEncoder, 
                                              arProjectionMatrix: float4x4? = nil, 
                                              arViewMatrix: float4x4? = nil) {
        guard let collisionManager = collisionManager else {
            return
        }
        
        // Render collision meshes if visible
        if collisionManager.isMeshVisible() {
            // Use AR matrices if provided, otherwise use fluid scene matrices
            if let projMatrix = arProjectionMatrix, 
               let viewMatrix = arViewMatrix {
                // Use camera world matrices for AR mode
                collisionManager.renderMeshesInEncoderForAR(
                    renderEncoder: renderEncoder,
                    projectionMatrix: projMatrix,
                    viewMatrix: viewMatrix
                )
            } else {
                // Use standard fluid scene matrices
                collisionManager.renderMeshesInEncoder(
                    renderEncoder: renderEncoder,
                    vertexUniformBuffer: scene.getVertexUniformBuffer()
                )
            }
        }
        
    }
    
    // MARK: - AR Mode Support removed - handled by IntegratedRenderer
    public func setParticleSizeMultiplier(_ multiplier: Float) {
        particleSizeMultiplier = multiplier
    }
    
    public func setMassScale(_ scale: Float) {
        massScale = scale
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
    
    public func updateComputeUniforms(worldTransform: float4x4) {
        frameIndex += 1
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
            worldTransform: worldTransform
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
            // FIXME
            updateComputeUniforms(worldTransform: matrix_identity_float4x4)
            
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
//    public func setWorldTransform(_ M: float4x4) {
//        worldTransform = M
//    }

    internal func ensureDepthBuffer(for renderPassDescriptor: MTLRenderPassDescriptor, screenSize: SIMD2<Float>, label: String) -> MTLTexture {
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
        return depthBuffer
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
