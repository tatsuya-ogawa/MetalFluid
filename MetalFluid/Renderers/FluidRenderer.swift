import Metal
import MetalKit
import simd

// Struct definitions matching Metal shared header file
// Struct definition for MemoryLayout verification (matching packing)
struct MPMParticle {
    var position: SIMD3<Float>  // Position
    var velocity: SIMD3<Float>  // Velocity
    var C: simd_float3x3  // Affine momentum matrix (3D)
    var mass: Float  // Mass
    var originalIndex: UInt32 // Original particle index used to fetch unsorted rigid info
    // Note: rigidId and initialOffset moved to separate MPMParticleRigidInfo
    //    var volume: Float             // Volume
    //    var Jp: Float  // Plastic deformation determinant
    //    var color: SIMD3<Float>  // Rendering color
}

// Separate compact rigid-related info per particle
struct MPMParticleRigidInfo {
    var rigidId: UInt32
    var initialOffset: SIMD3<Float>
}

// Match MPMTypes.h exactly to keep CPU/GPU strides aligned
struct MPMGridNode {
    var velocity_x: Float  // atomic<float> on GPU
    var velocity_y: Float  // atomic<float> on GPU
    var velocity_z: Float  // atomic<float> on GPU
    var mass: Float  // atomic<float> on GPU
}

// Compute shader specific uniforms
struct ComputeShaderUniforms {
    var deltaTime: Float
    var particleCount: UInt32
    var gravity: Float
    var gridSpacing: Float
    var domainOrigin: SIMD3<Float>
    var gridResolution: SIMD3<Int32>
    var gridNodeCount: UInt32
    var boundaryMin: SIMD3<Float>
    var boundaryMax: SIMD3<Float>
    var stiffness: Float
    var rest_density: Float
    var dynamic_viscosity: Float
    var massScale: Float
    var timeSalt: UInt32
    var materialMode: UInt32  // 0: fluid, 1: neo-hookean elastic, 2: rigid body
    var youngsModulus: Float  // Young's modulus for elastic material
    var poissonsRatio: Float  // Poisson's ratio for elastic material
    var rigidBodyCount: UInt32  // Number of active rigid bodies
}

// Vertex shader specific uniforms
struct VertexShaderUniforms {
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var gridSpacing: Float
    var physicalDomainOrigin: SIMD3<Float>   // For physics calculations
    var gridResolution: SIMD3<Int32>
    var rest_density: Float
    var particleSizeMultiplier: Float
    var sphere_size: Float
}


struct FilterUniforms {
    var direction: SIMD2<Float>
    var screenSize: SIMD2<Float>
    var depthThreshold: Float
    var filterRadius: Int32
    var projectedParticleConstant: Float
    var maxFilterSize: Float
}

struct FluidRenderUniforms {
    var texelSize: SIMD2<Float>
    var sphereSize: Float
    var invProjectionMatrix: float4x4
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var invViewMatrix: float4x4
}

struct GaussianUniforms {
    var direction: SIMD2<Float>
    var screenSize: SIMD2<Float>
    var filterRadius: Int32
}

struct SortKey {
    var key: UInt32      // Grid index as sort key
    var value: UInt32    // Original particle index
}

// MARK: - Texture Bundle Struct
internal struct FluidRenderTextures {
    let depthTexture: MTLTexture
    let tempDepthTexture: MTLTexture
    let filteredDepthTexture: MTLTexture
    let thicknessTexture: MTLTexture
    let tempThicknessTexture: MTLTexture
    let filteredThicknessTexture: MTLTexture
    let environmentTexture: MTLTexture
    let screenSize: SIMD2<Float>
    let bufferIndex: Int  // Buffer index for double buffering
}

// Rigid Body state for complete rigid body dynamics
struct RigidBodyState {
    var centerOfMass: SIMD3<Float>          // Center of mass position
    var linearVelocity: SIMD3<Float>        // Linear velocity
    var angularVelocity: SIMD3<Float>       // Angular velocity
    var orientation: SIMD4<Float>           // Orientation quaternion (x, y, z, w)
    var totalMass: Float                    // Total mass of rigid body
    var invInertiaTensor: simd_float3x3     // Inverse inertia tensor (world space)
    var accumulatedForce: SIMD3<Float>      // Accumulated force for this frame
    var accumulatedTorque: SIMD3<Float>     // Accumulated torque for this frame
    var particleCount: UInt32               // Number of particles in this rigid body
    var isActive: UInt32                    // 1 if active, 0 if inactive
    var linearDamping: Float                // Linear damping coefficient
    var angularDamping: Float               // Angular damping coefficient
    var restitution: Float                  // Coefficient of restitution
    var friction: Float                     // Friction coefficient
    var halfExtents: SIMD3<Float>             // Local half extents (AABB in rest pose)
    var boundingRadius: Float              // Bounding sphere radius
}

enum RenderMode {
    case particles
    case water
}

enum MaterialMode {
    case fluid
    case neoHookeanElastic
    case rigidBody
}

enum ParticleRenderMode {
    case pressureHeatmap
}

enum SortingAlgorithm {
    case none
    case bitonicSort
    case radixSort
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
        
        guard let particleBuffer = self.fluidRenderer?.renderParticleBuffer else{
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
            renderEncoder.setVertexBuffer(renderer.vertexUniformBuffer, offset: 0, index: 1)
            
            // For pressure heatmap mode, also bind the render grid buffer
            if renderer.currentParticleRenderMode == .pressureHeatmap {
                renderEncoder.setVertexBuffer(renderer.renderGridBuffer, offset: 0, index: 2)
            }
            
            renderEncoder.drawPrimitives(
                type: .point,
                vertexStart: 0,
                vertexCount: renderer.particleCount
            )
            
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
        guard let particleBuffer = self.fluidRenderer?.renderParticleBuffer else{
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
        
        guard let textures = renderer.getTexturesForScreenSize(screenSize) else {
            // Skip rendering if texture buffer limit exceeded
            return
        }
        
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
        let fluidUniformPointer = renderer.fluidRenderUniformBuffer.contents().bindMemory(
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
        
        renderer.ensureDepthBuffer(for: renderPassDescriptor, screenSize: screenSize, label: "WaterRenderDepthBuffer")
        
        // Set background color for water renderer
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.8, green: 0.8, blue: 0.8, alpha: 1.0)
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Step 1: Render collision mesh with depth writing enabled first
        renderer.renderCollisionMeshInEncoder(renderEncoder: renderEncoder)
        
        // Step 2: Create depth stencil state for fluid surface that tests depth but can write
        let fluidDepthStencilDescriptor = MTLDepthStencilDescriptor()
        fluidDepthStencilDescriptor.depthCompareFunction = .less
        fluidDepthStencilDescriptor.isDepthWriteEnabled = true  // Fluid can write depth since it outputs depth from fragment shader
        let fluidDepthStencilState = renderer.device.makeDepthStencilState(descriptor: fluidDepthStencilDescriptor)!
        
        // Step 3: Render fluid surface with depth testing and blending
        renderEncoder.setRenderPipelineState(renderer.fluidSurfacePipelineState)
        renderEncoder.setDepthStencilState(fluidDepthStencilState)
        renderEncoder.setVertexBuffer(renderer.fluidRenderUniformBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentTexture(textures.depthTexture, index: 0)
        renderEncoder.setFragmentTexture(textures.filteredThicknessTexture, index: 1)
        renderEncoder.setFragmentTexture(textures.environmentTexture, index: 2)
        renderEncoder.setFragmentBuffer(renderer.fluidRenderUniformBuffer, offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        renderEncoder.endEncoding()
        
        commandBuffer.commit()
    }
}

class MaterialParameters {
    
    // Number of simulation substeps per frame
    public var simulationSubsteps: Int {
        get{
            switch currentMaterialMode {
            case .fluid:
                return 2
            case .neoHookeanElastic:
                return 2
            case .rigidBody:
                return 1  // Rigid body is stable with single substep
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
            case .rigidBody:
                return -2.0  // Standard gravity for rigid body
            }
        }
    }
    public var materialMode: UInt32{
        get{
            return (currentMaterialMode == .fluid ? 0 : (currentMaterialMode == .neoHookeanElastic ? 1 : 2))  // 0: fluid, 1: elastic, 2: rigid
        }
    }
}

class MPMFluidRenderer: NSObject {
    let timeStep: Float = 0.1
    let materialParameters = MaterialParameters()
    // Public for testing
    public var device: MTLDevice!
    public var commandQueue: MTLCommandQueue!
    internal var pressureHeatmapPipelineState: MTLRenderPipelineState!
    
    // Depth rendering pipeline states
    internal var depthRenderPipelineState: MTLRenderPipelineState!
    internal var depthFilterPipelineState: MTLRenderPipelineState!
    
    // Fluid surface rendering pipeline states
    internal var fluidSurfacePipelineState: MTLRenderPipelineState!
    
    // Thickness rendering pipeline states
    internal var thicknessRenderPipelineState: MTLRenderPipelineState!
    internal var gaussianFilterPipelineState: MTLRenderPipelineState!
        
    // Depth stencil state for proper depth testing
    internal var depthStencilState: MTLDepthStencilState!
    
    // MLS-MPM Compute pipeline states - Public for testing
    public var clearGridPipelineState: MTLComputePipelineState!
    public var particlesToGridFluid1PipelineState: MTLComputePipelineState!
    public var particlesToGridFluid2PipelineState: MTLComputePipelineState!
    public var updateGridVelocityPipelineState: MTLComputePipelineState!
    public var gridToParticlesFluid1PipelineState: MTLComputePipelineState!
    
    // Elastic material compute pipeline states
    public var particlesToGridElasticPipelineState: MTLComputePipelineState!
    public var gridToParticlesElasticPipelineState: MTLComputePipelineState!
    
    // Rigid body material compute pipeline states (renamed for clarity)
    public var accumulateRigidBodyForcesPipelineState: MTLComputePipelineState!
    public var updateRigidBodyDynamicsPipelineState: MTLComputePipelineState!
    public var projectRigidBodyParticlesPipelineState: MTLComputePipelineState!
    public var solveRigidBodyCollisionsPipelineState: MTLComputePipelineState! // Collision solver between rigid bodies
    
    // SDF Collision (Projection-Based Dynamics) pipeline states
    public var processParticleSDFCollisionsPipelineState: MTLComputePipelineState!
    public var solveParticleConstraintsIterativePipelineState: MTLComputePipelineState!
    public var processRigidBodySDFCollisionsPipelineState: MTLComputePipelineState!
    public var solveRigidBodyConstraintsIterativePipelineState: MTLComputePipelineState!
    public var solveRigidBodyToRigidBodyCollisionsPipelineState: MTLComputePipelineState!
    
    // Force application pipeline state
    public var applyForceToGridPipelineState: MTLComputePipelineState!
    
    // Force queue system
    struct QueuedForce {
        let position: SIMD3<Float>
        let vector: SIMD3<Float>
        let radius: Float
    }
    private var forceQueue: [QueuedForce] = []
    private let forceQueueLock = NSLock()
    
    // Rigid body state buffer
    public var rigidBodyStateBuffer: MTLBuffer!
    
    // Separate rigid info buffers (compute)
    internal var computeRigidInfoBuffer: MTLBuffer!
    
    // Sort manager for particle sorting operations
    internal var sortManager: SortManager!
    
    // Fixed 2-stage pipeline: Compute -> Rendering
    
    internal var computeUniformBuffer: MTLBuffer!
    // Compute buffers (calculation only)
    internal var computeParticleBuffer: MTLBuffer!
    // Rendering buffers (display only)  
    internal var renderParticleBuffer: MTLBuffer!
    
    // Static buffers
    internal var vertexUniformBuffer: MTLBuffer!
    
    // Grid double buffering (like particles)
    internal var computeGridBuffer: MTLBuffer!  // For computation
    internal var renderGridBuffer: MTLBuffer!   // For rendering
    
    // Staging buffers for initial data upload (shared memory for CPU access)
    internal var particleStagingBuffer: MTLBuffer!
    internal var rigidInfoStagingBuffer: MTLBuffer!
        
    // Compute/Render state tracking
    internal var isComputing: Bool = false
    
    // G-Buffer double buffering support
    internal var currentTextureBufferIndex: Int = 0
    internal let maxTextureBuffers: Int = 2
    
    // buffers
    internal var filterUniformBuffer: MTLBuffer!
    internal var fluidRenderUniformBuffer: MTLBuffer!
    internal var gaussianUniformBuffer: MTLBuffer!
    // Cube index buffer for instanced rendering
    internal var cubeIndexBuffer: MTLBuffer?
    // Screen size for depth filtering
    internal var screenSize: SIMD2<Float> = SIMD2<Float>(800, 600)
    // Collision detection
    public var collisionManager: CollisionManager?

    // Cached argument buffer for SDF textures (avoid per-dispatch creation)
    internal var sdfArgumentEncoder: MTLArgumentEncoder?
    internal var sdfArgumentBuffer: MTLBuffer?

    // AR SDF collision
    public var useARCollision: Bool = false
    private var arSDFTexture: MTLTexture?
    private var arSDFBoundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)?

    // --- SDF rigid-body impulse aggregation & state ---
    struct SDFImpulseAccumulator {
        var impulse_x: Float
        var impulse_y: Float
        var impulse_z: Float
        var torque_x: Float
        var torque_y: Float
        var torque_z: Float
    }
    internal var sdfImpulseAccumulatorBuffer: MTLBuffer?
    internal var sdfRigidLinearVelocity: SIMD3<Float> = .zero
    internal var sdfRigidAngularVelocity: SIMD3<Float> = .zero
    
    // Texture cache for screen size optimization
    internal let textureCacheManager: TextureCacheManager<FluidRenderTextures>
    
    // Performance settings - Public for testing
    public var particleCount: Int
    public var gridSize: Int
    private let gridHeightMultiplier: Float
    public var gridNodes: Int { gridSize * Int(Float(gridSize) * gridHeightMultiplier) * gridSize }
    internal var frameIndex: Int = 0
    
    // Render mode state
    public var currentRenderMode: RenderMode = .particles
    public var currentParticleRenderMode: ParticleRenderMode = .pressureHeatmap
    
    // AR state
    public var isAREnabled: Bool = false
    public var showARMesh: Bool = true
    public var arMeshWireframe: Bool = true
    
    // Mode-specific renderers
    internal var particleRenderer: ParticleRenderer!
    internal var waterRenderer: WaterRenderer!
    internal var arRenderer: ARRenderer?
    
    // Particle size multiplier for smooth scaling
    public var particleSizeMultiplier: Float = 1.0
    // Mass scale multiplier for particle mass scaling
    public var massScale: Float = 1.0

    public let gridSpacing: Float = 1.0
    // MLS-MPM parameters - Public for testing
    public func getGridRes()->SIMD3<Int32>{
        return SIMD3<Int32>(
            Int32(gridSize),
            Int32(Float(gridSize) * gridHeightMultiplier),
            Int32(gridSize)
        )
    }
    func getRenderScale(scale:Float) -> Float{
        return scale / gridSpacing / Float(gridSize) * 2
    }
    internal var domainOrigin: SIMD3<Float>{
        get{
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
    }
    public func getDomainOriginTranslation() -> SIMD3<Float> {
        let gridRes = getGridRes()
        let domainExtentX: Float = Float(gridRes.x) * gridSpacing
        let domainExtentY: Float = Float(gridRes.y) * gridSpacing
        let domainExtentZ: Float = Float(gridRes.z) * gridSpacing
        let originOffsetX: Float = -0.5
        let originOffsetZ: Float = -0.5
        // Auto-adjust Y offset based on height multiplier to keep fluid centered
        // Higher multiplier needs more negative offset to center properly
        let originOffsetY: Float = -0.5 + (2.0 - gridHeightMultiplier) * 0.2
        let renderOrigin = SIMD3<Float>(
            originOffsetX * domainExtentX,
            originOffsetY * domainExtentY,
            originOffsetZ * domainExtentZ
        )
        return domainOrigin - renderOrigin
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
    
    // Uniform data for MLS-MPM - Structs defined by typealias above
    
    init(particleCount: Int, gridSize: Int, gridHeightMultiplier: Float) {
        self.particleCount = particleCount
        self.gridSize = gridSize
        self.gridHeightMultiplier = gridHeightMultiplier
        self.textureCacheManager = TextureCacheManager<FluidRenderTextures>(
            maxSize: 3,
            name: "FluidRenderTextures"
        )
        super.init()
        setupMetal()
        setupCollisionManager()  // Initialize collision manager once
        setupParticles()
        frameIndex = 0
        setupModeRenderers()
    }
    
    deinit {
        // Clean up texture cache on deallocation
        textureCacheManager.clearCache()
        print("🗑️ MPMFluidRenderer deinitialized, texture cache cleared")
    }
    
    internal func setupModeRenderers() {
        particleRenderer = ParticleRenderer(fluidRenderer: self)
        waterRenderer = WaterRenderer(fluidRenderer: self)
        arRenderer = ARRenderer(device: device, commandQueue: commandQueue)
        
        print("🚀 ARRenderer initialized. AR supported: \(arRenderer?.isARSupported ?? false)")
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
        setupBuffers()
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
    
    internal func setupBuffers() {
        // Setup fixed 2-stage pipeline buffers
        let particleBufferSize = MemoryLayout<MPMParticle>.stride * particleCount
        let computeUniformBufferSize = MemoryLayout<ComputeShaderUniforms>.stride
        
        // Create staging buffers for initial data upload (shared for CPU access)
        guard let particleStaging = device.makeBuffer(
            length: particleBufferSize,
            options: .storageModeShared
        ) else {
            fatalError("Failed to create particle staging buffer")
        }
        particleStaging.label = "ParticleStagingBuffer"
        particleStagingBuffer = particleStaging
        
        let rigidInfoBufferSize = MemoryLayout<MPMParticleRigidInfo>.stride * particleCount
        guard let rigidInfoStaging = device.makeBuffer(
            length: rigidInfoBufferSize,
            options: .storageModeShared
        ) else {
            fatalError("Failed to create rigid info staging buffer")
        }
        rigidInfoStaging.label = "RigidInfoStagingBuffer"
        rigidInfoStagingBuffer = rigidInfoStaging

        // Create compute buffers (private for GPU performance)
        guard let computeParticles = device.makeBuffer(
            length: particleBufferSize,
            options: .storageModePrivate
        ) else {
            fatalError("Failed to create compute particle buffer")
        }
        computeParticles.label = "ComputeParticleBuffer"
        computeParticleBuffer = computeParticles

        // Create compute rigid info buffer (private)
        guard let computeRigid = device.makeBuffer(
            length: rigidInfoBufferSize,
            options: .storageModePrivate
        ) else {
            fatalError("Failed to create compute rigid info buffer")
        }
        computeRigid.label = "ComputeRigidInfoBuffer"
        computeRigidInfoBuffer = computeRigid
        
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
                
        // Vertex shader uniform buffer
        let vertexUniformBufferSize = MemoryLayout<VertexShaderUniforms>.stride
        vertexUniformBuffer = device.makeBuffer(
            length: vertexUniformBufferSize,
            options: .storageModeShared
        )!
        
        // Grid double buffering - Using private storage for GPU performance  
        let gridBufferSize = MemoryLayout<MPMGridNode>.stride * gridNodes
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
        
        // Debug grid buffer
//        debugGridBuffer = device.makeBuffer(
//            length: gridBufferSize,
//            options: .storageModeShared
//        )!
        
        // Filter uniform buffer
        let filterUniformSize = MemoryLayout<FilterUniforms>.stride
        filterUniformBuffer = device.makeBuffer(
            length: filterUniformSize,
            options: .storageModeShared
        )!
        
        // Fluid render uniform buffer
        let fluidRenderUniformSize = MemoryLayout<FluidRenderUniforms>.stride
        fluidRenderUniformBuffer = device.makeBuffer(
            length: fluidRenderUniformSize,
            options: .storageModeShared
        )!
        
        
        // Gaussian uniform buffer
        let gaussianUniformSize = MemoryLayout<GaussianUniforms>.stride
        gaussianUniformBuffer = device.makeBuffer(
            length: gaussianUniformSize,
            options: .storageModeShared
        )!
        
        // Rigid body state buffer
        let maxRigidBodies = 10  // Maximum number of rigid bodies
        let rigidBodyBufferSize = MemoryLayout<RigidBodyState>.stride * maxRigidBodies
        rigidBodyStateBuffer = device.makeBuffer(
            length: rigidBodyBufferSize,
            options: .storageModeShared
        )!

        // SDF impulse accumulator buffer (one per SDF, currently use index 0)
        let accumulatorSize = MemoryLayout<SDFImpulseAccumulator>.stride * CollisionManager.MAX_RIGIDS
        sdfImpulseAccumulatorBuffer = device.makeBuffer(length: accumulatorSize, options: .storageModeShared)
        sdfImpulseAccumulatorBuffer?.label = "SDFImpulseAccumulatorBuffer"
    }
            
    func update(deltaTime: Float, screenSize: SIMD2<Float>, projectionMatrix: float4x4, viewMatrix: float4x4)
    {
        // Update compute shader uniforms using the helper method
        updateComputeUniforms()
        
        // Update vertex shader uniforms
        let vertexUniformPointer = vertexUniformBuffer.contents().bindMemory(
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
            computeEncoder.setBuffer(computeGridBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(computeUniformBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(forcePositionBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(forceVectorBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(forceRadiusBuffer, offset: 0, index: 4)
            
            let gridThreadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
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
        collisionManager?.resetSDFTransform()
        
        // Clear SDF impulse accumulator
        if let accBuf = sdfImpulseAccumulatorBuffer {
            let accPtr = accBuf.contents().bindMemory(to: SDFImpulseAccumulator.self, capacity: CollisionManager.MAX_RIGIDS)
            for i in 0..<CollisionManager.MAX_RIGIDS {
                accPtr[i] = SDFImpulseAccumulator(
                    impulse_x: 0, impulse_y: 0, impulse_z: 0,
                    torque_x: 0, torque_y: 0, torque_z: 0
                )
            }
        }
        
        print("🔄 Simulation reset - particles and SDF state restored")
    }
    
    // MARK: - AR SDF Integration
    
    func updateARSDF(from arRenderer: ARRenderer?) {
        guard let arRenderer = arRenderer else {
            useARCollision = false
            arSDFTexture = nil
            arSDFBoundingBox = nil
            return
        }
        
        #if canImport(ARKit)
        if #available(iOS 13.4, macOS 10.15.4, *) {
            guard arRenderer.hasARMeshData() else {
                useARCollision = false
                arSDFTexture = nil
                arSDFBoundingBox = nil
                return
            }
            
            // Generate or get existing AR SDF
            let sdfTexture = arRenderer.generateSDFFromCurrentMeshes(resolution: SIMD3<Int32>(64, 64, 64))
            
            if let sdf = sdfTexture {
                arSDFTexture = sdf
                arSDFBoundingBox = arRenderer.getARSDFBoundingBox()
                useARCollision = true
                print("✅ AR SDF updated for collision detection")
            } else {
                useARCollision = false
                arSDFTexture = nil
                arSDFBoundingBox = nil
                print("❌ Failed to update AR SDF")
            }
        }
        #endif
    }
    
    func enableARCollision(_ enable: Bool) {
        guard enable else {
            useARCollision = false
            return
        }
        
        // Only enable if we have valid AR SDF data
        useARCollision = (arSDFTexture != nil)
        if useARCollision {
            print("🔄 AR collision enabled")
        } else {
            print("⚠️ Cannot enable AR collision - no SDF data available")
        }
    }
    
    func getARCollisionInfo() -> (isEnabled: Bool, hasData: Bool) {
        return (useARCollision, arSDFTexture != nil)
    }
    
    // Helper method to render collision mesh within an existing render encoder
    internal func renderCollisionMeshInEncoder(renderEncoder: MTLRenderCommandEncoder) {
        guard let collisionManager = collisionManager,
              collisionManager.isMeshVisible() else {
            return
        }
        
        // Render collision mesh using the existing encoder
        collisionManager.renderMeshInEncoder(
            renderEncoder: renderEncoder,
            vertexUniformBuffer: vertexUniformBuffer
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
            // Reinitialize simulation with new particle count
            setupMetal()
            setupParticles()
            setupModeRenderers()
            sortManager?.updateParticleCount(count)
        }
    }
    
    private func updateComputeUniforms() {
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let currentTime = CACurrentMediaTime()
        let timeSalt = UInt32(currentTime*1000)%UInt32.max
        let nodeCount = UInt32(gridNodes)
        
        // Update compute shader uniforms
        let computeUniformPointer = computeUniformBuffer.contents().bindMemory(
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
            rigidBodyCount: materialParameters.currentMaterialMode == .rigidBody ? 1 : 0
        )
    }
    
    public func setGridSize(_ size: Int) {
        if size != gridSize {
            gridSize = size
            print("📐 Grid size set to: \(gridSize)")
            // Reinitialize simulation with new grid size
            setupMetal()
            setupParticles()
            setupModeRenderers()
            
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
}
