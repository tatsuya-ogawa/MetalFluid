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
    var rigidId: UInt32  // Rigid body ID (0 = no rigid body, >0 = rigid body index)
    var initialOffset: SIMD3<Float>  // Initial relative position from rigid body center of mass
    //    var volume: Float             // Volume
    //    var Jp: Float  // Plastic deformation determinant
    //    var color: SIMD3<Float>  // Rendering color
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

struct CollisionUniforms {
    var sdfOrigin: SIMD3<Float>
    var sdfSize: SIMD3<Float>
    var sdfResolution: SIMD3<Int32>
    var collisionStiffness: Float
    var collisionDamping: Float
    var enableCollision: UInt32
    var collisionTransform: float4x4
    var collisionInvTransform: float4x4
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
        
        if performCompute {
            renderer.compute(commandBuffer: commandBuffer)
        }
        
        // Update matrices for rendering
        let screenSize = SIMD2<Float>(
            Float(renderPassDescriptor.colorAttachments[0].texture!.width),
            Float(renderPassDescriptor.colorAttachments[0].texture!.height)
        )
        
        // Only create depth buffer if the render pass descriptor doesn't have depth attachment
        // AND the pipeline expects depth (which it does since depthAttachmentPixelFormat = .depth32Float)
        if renderPassDescriptor.depthAttachment.texture == nil {
            let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float,
                width: Int(screenSize.x),
                height: Int(screenSize.y),
                mipmapped: false
            )
            depthTextureDescriptor.usage = .renderTarget
            let depthBuffer = renderer.device.makeTexture(descriptor: depthTextureDescriptor)!
            
            renderPassDescriptor.depthAttachment.texture = depthBuffer
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
            renderPassDescriptor.depthAttachment.storeAction = .store
        }
        
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
            renderEncoder.setVertexBuffer(renderer.particleBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(renderer.vertexUniformBuffer, offset: 0, index: 1)
            
            // For pressure heatmap mode, also bind the grid buffer
            if renderer.currentParticleRenderMode == .pressureHeatmap {
                renderEncoder.setVertexBuffer(renderer.gridBuffer, offset: 0, index: 2)
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
        
        if performCompute {
            renderer.compute(commandBuffer: commandBuffer)
        }
        
        // Get screen size from render pass descriptor
        guard let colorTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return
        }
        let screenSize = SIMD2<Float>(Float(colorTexture.width), Float(colorTexture.height))
        
        renderer.updateScreenSize(screenSize)
        
        // Step 1: Render depth map
        renderer.renderDepthMap(commandBuffer: commandBuffer)
        
        // Step 2: Apply bilateral filter to depth (4 iterations)
        for _ in 0..<4 {
            renderer.applyDepthFilter(
                commandBuffer: commandBuffer,
                depthThreshold: 0.01,
                filterRadius: 3
            )
        }
        
        // Step 3: Render thickness map
        renderer.renderThicknessMap(commandBuffer: commandBuffer)
        
        // Step 4: Apply Gaussian filter to thickness (1 iteration)
        renderer.applyThicknessFilter(commandBuffer: commandBuffer, filterRadius: 4)
        
        // Step 5: Render final fluid surface
        let fluidUniformPointer = renderer.fluidRenderUniformBuffer.contents().bindMemory(
            to: FluidRenderUniforms.self,
            capacity: 1
        )
        
        let texelSize = SIMD2<Float>(1.0 / screenSize.x, 1.0 / screenSize.y)
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
        
        // Ensure depth buffer is available for mesh rendering
        if renderPassDescriptor.depthAttachment.texture == nil {
            let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float,
                width: Int(screenSize.x),
                height: Int(screenSize.y),
                mipmapped: false
            )
            depthTextureDescriptor.usage = .renderTarget
            let depthBuffer = renderer.device.makeTexture(descriptor: depthTextureDescriptor)!
            
            renderPassDescriptor.depthAttachment.texture = depthBuffer
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
            renderPassDescriptor.depthAttachment.storeAction = .store
        }
        
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
        renderEncoder.setFragmentTexture(renderer.depthTexture, index: 0)
        renderEncoder.setFragmentTexture(renderer.filteredThicknessTexture, index: 1)
        renderEncoder.setFragmentTexture(renderer.environmentTexture, index: 2)
        renderEncoder.setFragmentBuffer(renderer.fluidRenderUniformBuffer, offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        renderEncoder.endEncoding()
        
        commandBuffer.commit()
    }
}

class MPMFluidRenderer: NSObject {
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
    
    // Rigid body material compute pipeline states
    public var particlesToGridRigidPipelineState: MTLComputePipelineState!
    public var gridToParticlesRigid1PipelineState: MTLComputePipelineState!
    public var gridToParticlesRigid2PipelineState: MTLComputePipelineState!
    public var gridToParticlesRigid3PipelineState: MTLComputePipelineState!
    public var gridToParticlesRigid4PipelineState: MTLComputePipelineState!
    
    // Rigid body state buffer
    public var rigidBodyStateBuffer: MTLBuffer!
    
    // Bitonic sort pipeline states
    internal var extractSortKeysPipelineState: MTLComputePipelineState!
    internal var bitonicSortPipelineState: MTLComputePipelineState!
    internal var reorderParticlesPipelineState: MTLComputePipelineState!
    internal var verifySortPipelineState: MTLComputePipelineState!
    
    // Radix sort pipeline states
    internal var extractSortKeysRadixPipelineState: MTLComputePipelineState!
    internal var radixSortPassPipelineState: MTLComputePipelineState!
    internal var radixSortLocalPipelineState: MTLComputePipelineState!
    internal var initializeRadixHistogramPipelineState: MTLComputePipelineState!
    internal var computePrefixSumsPipelineState: MTLComputePipelineState!
    internal var reorderParticlesRadixPipelineState: MTLComputePipelineState!
    internal var verifyRadixSortPipelineState: MTLComputePipelineState!
    
    // Buffers - compute
    internal var particleBuffer: MTLBuffer!
    internal var computeUniformBuffer: MTLBuffer!
    internal var vertexUniformBuffer: MTLBuffer!
    internal var gridBuffer: MTLBuffer!
    
    // Depth textures and buffers
    internal var depthTexture: MTLTexture!
    internal var tempDepthTexture: MTLTexture!
    internal var filteredDepthTexture: MTLTexture!
    internal var filterUniformBuffer: MTLBuffer!
    
    // Fluid surface rendering buffers and textures
    internal var fluidRenderUniformBuffer: MTLBuffer!
    internal var thicknessTexture: MTLTexture!
    internal var tempThicknessTexture: MTLTexture!
    internal var filteredThicknessTexture: MTLTexture!
    internal var gaussianUniformBuffer: MTLBuffer!
    internal var environmentTexture: MTLTexture!
    
    // Cube index buffer for instanced rendering
    internal var cubeIndexBuffer: MTLBuffer?
    
    // Screen size for depth filtering
    internal var screenSize: SIMD2<Float> = SIMD2<Float>(800, 600)
    
    // Bitonic sort buffers
    internal var sortKeysBuffer: MTLBuffer!
    internal var sortedParticleBuffer: MTLBuffer!
    
    // Radix sort buffers
    internal var radixSortKeysBuffer: MTLBuffer!
    internal var radixTempKeysBuffer: MTLBuffer!
    internal var radixHistogramBuffer: MTLBuffer!
    internal var radixSortedParticleBuffer: MTLBuffer!
    
    // Sort constants
    internal var maxThreadsPerGroup: Int = 256
    
    // Collision detection
    internal var collisionManager: CollisionManager?
    
    // Performance settings - Public for testing
    public var particleCount: Int
    public var gridSize: Int
    private let gridHeightMultiplier: Float
    public var gridNodes: Int { gridSize * Int(Float(gridSize) * gridHeightMultiplier) * gridSize }
    internal var frameIndex: Int = 0
    
    // Number of simulation substeps per frame
    public var simulationSubsteps: Int {
        get{
            switch currentMaterialMode {
            case .fluid:
                return 2
            case .neoHookeanElastic:
                return 1
            case .rigidBody:
                return 1  // Rigid body is stable with single substep
            }
        }
    }
    // Particle sorting configuration
    public var enableParticleSorting: Bool = false
    public var sortingFrequency: Int = 4  // Sort every N frames
    public var currentSortingAlgorithm: SortingAlgorithm = .radixSort
    
    // Render mode state
    public var currentRenderMode: RenderMode = .particles
    public var currentParticleRenderMode: ParticleRenderMode = .pressureHeatmap
    
    // Material mode state
    public var currentMaterialMode: MaterialMode = .rigidBody
    public var youngsModulus: Float = 2e7  // Young's modulus (Pa) - Increased significantly for gravity resistance
    public var poissonsRatio: Float = 0.15  // Poisson's ratio - Lower for stiffer response
    
    // Mode-specific renderers
    internal var particleRenderer: ParticleRenderer!
    internal var waterRenderer: WaterRenderer!
    
    // Particle size multiplier for smooth scaling
    public var particleSizeMultiplier: Float = 1.0
    
    // Mass scale multiplier for particle mass scaling
    public var massScale: Float = 1.0
    
    // MLS-MPM parameters - Public for testing
    public func getGridRes()->SIMD3<Int32>{
        return SIMD3<Int32>(
            Int32(gridSize),
            Int32(Float(gridSize) * gridHeightMultiplier),
            Int32(gridSize)
        )
    }
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
                return -1.0
            case .rigidBody:
                return -2.0  // Standard gravity for rigid body
            }
        }
    }
    public let gridSpacing: Float = 1.0
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
        super.init()
        setupMetal()
        setupCollisionManager()  // Initialize collision manager once
        setupParticles()
        frameIndex = 0
        setupModeRenderers()
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
        setupBitonicSortPipelines()
        setupRadixSortPipelines()
        setupRenderPipeline()
        setupDepthPipelines()
        setupFluidSurfacePipeline()
        setupThicknessPipelines()
        setupBuffers()
        setupBitonicSortBuffers()
        setupRadixSortBuffers()
        setupDepthTextures(screenSize: screenSize)
        setupFluidTextures(screenSize: screenSize)
    }
    
    private func setupCollisionManager() {
        guard let device = device else { return }
        // Initialize collision manager (only once)
        collisionManager = CollisionManager(device: device)
    }
    
    internal func setupBuffers() {
        // MPM Particle buffer
        let particleBufferSize =
        MemoryLayout<MPMParticle>.stride * particleCount
        particleBuffer = device.makeBuffer(
            length: particleBufferSize,
            options: .storageModeShared
        )!
        
        // Compute shader uniform buffer
        let computeUniformBufferSize = MemoryLayout<ComputeShaderUniforms>.stride
        computeUniformBuffer = device.makeBuffer(
            length: computeUniformBufferSize,
            options: .storageModeShared
        )!
        
        // Vertex shader uniform buffer
        let vertexUniformBufferSize = MemoryLayout<VertexShaderUniforms>.stride
        vertexUniformBuffer = device.makeBuffer(
            length: vertexUniformBufferSize,
            options: .storageModeShared
        )!
        
        // Grid buffer - Using correct struct size
        let gridBufferSize = MemoryLayout<MPMGridNode>.stride * gridNodes
        gridBuffer = device.makeBuffer(
            length: gridBufferSize,
            options: .storageModeShared
        )!
        
        // Debug grid buffer
        debugGridBuffer = device.makeBuffer(
            length: gridBufferSize,
            options: .storageModeShared
        )!
        
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
            rest_density: restDensity,
            particleSizeMultiplier: particleSizeMultiplier,
            sphere_size: 0.025 * particleSizeMultiplier  // Same calculation as WebGPU
        )
        
        frameIndex += 1
    }
    
    // For interaction (add force on tap)
    func addForce(at position: SIMD2<Float>, force: SIMD2<Float>) {
        let particlePointer = particleBuffer.contents().bindMemory(
            to: MPMParticle.self,
            capacity: particleCount
        )
        
        // Convert 2D screen position to 3D world position (assume z=0)
        let worldPos = SIMD3<Float>(position.x, position.y, 0.0)
        let force3D = SIMD3<Float>(force.x, force.y, 0.0)
        
        for i in 0..<particleCount {
            let distance = length(particlePointer[i].position - worldPos)
            if distance < 0.15 {
                let falloff = exp(-distance * 8.0)
                particlePointer[i].velocity += force3D * falloff * 0.5
            }
        }
    }
    
    // Reset simulation
    func reset() {
        setupParticles()
        frameIndex = 0
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
    public var debugGridBuffer: MTLBuffer!
    
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
        }
    }
    
    private func updateComputeUniforms() {
        let (boundaryMin, boundaryMax) = getBoundaryMinMax()
        let currentTime = CACurrentMediaTime()
        let timeSalt = UInt32(currentTime*1000)%UInt32.max
        let timeStep: Float = 0.1
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
            gravity: gravity,
            gridSpacing: gridSpacing,
            domainOrigin: domainOrigin,
            gridResolution: gridRes,
            gridNodeCount: nodeCount,
            boundaryMin: boundaryMin,
            boundaryMax: boundaryMax,
            stiffness: stiffness,
            rest_density: restDensity,
            dynamic_viscosity: dynamic_viscosity,
            massScale: massScale,
            timeSalt: timeSalt,
            materialMode: UInt32(currentMaterialMode == .fluid ? 0 : (currentMaterialMode == .neoHookeanElastic ? 1 : 2)),  // 0: fluid, 1: elastic, 2: rigid
            youngsModulus: youngsModulus,
            poissonsRatio: poissonsRatio,
            rigidBodyCount: currentMaterialMode == .rigidBody ? 1 : 0
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
        }
    }
}
