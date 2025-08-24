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
}

struct CollisionUniforms {
    var sdfOrigin: SIMD3<Float>
    var sdfSize: SIMD3<Float>
    var sdfResolution: SIMD3<Int32>
    var collisionStiffness: Float
    var collisionDamping: Float
    var enableCollision: UInt32
    var fillMode: UInt32  // 0: surface collision, 1: fill inside
}

// Vertex shader specific uniforms
struct VertexShaderUniforms {
    var mvpMatrix: float4x4
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

enum RenderMode {
    case particles
    case water
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
            renderPassDescriptor.depthAttachment.storeAction = .dontCare
        }
        
        // Render particles
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(
            descriptor: renderPassDescriptor
        ) {
            // Select pipeline based on particle render mode
            guard let pipelineState = renderer.pressureHeatmapPipelineState else{
                fatalError("Unable to select pipeline state")
            }
            
            renderEncoder.setRenderPipelineState(pipelineState)
            renderEncoder.setDepthStencilState(renderer.depthStencilState)
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
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderEncoder.setRenderPipelineState(renderer.fluidSurfacePipelineState)
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
    public var particlesToGrid1PipelineState: MTLComputePipelineState!
    public var particlesToGrid2PipelineState: MTLComputePipelineState!
    public var updateGridVelocityPipelineState: MTLComputePipelineState!
    public var gridToParticlesPipelineState: MTLComputePipelineState!
    
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
    internal var sdfTexture: MTLTexture?
    internal var collisionUniformBuffer: MTLBuffer!
    internal var sdfGenerator: SDFGenerator!
    
    // Performance settings - Public for testing
    public var particleCount: Int
    public var gridSize: Int
    private let gridHeightMultiplier: Float
    public var gridNodes: Int { gridSize * Int(Float(gridSize) * gridHeightMultiplier) * gridSize }
    internal var frameIndex: Int = 0
    
    // Number of simulation substeps per frame
    public var simulationSubsteps: Int = 2
    
    // Particle sorting configuration
    public var enableParticleSorting: Bool = false
    public var sortingFrequency: Int = 4  // Sort every N frames
    public var currentSortingAlgorithm: SortingAlgorithm = .radixSort
    
    // Render mode state
    public var currentRenderMode: RenderMode = .particles
    public var currentParticleRenderMode: ParticleRenderMode = .pressureHeatmap
    
    // Mode-specific renderers
    internal var particleRenderer: ParticleRenderer!
    internal var waterRenderer: WaterRenderer!
    
    // Particle size multiplier for smooth scaling
    public var particleSizeMultiplier: Float = 1.0
    
    // Mass scale multiplier for particle mass scaling
    public var massScale: Float = 1.0
    
    // MLS-MPM parameters - Public for testing
    public let particleMass: Float = 1
    public let restDensity: Float = 4.0
    let stiffness: Float = 3.0
    let dynamic_viscosity: Float = 0.1
    public let gravity: Float = -2.5 //-9.81
    public let gridSpacing: Float = 1.0
    func getRenderScale(scale:Float) -> Float{
        return scale / gridSpacing / Float(gridSize) * 2
    }
    internal var domainOrigin: SIMD3<Float>{
        get{
            let domainExtentX: Float = Float(gridSize) * gridSpacing
            let domainExtentY: Float = Float(gridSize) * gridHeightMultiplier * gridSpacing
            let domainExtentZ: Float = Float(gridSize) * gridSpacing
            let originOffset:Float = 0.0
            return SIMD3<Float>(
                originOffset * domainExtentX,
                originOffset * domainExtentY,
                originOffset * domainExtentZ
            )
        }
    }
    public func getDomainOriginTranslation() -> SIMD3<Float> {
        let domainExtentX: Float = Float(gridSize) * gridSpacing
        let domainExtentY: Float = Float(gridSize) * gridHeightMultiplier * gridSpacing
        let domainExtentZ: Float = Float(gridSize) * gridSpacing
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
    internal func getBoundaryMinMax()->(SIMD3<Float>,SIMD3<Float>) {
        let boundaryMin = domainOrigin + SIMD3<Float>(pad, pad, pad) * gridSpacing
        let boundaryMax = domainOrigin + SIMD3<Float>(
            Float(gridSize) - pad,
            Float(gridSize) * gridHeightMultiplier - pad,
            Float(gridSize) - pad
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
        setupParticles()
        frameIndex = 0
        setupModeRenderers()
    }
    
    internal func setupModeRenderers() {
        particleRenderer = ParticleRenderer(fluidRenderer: self)
        waterRenderer = WaterRenderer(fluidRenderer: self)
    }
    
    // Load OBJ file and generate SDF for collision detection
    public func loadCollisionMesh(objURL: URL, resolution: Int32 = 64, fillMode: Bool = false) {
        let triangles = sdfGenerator.loadOBJ(from: objURL)
        
        if triangles.isEmpty {
            print("No triangles loaded from OBJ file")
            return
        }
        
        // Calculate bounding box
        var minBounds = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxBounds = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
        
        for triangle in triangles {
            minBounds = min(minBounds, triangle.v0)
            minBounds = min(minBounds, triangle.v1)
            minBounds = min(minBounds, triangle.v2)
            maxBounds = max(maxBounds, triangle.v0)
            maxBounds = max(maxBounds, triangle.v1)
            maxBounds = max(maxBounds, triangle.v2)
        }
        
        // Expand bounds slightly for safety
        let padding: Float = 2.0
        minBounds -= SIMD3<Float>(padding, padding, padding)
        maxBounds += SIMD3<Float>(padding, padding, padding)
        
        // Generate SDF texture with specified resolution
        let sdfResolution = SIMD3<Int32>(resolution, resolution, resolution)
        sdfTexture = sdfGenerator.generateSDF(
            triangles: triangles,
            resolution: sdfResolution,
            boundingBox: (min: minBounds, max: maxBounds)
        )
        
        if sdfTexture != nil {
            // Update collision uniforms
            let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
                to: CollisionUniforms.self,
                capacity: 1
            )
            
            collisionUniformPointer[0] = CollisionUniforms(
                sdfOrigin: minBounds,
                sdfSize: maxBounds - minBounds,
                sdfResolution: sdfResolution,
                collisionStiffness: 1.0,
                collisionDamping: 0.8,
                enableCollision: 1,
                fillMode: fillMode ? 1 : 0
            )
            
            print("Successfully loaded collision mesh with \(triangles.count) triangles")
            print("SDF resolution: \(resolution)x\(resolution)x\(resolution)")
            print("Fill mode: \(fillMode ? "Inside fill" : "Surface collision")")
            print("SDF bounds: \(minBounds) to \(maxBounds)")
        } else {
            print("Failed to generate SDF texture")
        }
    }
    
    // Update SDF resolution for existing collision mesh
    public func updateSDFResolution(_ newResolution: Int32) {
        guard sdfTexture != nil else {
            print("No collision mesh loaded")
            return
        }
        
        // Regenerate with new resolution (requires re-loading the mesh)
        print("SDF resolution updated to \(newResolution)x\(newResolution)x\(newResolution)")
        print("Note: Re-load collision mesh with loadCollisionMesh() to apply new resolution")
    }
    
    // Get current SDF resolution
    public func getCurrentSDFResolution() -> Int32? {
        guard sdfTexture != nil else { return nil }
        
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return collisionUniformPointer[0].sdfResolution.x
    }
    
    // Toggle fill mode (inside fill vs surface collision)
    public func setFillMode(_ fillMode: Bool) {
        guard sdfTexture != nil else {
            print("No collision mesh loaded")
            return
        }
        
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        collisionUniformPointer[0].fillMode = fillMode ? 1 : 0
        
        print("Fill mode set to: \(fillMode ? "Inside fill" : "Surface collision")")
    }
    
    // Get current fill mode
    public func getFillMode() -> Bool {
        guard sdfTexture != nil else { return false }
        
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return collisionUniformPointer[0].fillMode == 1
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
        
        // Collision uniform buffer
        let collisionUniformSize = MemoryLayout<CollisionUniforms>.stride
        collisionUniformBuffer = device.makeBuffer(
            length: collisionUniformSize,
            options: .storageModeShared
        )!
        
        // Initialize SDF generator
        sdfGenerator = SDFGenerator(device: device)
        
        // Gaussian uniform buffer
        let gaussianUniformSize = MemoryLayout<GaussianUniforms>.stride
        gaussianUniformBuffer = device.makeBuffer(
            length: gaussianUniformSize,
            options: .storageModeShared
        )!
    }
            
    func update(deltaTime: Float, screenSize: SIMD2<Float>, mvpMatrix: float4x4, projectionMatrix: float4x4, viewMatrix: float4x4)
    {
        let timeStep:Float = 0.1//min(deltaTime, 0.2)
        let gridRes = SIMD3<Int32>(
            Int32(gridSize),
            Int32(Float(gridSize) * gridHeightMultiplier),
            Int32(gridSize)
        )
        let nodeCount = UInt32(gridNodes)
        let (boundaryMin,boundaryMax) = getBoundaryMinMax()
        let currentTime = CACurrentMediaTime()
        let timeSalt = UInt32(currentTime*1000)%UInt32.max
        
        // Update compute shader uniforms
        let computeUniformPointer = computeUniformBuffer.contents().bindMemory(
            to: ComputeShaderUniforms.self,
            capacity: 1
        )
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
            timeSalt: timeSalt
        )
        
        // Update vertex shader uniforms
        let vertexUniformPointer = vertexUniformBuffer.contents().bindMemory(
            to: VertexShaderUniforms.self,
            capacity: 1
        )
        
        // Apply render offset to mvpMatrix with appropriate scaling
        let renderOffset = getDomainOriginTranslation()
        let renderScale: Float = getRenderScale(scale: 1.0) // Use existing render scale function
        let scaledOffset = renderOffset * renderScale / 400
        
        let translationMatrix = float4x4(
            [1, 0, 0, scaledOffset.x],
            [0, 1, 0, scaledOffset.y],
            [0, 0, 1, scaledOffset.z],
            [0, 0, 0, 1]
        )
        let adjustedMvpMatrix = mvpMatrix * translationMatrix
        
        vertexUniformPointer[0] = VertexShaderUniforms(
            mvpMatrix: adjustedMvpMatrix,
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
    
    public func setGridSize(_ size: Int) {
        if size != gridSize {
            gridSize = size
            print("📐 Grid size set to: \(gridSize)")
            // Reinitialize simulation with new grid size
            setupMetal()
            setupParticles()
            setupModeRenderers()
        }
    }
}
