import Foundation
import Metal
import MetalKit
import simd

class CollisionManager {
    private let device: MTLDevice
    
    // SDF generation and collision detection
    private var sdfGenerator: SDFGenerator
    private var sdfTexture: MTLTexture?
    private var collisionUniformBuffer: MTLBuffer
    
    // Mesh rendering
    private var meshRenderer: CollisionMeshRenderer
    
    // Current mesh data
    private var currentTriangles: [Triangle] = []
    
    init(device: MTLDevice) {
        self.device = device
        self.sdfGenerator = SDFGenerator(device: device)
        self.meshRenderer = CollisionMeshRenderer(device: device)
        
        // Create collision uniform buffer
        let collisionUniformSize = MemoryLayout<CollisionUniforms>.stride
        guard let buffer = device.makeBuffer(length: collisionUniformSize, options: .storageModeShared) else {
            fatalError("Failed to create collision uniform buffer")
        }
        self.collisionUniformBuffer = buffer
        
        // Initialize with default values
        initializeCollisionUniforms()
    }
    
    private func initializeCollisionUniforms() {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        
        collisionUniformPointer[0] = CollisionUniforms(
            sdfOrigin: SIMD3<Float>(0, 0, 0),
            sdfSize: SIMD3<Float>(1, 1, 1),
            sdfResolution: SIMD3<Int32>(64, 64, 64),
            collisionStiffness: 1.0,
            collisionDamping: 0.8,
            enableCollision: 0, // Disabled by default
            fillMode: 0 // Surface collision by default
        )
    }
    
    // MARK: - Mesh Loading
    
    func loadMesh(objURL: URL, resolution: SIMD3<Int32>, fillMode: Bool = false, gridBoundaryMin: SIMD3<Float>? = nil) {
        // Calculate offset position for bunny (align bottom with grid boundary)
        var offsetToBottom: SIMD3<Float>? = nil
        if let gridMin = gridBoundaryMin {
            // Position bunny at the bottom center of the grid
            offsetToBottom = SIMD3<Float>(0, gridMin.y, 0) // Center X and Z, bottom Y
        }
        
        let triangles = sdfGenerator.loadOBJ(from: objURL, offsetToBottom: offsetToBottom)
        
        if triangles.isEmpty {
            print("No triangles loaded from OBJ file")
            return
        }
        
        currentTriangles = triangles
        
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
        
        // Generate SDF texture with specified resolution using GPU
        let sdfResolution = resolution
        print("🚀 Starting GPU SDF generation...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        sdfTexture = sdfGenerator.generateSDF(
            triangles: triangles,
            resolution: sdfResolution,
            boundingBox: (min: minBounds, max: maxBounds)
        )
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let duration = endTime - startTime
        print("⚡ GPU SDF generation completed in \(String(format: "%.3f", duration))s")
        
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
            
            // Load mesh into renderer for visualization
            meshRenderer.loadMesh(triangles: triangles)
            
            print("Successfully loaded collision mesh with \(triangles.count) triangles")
            print("SDF resolution: \(resolution)x\(resolution)x\(resolution)")
            print("Fill mode: \(fillMode ? "Inside fill" : "Surface collision")")
            print("SDF bounds: \(minBounds) to \(maxBounds)")
        } else {
            print("Failed to generate SDF texture")
        }
    }
    
    // MARK: - Configuration
    
    func setEnabled(_ enabled: Bool) {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        collisionUniformPointer[0].enableCollision = enabled ? 1 : 0
    }
    
    func isEnabled() -> Bool {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return collisionUniformPointer[0].enableCollision == 1
    }
    
    func setFillMode(_ fillMode: Bool) {
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
    
    func getFillMode() -> Bool {
        guard sdfTexture != nil else { return false }
        
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return collisionUniformPointer[0].fillMode == 1
    }
    
    func getCurrentSDFResolution() -> Int32? {
        guard sdfTexture != nil else { return nil }
        
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return collisionUniformPointer[0].sdfResolution.x
    }
    
    func setCollisionStiffness(_ stiffness: Float) {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        collisionUniformPointer[0].collisionStiffness = stiffness
    }
    
    func setCollisionDamping(_ damping: Float) {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        collisionUniformPointer[0].collisionDamping = damping
    }
    
    // MARK: - Rendering Controls
    
    func setMeshVisible(_ visible: Bool) {
        meshRenderer.isVisible = visible
    }
    
    func isMeshVisible() -> Bool {
        return meshRenderer.isVisible
    }
    
    func setMeshColor(_ color: SIMD4<Float>) {
        meshRenderer.setColor(color)
    }
    
    func setMeshWireframe(_ wireframe: Bool) {
        meshRenderer.setWireframeMode(wireframe)
    }
    
    // MARK: - Internal Access (for FluidRenderer)
    
    internal func getSDFTexture() -> MTLTexture? {
        return sdfTexture
    }
    
    internal func getCollisionUniformBuffer() -> MTLBuffer {
        return collisionUniformBuffer
    }
    
    internal func renderMesh(renderPassDescriptor: MTLRenderPassDescriptor,
                           commandBuffer: MTLCommandBuffer,
                           vertexUniformBuffer: MTLBuffer) {
        meshRenderer.render(
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            vertexUniformBuffer: vertexUniformBuffer
        )
    }
}
