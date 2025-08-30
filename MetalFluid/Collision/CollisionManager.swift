import Foundation
import Metal
import MetalKit
import simd
struct Triangle {
    var v0: SIMD3<Float>
    var v1: SIMD3<Float>
    var v2: SIMD3<Float>
}
class CollisionManager {
    static let MAX_RIGIDS = 8
    let padding: Float = 1.0
    private let device: MTLDevice
    
    // SDF generation and collision detection
    private var sdfGenerator: SDFGenerator
    private var sdfTexture: MTLTexture?
    private var collisionUniformBuffer: MTLBuffer
    
    // Mesh rendering
    private var meshRenderer: CollisionMeshRenderer
    
    // Current mesh data
    private var currentTriangles: [Triangle] = []
    
    // Scale and offset control
    public var meshScale: Float = 1.0 {
        didSet {
            updateCollisionTransform()
        }
    }
    
    public var meshYOffset: Float = 0.0 {
        didSet {
            updateCollisionTransform()
        }
    }
    
    public var meshYRotation: Float = 0.0 {
        didSet {
            updateCollisionTransform()
        }
    }
    
    // Store current transform parameters for updates
    private var currentMeshMin: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    private var currentMeshMax: SIMD3<Float> = SIMD3<Float>(1, 1, 1)
    private var currentGridMin: SIMD3<Float>?
    private var currentGridMax: SIMD3<Float>?
    
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
            collisionStiffness: 1.0,  // Not used in velocity-based approach
            collisionDamping: 0.8,    // Not used in velocity-based approach  
            enableCollision: 0, // Disabled by default
            collisionTransform: matrix_identity_float4x4, // Default identity transform
            collisionInvTransform: matrix_identity_float4x4 // Default identity inverse transform
        )
    }
    
    // MARK: - Private Helper Methods
    
    private func calculateCollisionTransform(meshMin: SIMD3<Float>, meshMax: SIMD3<Float>, 
                                           gridMin: SIMD3<Float>?, gridMax: SIMD3<Float>?, 
                                           scale: SIMD3<Float> = SIMD3<Float>(1, 1, 1),
                                           rotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0),
                                           offset: SIMD3<Float> = SIMD3<Float>(0, 0, 0)) -> (float4x4, float4x4) {
        guard let gridMin = gridMin, let gridMax = gridMax else {
            return (matrix_identity_float4x4, matrix_identity_float4x4)
        }
        
        let gridCenter = (gridMin + gridMax) * 0.5
        let meshCenter = (meshMin + meshMax) * 0.5
        
        // Calculate translation to center mesh in grid (XZ) and align bottom (Y), plus offset
        // For Y alignment, we need to account for scaling effect on mesh height
        let scaledMeshHeight = (meshMax.y - meshMin.y) * scale.y
        let scaledMeshBottom = meshMin.y * scale.y + meshCenter.y * (scale.y - 1.0)
        
        let translation = SIMD3<Float>(
            gridCenter.x - meshCenter.x + offset.x,  // Center X + offset
            gridMin.y - scaledMeshBottom + offset.y, // Align scaled bottom Y + offset
            gridCenter.z - meshCenter.z + offset.z   // Center Z + offset
        )
        
        // Create transform matrix: T * R * S (applied in reverse order)
        let scaleMatrix = float4x4(scaling: scale)
        let rotationMatrix = float4x4(rotationX: rotation.x) * 
                            float4x4(rotationY: rotation.y) * 
                            float4x4(rotationZ: rotation.z)
        let translationMatrix = float4x4(translation: translation)
        
        let transform = translationMatrix * rotationMatrix * scaleMatrix
        let invTransform = transform.inverse
        
        return (transform, invTransform)
    }
    /// Calculate bounding box for given triangles
    func calculateBoundingBox(triangles: [Triangle]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard !triangles.isEmpty else {
            return (min: SIMD3<Float>(0, 0, 0), max: SIMD3<Float>(0, 0, 0))
        }
        
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
        
        return (min: minBounds, max: maxBounds)
    }
    private func updateCollisionTransform() {
        guard currentGridMin != nil && currentGridMax != nil else { return }
        
        let scaleVec = SIMD3<Float>(meshScale, meshScale, meshScale)
        let offsetVec = SIMD3<Float>(0.0, meshYOffset, 0.0)
        let rotationVec = SIMD3<Float>(0.0, meshYRotation * Float.pi / 180.0, 0.0) // Convert degrees to radians
        let (transform, invTransform) = calculateCollisionTransform(
            meshMin: currentMeshMin,
            meshMax: currentMeshMax,
            gridMin: currentGridMin,
            gridMax: currentGridMax,
            scale: scaleVec,
            rotation: rotationVec,
            offset: offsetVec
        )
        
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        
        collisionUniformPointer[0].collisionTransform = transform
        collisionUniformPointer[0].collisionInvTransform = invTransform
    }
    
    public func processAndGenerateSDF(triangles: [Triangle], resolution: SIMD3<Int32>, gridBoundaryMin: SIMD3<Float>?, gridBoundaryMax: SIMD3<Float>?, scale: SIMD3<Float> = SIMD3<Float>(1, 1, 1), offset: SIMD3<Float> = SIMD3<Float>(0, 0, 0), rotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0)) {
        currentTriangles = triangles
        
        // Store current parameters for scale updates
        let boundingBox = calculateBoundingBox(triangles: triangles)
        currentMeshMin = boundingBox.min
        currentMeshMax = boundingBox.max
        currentGridMin = gridBoundaryMin
        currentGridMax = gridBoundaryMax
        
        // Expand bounds slightly for safety
        let minBounds = boundingBox.min - SIMD3<Float>(padding, padding, padding)
        let maxBounds = boundingBox.max + SIMD3<Float>(padding, padding, padding)
        
        // Generate SDF texture with specified resolution using GPU
        print("🚀 Starting GPU SDF generation...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        sdfTexture = sdfGenerator.generateSDF(
            triangles: triangles,
            resolution: resolution,
            boundingBox: (min: minBounds, max: maxBounds)
        )
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let duration = endTime - startTime
        print("⚡ GPU SDF generation completed in \(String(format: "%.3f", duration))s")
        
        if sdfTexture != nil {
            // Calculate collision transform using meshScale combined with provided scale
            let combinedScale = SIMD3<Float>(meshScale, meshScale, meshScale) * scale
            let combinedOffset = offset + SIMD3<Float>(0.0, meshYOffset, 0.0)
            let combinedRotation = rotation + SIMD3<Float>(0.0, meshYRotation * Float.pi / 180.0, 0.0)
            let (transform, invTransform) = calculateCollisionTransform(
                meshMin: minBounds,
                meshMax: maxBounds,
                gridMin: gridBoundaryMin,
                gridMax: gridBoundaryMax,
                scale: combinedScale,
                rotation: combinedRotation,
                offset: combinedOffset
            )
            
            // Update collision uniforms
            let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
                to: CollisionUniforms.self,
                capacity: 1
            )
            
            collisionUniformPointer[0] = CollisionUniforms(
                sdfOrigin: minBounds,
                sdfSize: maxBounds - minBounds,
                sdfResolution: resolution,
                collisionStiffness: 1.0,  // Not directly used in new velocity-based approach
                collisionDamping: 0.8,    // Not directly used in new velocity-based approach
                enableCollision: 1,
                collisionTransform: transform,
                collisionInvTransform: invTransform
            )
            
            // Load mesh into renderer for visualization
            meshRenderer.loadMesh(triangles: triangles)
            
            print("Successfully loaded collision mesh with \(triangles.count) triangles")
            print("SDF resolution: \(resolution)x\(resolution)x\(resolution)")
            print("SDF bounds: \(minBounds) to \(maxBounds)")
        } else {
            print("Failed to generate SDF texture")
        }
    }
    
    // MARK: - Configuration
    func updateGridBoundaries(gridBoundaryMin: SIMD3<Float>, gridBoundaryMax: SIMD3<Float>) {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        
        // Get current collision bounds from the uniform
        let minBounds = collisionUniformPointer[0].sdfOrigin
        let maxBounds = collisionUniformPointer[0].sdfOrigin + collisionUniformPointer[0].sdfSize
        
        // Calculate collision transform using the helper method
        let (transform, invTransform) = calculateCollisionTransform(
            meshMin: minBounds,
            meshMax: maxBounds,
            gridMin: gridBoundaryMin,
            gridMax: gridBoundaryMax
        )
        
        // Update the collision transforms
        collisionUniformPointer[0].collisionTransform = transform
        collisionUniformPointer[0].collisionInvTransform = invTransform
        
        print("Updated collision transform for new grid boundaries")
    }
    
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
            vertexUniformBuffer: vertexUniformBuffer,
            collisionUniformBuffer: collisionUniformBuffer
        )
    }
    
    // New method to render within an existing render encoder
    internal func renderMeshInEncoder(renderEncoder: MTLRenderCommandEncoder,
                                    vertexUniformBuffer: MTLBuffer) {
        meshRenderer.renderInEncoder(
            renderEncoder: renderEncoder,
            vertexUniformBuffer: vertexUniformBuffer,
            collisionUniformBuffer: collisionUniformBuffer
        )
    }
}
