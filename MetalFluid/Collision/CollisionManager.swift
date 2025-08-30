import Foundation
import Metal
import MetalKit
import simd
struct Item {
    var scale: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 1.0)
    var translate: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, 0.0)
    var rotate: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, 0.0) // in radians
}

class CollisionItem{
    public var sdfTexture: MTLTexture?
    public var collisionUniformBuffer: MTLBuffer
    // Transform control
    public var item: Item = Item() {
        didSet {
            updateCollisionTransform()
        }
    }
    
    // Store current transform parameters for updates
    public var currentMeshMin: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    public var currentMeshMax: SIMD3<Float> = SIMD3<Float>(1, 1, 1)
    public var currentGridMin: SIMD3<Float>?
    public var currentGridMax: SIMD3<Float>?
    
    // MARK: - Private Helper Methods
    
    public func calculateCollisionTransformCenterOfBottom(meshMin: SIMD3<Float>, meshMax: SIMD3<Float>,
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
    private func updateCollisionTransform() {
        guard currentGridMin != nil && currentGridMax != nil else { return }
        
        let scaleVec = item.scale
        let offsetVec = item.translate
        let rotationVec = item.rotate
        let (transform, invTransform) = calculateCollisionTransformCenterOfBottom(
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
    private let device: MTLDevice
    init(device: MTLDevice) {
        self.device = device
        // Create collision uniform buffer
        let collisionUniformSize = MemoryLayout<CollisionUniforms>.stride
        guard let buffer = device.makeBuffer(length: collisionUniformSize, options: .storageModeShared) else {
            fatalError("Failed to create collision uniform buffer")
        }
        self.collisionUniformBuffer = buffer
        meshRendererItem = CollisionMeshRendererItem(device: device)
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
            sdfMassCenter: SIMD3<Float>(0.5, 0.5, 0.5),
            sdfResolution: SIMD3<Int32>(64, 64, 64),
            collisionStiffness: 1.0,  // Not used in velocity-based approach
            collisionDamping: 0.8,    // Not used in velocity-based approach
            enableCollision: 0, // Disabled by default
            sdfMass: 100.0,
            collisionTransform: matrix_identity_float4x4, // Default identity transform
            collisionInvTransform: matrix_identity_float4x4 // Default identity inverse transform
        )
    }
    // Store initial SDF state for reset functionality
    public var initialCollisionTransform: float4x4 = matrix_identity_float4x4
    public var initialCollisionInvTransform: float4x4 = matrix_identity_float4x4
    public var canMove: Bool = false
    public var useGravity: Bool = false
    // Current mesh data
    public var currentTriangles: [Triangle] = []
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
        let (transform, invTransform) = calculateCollisionTransformCenterOfBottom(
            meshMin: minBounds,
            meshMax: maxBounds,
            gridMin: gridBoundaryMin,
            gridMax: gridBoundaryMax
        )
        
        // Update the collision transforms (preserve mass)
        let prevMass = collisionUniformPointer[0].sdfMass
        collisionUniformPointer[0].collisionTransform = transform
        collisionUniformPointer[0].collisionInvTransform = invTransform
        collisionUniformPointer[0].sdfMass = prevMass
        
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
    
    func resetSDFTransform() {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        collisionUniformPointer[0].collisionTransform = initialCollisionTransform
        collisionUniformPointer[0].collisionInvTransform = initialCollisionInvTransform
        
        print("🔄 SDF transform reset to initial position")
    }
        
    // MARK: - Internal Access (for FluidRenderer)
    
    internal func getSDFTexture() -> MTLTexture? {
        return sdfTexture
    }
    
    internal func getCollisionUniformBuffer() -> MTLBuffer {
        return collisionUniformBuffer
    }
    var meshRendererItem: CollisionMeshRendererItem
    // MARK: - Rendering Controls
    func setMeshColor(_ color: SIMD4<Float>) {
        meshRendererItem.setColor(color)
    }
    
    func setMeshWireframe(_ wireframe: Bool) {
        meshRendererItem.setWireframeMode(wireframe)
    }
    let padding: Float = 1.0
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

    public func processAndGenerateSDF(sdfGenerator: SDFGenerator,triangles: [Triangle], resolution: SIMD3<Int32>, gridBoundaryMin: SIMD3<Float>?, gridBoundaryMax: SIMD3<Float>?) {
        
        let scale: SIMD3<Float> = SIMD3<Float>(1, 1, 1)
        let offset: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
        let rotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
        
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
        
        sdfTexture = sdfGenerator.generateSDF(
            triangles: triangles,
            resolution: resolution,
            boundingBox: (min: minBounds, max: maxBounds)
        )
                
        if sdfTexture != nil {
            // Calculate collision transform using item scale combined with provided scale
            let combinedScale = item.scale * scale
            let combinedOffset = offset + item.translate
            let combinedRotation = rotation + item.rotate
            let (transform, invTransform) = calculateCollisionTransformCenterOfBottom(
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
            
            // Estimate mass from AABB volume (simple proxy), allow future override
            let extent = (maxBounds - minBounds)
            let volume = max(1e-4, extent.x * extent.y * extent.z)
            let estimatedMass: Float = volume * 2.5 // density coefficient

            // Compute mass center as area-weighted triangle centroids (approx surface COM)
            var weightedSum = SIMD3<Float>(repeating: 0)
            var totalArea: Float = 0
            for tri in triangles {
                let a = tri.v0, b = tri.v1, c = tri.v2
                let area = length(cross(b - a, c - a)) * 0.5
                let centroid = (a + b + c) / 3.0
                weightedSum += centroid * area
                totalArea += area
            }
            let surfCenter = totalArea > 0 ? (weightedSum / totalArea) : ((minBounds + maxBounds) * 0.5)

            collisionUniformPointer[0] = CollisionUniforms(
                sdfOrigin: minBounds,
                sdfSize: maxBounds - minBounds,
                sdfMassCenter: surfCenter,
                sdfResolution: resolution,
                collisionStiffness: 1.0,  // Not directly used in new velocity-based approach
                collisionDamping: 0.8,    // Not directly used in new velocity-based approach
                enableCollision: 1,
                sdfMass: estimatedMass,
                collisionTransform: transform,
                collisionInvTransform: invTransform
            )
            
            // Store initial transform for reset functionality
            initialCollisionTransform = transform
            initialCollisionInvTransform = invTransform
            
            
            // Load mesh into renderer for visualization
            meshRendererItem.loadMesh(triangles: triangles)
            
            print("Successfully loaded collision mesh with \(triangles.count) triangles")
            print("SDF resolution: \(resolution)x\(resolution)x\(resolution)")
            print("SDF bounds: \(minBounds) to \(maxBounds)")
        } else {
            print("Failed to generate SDF texture")
        }
    }

}
class CollisionManager {
    static let MAX_COLLISION_SDF = 8
    private let device: MTLDevice
    
    // SDF generation and collision detection
    public var sdfGenerator: SDFGenerator
    
    // Mesh rendering
    private var meshRenderer: CollisionMeshRenderer
    
    init(device: MTLDevice) {
        self.device = device
        self.sdfGenerator = SDFGenerator(device: device)
        self.meshRenderer = CollisionMeshRenderer(device: device)
        self.representativeItem = CollisionItem(device: device)
    }
    private func renderMesh(item:CollisionItem,renderPassDescriptor: MTLRenderPassDescriptor,
                           commandBuffer: MTLCommandBuffer,
                           vertexUniformBuffer: MTLBuffer) {
        meshRenderer.render(
            item:item.meshRendererItem,
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            vertexUniformBuffer: vertexUniformBuffer,
            collisionUniformBuffer: item.collisionUniformBuffer
        )
    }
    
    // New method to render within an existing render encoder
    private func renderMeshInEncoder(item:CollisionItem,renderEncoder: MTLRenderCommandEncoder,
                                    vertexUniformBuffer: MTLBuffer) {
        meshRenderer.renderInEncoder(
            item:item.meshRendererItem,
            renderEncoder: renderEncoder,
            vertexUniformBuffer: vertexUniformBuffer,
            collisionUniformBuffer: item.collisionUniformBuffer
        )
    }
    func renderMeshesInEncoder(renderEncoder: MTLRenderCommandEncoder,
                             vertexUniformBuffer: MTLBuffer) {
        for item in items {
            self.renderMeshInEncoder(item: item, renderEncoder: renderEncoder, vertexUniformBuffer: vertexUniformBuffer)
        }
    }
    private var internalItems:[CollisionItem] = []
    public var representativeItem:CollisionItem
    public var items:[CollisionItem]{
        get{
            return Array(([representativeItem] + internalItems).prefix(CollisionManager.MAX_COLLISION_SDF))
        }
    }
    func updateGridBoundaries(gridBoundaryMin: SIMD3<Float>, gridBoundaryMax: SIMD3<Float>) {
        for item in items {
            item.updateGridBoundaries(gridBoundaryMin: gridBoundaryMin, gridBoundaryMax: gridBoundaryMax)
        }
    }
    func resetSDFTransforms(){
        for item in items {
            item.resetSDFTransform()
        }
    }
    func isMeshVisible() -> Bool {
        return meshRenderer.isVisible
    }
    func setMeshVisible(_ visible: Bool) {
        meshRenderer.isVisible = visible
    }    
}
