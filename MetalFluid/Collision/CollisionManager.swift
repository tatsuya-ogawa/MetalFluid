import Foundation
import Metal
import MetalKit
import simd

class CollisionItem{
    private struct SdfTransform {
        var scale: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 1.0)
        var translate: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, 0.0)
        var rotate: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, 0.0) // in radians
    }

    public var sdfTexture: MTLTexture?
    public var collisionUniformBuffer: MTLBuffer
    // Transform control
    private var sdfTransform: SdfTransform = SdfTransform()
    
    // Store current transform parameters for updates
    public var currentMeshMin: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    public var currentMeshMax: SIMD3<Float> = SIMD3<Float>(1, 1, 1)
    public var currentGridMin: SIMD3<Float>?
    public var currentGridMax: SIMD3<Float>?
    public let isARMode: Bool
    // MARK: - Private Helper Methods
    
    public func calculateCollisionTransformCenterOfBottom(
        meshMin: SIMD3<Float>, meshMax: SIMD3<Float>,
        gridMin: SIMD3<Float>?, gridMax: SIMD3<Float>?,
        scale: SIMD3<Float> = SIMD3<Float>(1, 1, 1),
        rotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0),
        offset: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    ) -> (float4x4, float4x4) {
        guard let gridMin = gridMin, let gridMax = gridMax else {
            return (matrix_identity_float4x4, matrix_identity_float4x4)
        }

        let gridCenter = (gridMin + gridMax) * 0.5
        let meshCenter = (meshMin + meshMax) * 0.5

        // ▼ Pivot: Center of mesh bottom face
        let pivot = SIMD3<Float>(meshCenter.x, meshMin.y, meshCenter.z)

        // ▼ Target position on grid (where bottom face center should be placed) + offset
        let target = SIMD3<Float>(gridCenter.x + offset.x,
                                  gridMin.y   + offset.y,
                                  gridCenter.z + offset.z)

        // Matrices (rotation order: X→Y→Z)
        let S = float4x4(scaling: scale)
        let Rx = float4x4(rotationX: rotation.x)
        let Ry = float4x4(rotationY: rotation.y)
        let Rz = float4x4(rotationZ: rotation.z)
        let R = Rx * Ry * Rz

        // Translations for rotating and scaling around pivot
        let T_toPivot   = float4x4(translation: -pivot)
        let T_fromPivot = float4x4(translation:  pivot)

        // Finally, translate everything so that pivot (=bottom face center) aligns with target
        // Since rotation and scaling around pivot doesn't move the pivot itself, the difference is simply target - pivot
        let T_align = float4x4(translation: target - pivot)

        // Composition (applied from right): T_align * T_fromPivot * R * S * T_toPivot
        let transform = T_align * T_fromPivot * R * S * T_toPivot
        let invTransform = transform.inverse

        return (transform, invTransform)
    }
    
    private let device: MTLDevice
    init(device: MTLDevice,isARMode:Bool) {
        self.device = device
        // Create collision uniform buffer
        let collisionUniformSize = MemoryLayout<CollisionUniforms>.stride
        guard let buffer = device.makeBuffer(length: collisionUniformSize, options: .storageModeShared) else {
            fatalError("Failed to create collision uniform buffer")
        }
        self.collisionUniformBuffer = buffer
        self.isARMode = isARMode
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
            collisionFlags: 0, // All flags disabled by default
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
    public var skipRenderInAR: Bool = false
    // Current mesh data
    public var currentTriangles: [Triangle] = []
    func updateGridBoundaries(gridBoundaryMin: SIMD3<Float>, gridBoundaryMax: SIMD3<Float>) {
        currentGridMin = gridBoundaryMin
        currentGridMax = gridBoundaryMax
        updateCollisionTransformToCenter(item: sdfTransform,currentMeshMin: currentMeshMin, currentMeshMax: currentMeshMax, currentGridMin: currentGridMin, currentGridMax: currentGridMax)
    }
    // MARK: - Configuration
    private func updateCollisionTransformToCenter(item:SdfTransform, currentMeshMin: SIMD3<Float>, currentMeshMax: SIMD3<Float>, currentGridMin: SIMD3<Float>?, currentGridMax: SIMD3<Float>?) {
        guard currentGridMin != nil && currentGridMax != nil else { return }
        
        let scaleVec = item.scale
        let offsetVec = item.translate
        let rotationVec = item.rotate
        
        let (transform, invTransform) = isARMode ? 
            (matrix_identity_float4x4, matrix_identity_float4x4) :
            calculateCollisionTransformCenterOfBottom(
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
   
    func setEnabled(_ enabled: Bool) {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        if enabled {
            collisionUniformPointer[0].collisionFlags |= 1  // Set COLLISION_ENABLE_BIT
        } else {
            collisionUniformPointer[0].collisionFlags &= ~1  // Clear COLLISION_ENABLE_BIT
        }
    }
    
    func isEnabled() -> Bool {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return (collisionUniformPointer[0].collisionFlags & 1) != 0  // Check COLLISION_ENABLE_BIT
    }
    
    func setStaticSDF(_ isStatic: Bool) {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        if isStatic {
            collisionUniformPointer[0].collisionFlags |= 2  // Set COLLISION_STATIC_BIT
        } else {
            collisionUniformPointer[0].collisionFlags &= ~2  // Clear COLLISION_STATIC_BIT
        }
    }
    
    func isStaticSDF() -> Bool {
        let collisionUniformPointer = collisionUniformBuffer.contents().bindMemory(
            to: CollisionUniforms.self,
            capacity: 1
        )
        return (collisionUniformPointer[0].collisionFlags & 2) != 0  // Check COLLISION_STATIC_BIT
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
        currentTriangles = triangles
        
        // Store current parameters for scale updates
        let boundingBox = calculateBoundingBox(triangles: triangles)
        currentMeshMin = boundingBox.min
        currentMeshMax = boundingBox.max
        currentGridMin = gridBoundaryMin
        currentGridMax = gridBoundaryMax
        // Expand bounds slightly for safety
        let meshSize = boundingBox.max - boundingBox.min
        let padding = meshSize * 0.025  // 2.5% padding
        let minBounds = boundingBox.min - padding
        let maxBounds = boundingBox.max + padding
        
        sdfTexture = sdfGenerator.generateSDF(
            triangles: triangles,
            resolution: resolution,
            boundingBox: (min: minBounds, max: maxBounds)
        )
                
        if sdfTexture != nil {
            // Calculate collision transform using item scale combined with provided scale
            let combinedScale = sdfTransform.scale
            let combinedOffset = sdfTransform.translate
            let combinedRotation = sdfTransform.rotate
            
            let (transform, invTransform) = isARMode ?
                (matrix_identity_float4x4, matrix_identity_float4x4) :
                calculateCollisionTransformCenterOfBottom(
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
                collisionFlags: 1,  // Enable collision by default
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
    public func updateSdfTransform(
        scale: SIMD3<Float>?,
        translate: SIMD3<Float>?,
        rotate: SIMD3<Float>?){
            self.sdfTransform = SdfTransform(scale: scale ?? sdfTransform.scale, translate: translate ?? sdfTransform.translate, rotate: rotate ?? sdfTransform.rotate)
            updateCollisionTransformToCenter(item: sdfTransform,currentMeshMin: currentMeshMin, currentMeshMax: currentMeshMax, currentGridMin: currentGridMin, currentGridMax: currentGridMax)
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
        self.representativeItem = CollisionItem(device: device,isARMode: false)
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
    
    private func renderMeshInEncoderForAR(item: CollisionItem, 
                                        renderEncoder: MTLRenderCommandEncoder, 
                                        projectionMatrix: float4x4, 
                                        viewMatrix: float4x4) {
        meshRenderer.renderInEncoderForAR(
            item: item.meshRendererItem,
            renderEncoder: renderEncoder,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            collisionUniformBuffer: item.collisionUniformBuffer
        )
    }
    
    func renderMeshesInEncoder(renderEncoder: MTLRenderCommandEncoder,
                             vertexUniformBuffer: MTLBuffer) {
        for item in items {
            self.renderMeshInEncoder(item: item, renderEncoder: renderEncoder, vertexUniformBuffer: vertexUniformBuffer)
        }
    }
    
    // AR mode rendering - uses AR frame matrices instead of fluid scene matrices
    func renderMeshesInEncoderForAR(renderEncoder: MTLRenderCommandEncoder,
                                   projectionMatrix: float4x4,
                                   viewMatrix: float4x4) {
        for item in items {
            // Skip rendering if skipRenderInAR is enabled for this item
            if item.skipRenderInAR {
                continue
            }
            self.renderMeshInEncoderForAR(item: item, 
                                        renderEncoder: renderEncoder, 
                                        projectionMatrix: projectionMatrix, 
                                        viewMatrix: viewMatrix)
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
    
    // Debug support methods for SDF visualization
    func getFirstSDFTexture() -> MTLTexture? {
        return items.first?.sdfTexture
    }
    
    func getFirstCollisionUniformsBuffer() -> MTLBuffer? {
        return items.first?.collisionUniformBuffer
    }
}
