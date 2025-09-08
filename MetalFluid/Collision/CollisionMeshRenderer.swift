import Foundation
import Metal
import MetalKit
import simd
public class CollisionMeshRendererItem {
    private let device: MTLDevice
    
    // Rendering resources
    public var meshBuffer: MTLBuffer?
    public var meshIndexBuffer: MTLBuffer?
    public var wireframeIndexBuffer: MTLBuffer?
    public var vertexCount: Int = 0
    public var indexCount: Int = 0
    public var wireframeIndexCount: Int = 0
    
    // Mesh data
    private var triangles: [Triangle] = []
    
    // Settings
    public var wireframeMode: Bool = false
    public var meshColor: SIMD4<Float> = SIMD4<Float>(1.0, 1.0, 1.0, 0.8) // Semi-transparent red
    
    // Mesh color uniforms buffer
    public var meshUniformsBuffer: MTLBuffer?
    
    public init(device: MTLDevice) {
        self.device = device
        setupMeshUniforms()
    }
    
    private func setupMeshUniforms() {
        let uniformsSize = MemoryLayout<CollisionMeshUniforms>.stride
        meshUniformsBuffer = device.makeBuffer(length: uniformsSize, options: .storageModeShared)
    }
    
    func loadMesh(triangles: [Triangle]) {
        self.triangles = triangles
        generateMeshBuffers()
    }
    
    private func generateMeshBuffers() {
        guard !triangles.isEmpty else { return }
        
        var vertices: [CollisionVertex] = []
        var solidIndices: [UInt32] = []
        var wireframeIndices: [UInt32] = []
        
        // Convert triangles to vertices with normals
        for (triangleIndex, triangle) in triangles.enumerated() {
            let v0 = triangle.v0
            let v1 = triangle.v1
            let v2 = triangle.v2
            
            // Calculate face normal
            let edge1 = v1 - v0
            let edge2 = v2 - v0
            let normal = normalize(cross(edge1, edge2))
            
            // Add vertices
            let baseIndex = UInt32(triangleIndex * 3)
            vertices.append(CollisionVertex(position: v0, normal: normal))
            vertices.append(CollisionVertex(position: v1, normal: normal))
            vertices.append(CollisionVertex(position: v2, normal: normal))
            
            // Add solid indices (triangles)
            solidIndices.append(baseIndex + 0)
            solidIndices.append(baseIndex + 1)
            solidIndices.append(baseIndex + 2)
            
            // Add wireframe indices (lines for triangle edges)
            wireframeIndices.append(baseIndex + 0) // v0 -> v1
            wireframeIndices.append(baseIndex + 1)
            wireframeIndices.append(baseIndex + 1) // v1 -> v2
            wireframeIndices.append(baseIndex + 2)
            wireframeIndices.append(baseIndex + 2) // v2 -> v0
            wireframeIndices.append(baseIndex + 0)
        }
        
        vertexCount = vertices.count
        indexCount = solidIndices.count
        wireframeIndexCount = wireframeIndices.count
        
        // Create vertex buffer
        let vertexBufferSize = MemoryLayout<CollisionVertex>.stride * vertexCount
        meshBuffer = device.makeBuffer(bytes: vertices, length: vertexBufferSize, options: .storageModeShared)
        
        // Create solid index buffer
        let solidIndexBufferSize = MemoryLayout<UInt32>.stride * solidIndices.count
        meshIndexBuffer = device.makeBuffer(bytes: solidIndices, length: solidIndexBufferSize, options: .storageModeShared)
        
        // Create wireframe index buffer
        let wireframeIndexBufferSize = MemoryLayout<UInt32>.stride * wireframeIndices.count
        wireframeIndexBuffer = device.makeBuffer(bytes: wireframeIndices, length: wireframeIndexBufferSize, options: .storageModeShared)
        
        print("Generated collision mesh with \(vertexCount) vertices, \(indexCount/3) triangles, \(wireframeIndexCount/2) wireframe lines")
    }
    
    public func updateMeshUniforms() {
        guard let meshUniformsBuffer = meshUniformsBuffer else { return }
        
        let uniforms = CollisionMeshUniforms(meshColor: meshColor)
        
        let pointer = meshUniformsBuffer.contents().bindMemory(to: CollisionMeshUniforms.self, capacity: 1)
        pointer.pointee = uniforms
    }
    
    // Utility methods
    func setColor(_ color: SIMD4<Float>) {
        meshColor = color
        updateMeshUniforms() // Update uniforms immediately when color changes
    }
    
    func setWireframeMode(_ enabled: Bool) {
        wireframeMode = enabled
        print("Collision mesh wireframe mode: \(enabled ? "ON" : "OFF")")
    }
}
public class CollisionMeshRenderer {
    private let device: MTLDevice
    public var isVisible: Bool = true

    // Rendering resources
    private var solidPipelineState: MTLRenderPipelineState?
    private var wireframePipelineState: MTLRenderPipelineState?
    private var depthStencilState: MTLDepthStencilState?
    
    public init(device: MTLDevice) {
        self.device = device
        setupPipelines()
        setupDepthStencil()
    }
        
    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to create default library for collision mesh")
            return
        }
        
        // Setup solid pipeline
        setupSolidPipeline(library: library)
        
        // Setup wireframe pipeline
        setupWireframePipeline(library: library)
    }
    
    private func setupSolidPipeline(library: MTLLibrary) {
        guard let vertexFunction = library.makeFunction(name: "collisionMeshVertexShader"),
              let fragmentFunction = library.makeFunction(name: "collisionMeshFragmentShader") else {
            print("Failed to find collision mesh shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        // Disable blending for opaque mesh rendering
        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = false
        
        do {
            solidPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create solid collision mesh pipeline state: \(error)")
        }
    }
    
    private func setupWireframePipeline(library: MTLLibrary) {
        guard let vertexFunction = library.makeFunction(name: "collisionMeshWireframeVertexShader"),
              let fragmentFunction = library.makeFunction(name: "collisionMeshWireframeFragmentShader") else {
            print("Failed to find wireframe shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        // Disable blending for wireframe mesh rendering  
        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = false
        
        do {
            wireframePipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create wireframe collision mesh pipeline state: \(error)")
        }
    }
    
    private func setupDepthStencil() {
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
    }
        
    // Render within an existing render encoder (doesn't create or end encoding)
    func renderInEncoder(item:CollisionMeshRendererItem,renderEncoder: MTLRenderCommandEncoder,
                        vertexUniformBuffer: MTLBuffer,
                        collisionUniformBuffer: MTLBuffer) {
        
        guard isVisible,
              let depthStencilState = depthStencilState,
              let meshBuffer = item.meshBuffer,
              let meshIndexBuffer = item.meshIndexBuffer,
              let meshUniformsBuffer = item.meshUniformsBuffer,
              item.indexCount > 0 else {
            return
        }
        
        // Additional check for wireframe mode
        if item.wireframeMode && (item.wireframeIndexBuffer == nil || item.wireframeIndexCount == 0) {
            return
        }
        
        // Update mesh uniforms
        item.updateMeshUniforms()
        
        // Select pipeline based on mode
        let pipelineState = item.wireframeMode ? wireframePipelineState : solidPipelineState
        guard let pipeline = pipelineState else {
            return
        }
        
        renderEncoder.setRenderPipelineState(pipeline)
        renderEncoder.setDepthStencilState(depthStencilState)
        
        // Set vertex buffer and uniforms
        renderEncoder.setVertexBuffer(meshBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(vertexUniformBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(meshUniformsBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(collisionUniformBuffer, offset: 0, index: 3)
        
        // Draw based on mode
        if item.wireframeMode {
            // Draw wireframe as lines
            renderEncoder.drawIndexedPrimitives(
                type: .line,
                indexCount: item.wireframeIndexCount,
                indexType: .uint32,
                indexBuffer: item.wireframeIndexBuffer!,
                indexBufferOffset: 0
            )
        } else {
            // Draw solid triangles
            renderEncoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: item.indexCount,
                indexType: .uint32,
                indexBuffer: meshIndexBuffer,
                indexBufferOffset: 0
            )
        }
    }
    
    // AR mode rendering - uses AR frame matrices directly
    func renderInEncoderForAR(item: CollisionMeshRendererItem,
                             renderEncoder: MTLRenderCommandEncoder,
                             projectionMatrix: float4x4,
                             viewMatrix: float4x4,
                             collisionUniformBuffer: MTLBuffer) {
        
        guard isVisible,
              let depthStencilState = depthStencilState,
              let meshBuffer = item.meshBuffer,
              let meshIndexBuffer = item.meshIndexBuffer,
              let meshUniformsBuffer = item.meshUniformsBuffer,
              item.indexCount > 0 else {
            return
        }
        
        // Additional check for wireframe mode
        if item.wireframeMode && (item.wireframeIndexBuffer == nil || item.wireframeIndexCount == 0) {
            return
        }
        
        // Set up rendering pipeline
        let pipelineState = item.wireframeMode ? wireframePipelineState : solidPipelineState
        renderEncoder.setRenderPipelineState(pipelineState!)
        renderEncoder.setDepthStencilState(depthStencilState)
        
        // Set vertex data
        renderEncoder.setVertexBuffer(meshBuffer, offset: 0, index: 0)
        
        // Create AR vertex uniforms with minimal required fields
        var arVertexUniforms = VertexShaderUniforms(
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            gridSpacing: 1.0,  // Not used in collision rendering
            physicalDomainOrigin: SIMD3<Float>(0, 0, 0),  // Not used
            gridResolution: SIMD3<Int32>(0, 0, 0),  // Not used
            rest_density: 1.0,  // Not used
            particleSizeMultiplier: 1.0,  // Not used
        )
        renderEncoder.setVertexBytes(&arVertexUniforms, length: MemoryLayout<VertexShaderUniforms>.stride, index: 1)
        
        renderEncoder.setVertexBuffer(meshUniformsBuffer, offset: 0, index: 2)
        
        // Create AR collision uniforms with identity transform (AR meshes are already in world coordinates)
        let originalUniforms = collisionUniformBuffer.contents().bindMemory(to: CollisionUniforms.self, capacity: 1)[0]
        var arCollisionUniforms = originalUniforms
        arCollisionUniforms.collisionTransform = matrix_identity_float4x4
        arCollisionUniforms.collisionInvTransform = matrix_identity_float4x4
        renderEncoder.setVertexBytes(&arCollisionUniforms, length: MemoryLayout<CollisionUniforms>.stride, index: 3)
        
        // Draw based on mode
        if item.wireframeMode {
            renderEncoder.drawIndexedPrimitives(
                type: .line,
                indexCount: item.wireframeIndexCount,
                indexType: .uint32,
                indexBuffer: item.wireframeIndexBuffer!,
                indexBufferOffset: 0
            )
        } else {
            renderEncoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: item.indexCount,
                indexType: .uint32,
                indexBuffer: meshIndexBuffer,
                indexBufferOffset: 0
            )
        }
    }
}
