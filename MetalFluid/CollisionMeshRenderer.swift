import Foundation
import Metal
import MetalKit
import simd

struct CollisionVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
}

// Collision mesh uniforms structure (matching Metal shader)
struct CollisionMeshUniforms {
    var meshColor: simd_float4
}

class CollisionMeshRenderer {
    private let device: MTLDevice
    
    // Rendering resources
    private var meshBuffer: MTLBuffer?
    private var meshIndexBuffer: MTLBuffer?
    private var vertexCount: Int = 0
    private var indexCount: Int = 0
    private var solidPipelineState: MTLRenderPipelineState?
    private var wireframePipelineState: MTLRenderPipelineState?
    private var depthStencilState: MTLDepthStencilState?
    
    // Mesh data
    private var triangles: [Triangle] = []
    
    // Settings
    public var isVisible: Bool = true
    public var wireframeMode: Bool = false
    public var meshColor: SIMD4<Float> = SIMD4<Float>(1.0, 1.0, 1.0, 0.8) // Semi-transparent red
    
    // Mesh color uniforms buffer
    private var meshUniformsBuffer: MTLBuffer?
    
    init(device: MTLDevice) {
        self.device = device
        setupPipelines()
        setupDepthStencil()
        setupMeshUniforms()
    }
    
    private func setupMeshUniforms() {
        let uniformsSize = MemoryLayout<CollisionMeshUniforms>.stride
        meshUniformsBuffer = device.makeBuffer(length: uniformsSize, options: .storageModeShared)
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
        
        // Enable blending for transparency
        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = true
        colorAttachment.rgbBlendOperation = .add
        colorAttachment.alphaBlendOperation = .add
        colorAttachment.sourceRGBBlendFactor = .sourceAlpha
        colorAttachment.sourceAlphaBlendFactor = .sourceAlpha
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
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
        
        // Enable blending for wireframe
        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = true
        colorAttachment.rgbBlendOperation = .add
        colorAttachment.alphaBlendOperation = .add
        colorAttachment.sourceRGBBlendFactor = .sourceAlpha
        colorAttachment.sourceAlphaBlendFactor = .sourceAlpha
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
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
    
    func loadMesh(triangles: [Triangle]) {
        self.triangles = triangles
        generateMeshBuffers()
    }
    
    private func generateMeshBuffers() {
        guard !triangles.isEmpty else { return }
        
        var vertices: [CollisionVertex] = []
        var indices: [UInt32] = []
        
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
            
            // Add indices
            indices.append(baseIndex + 0)
            indices.append(baseIndex + 1)
            indices.append(baseIndex + 2)
        }
        
        vertexCount = vertices.count
        indexCount = indices.count
        
        // Create vertex buffer
        let vertexBufferSize = MemoryLayout<CollisionVertex>.stride * vertexCount
        meshBuffer = device.makeBuffer(bytes: vertices, length: vertexBufferSize, options: .storageModeShared)
        
        // Create index buffer
        let indexBufferSize = MemoryLayout<UInt32>.stride * indexCount
        meshIndexBuffer = device.makeBuffer(bytes: indices, length: indexBufferSize, options: .storageModeShared)
        
        print("Generated collision mesh with \(vertexCount) vertices, \(indexCount/3) triangles")
    }
    
    private func updateMeshUniforms() {
        guard let meshUniformsBuffer = meshUniformsBuffer else { return }
        
        let uniforms = CollisionMeshUniforms(meshColor: meshColor)
        
        let pointer = meshUniformsBuffer.contents().bindMemory(to: CollisionMeshUniforms.self, capacity: 1)
        pointer.pointee = uniforms
    }
    
    func render(renderPassDescriptor: MTLRenderPassDescriptor,
                commandBuffer: MTLCommandBuffer,
                vertexUniformBuffer: MTLBuffer,
                collisionUniformBuffer: MTLBuffer) {
        
        guard isVisible,
              let depthStencilState = depthStencilState,
              let meshBuffer = meshBuffer,
              let meshIndexBuffer = meshIndexBuffer,
              let meshUniformsBuffer = meshUniformsBuffer,
              indexCount > 0 else {
            return
        }
        
        // Update mesh uniforms
        updateMeshUniforms()
        
        // Select pipeline based on mode
        let pipelineState = wireframeMode ? wireframePipelineState : solidPipelineState
        guard let pipeline = pipelineState else {
            return
        }
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderEncoder.label = wireframeMode ? "Collision Mesh Wireframe" : "Collision Mesh Solid"
        renderEncoder.setRenderPipelineState(pipeline)
        renderEncoder.setDepthStencilState(depthStencilState)
        
        // Set vertex buffer and uniforms
        renderEncoder.setVertexBuffer(meshBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(vertexUniformBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(meshUniformsBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(collisionUniformBuffer, offset: 0, index: 3)
        
        // Draw triangles
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: meshIndexBuffer,
            indexBufferOffset: 0
        )
        
        renderEncoder.endEncoding()
    }
    
    // Render within an existing render encoder (doesn't create or end encoding)
    func renderInEncoder(renderEncoder: MTLRenderCommandEncoder,
                        vertexUniformBuffer: MTLBuffer,
                        collisionUniformBuffer: MTLBuffer) {
        
        guard isVisible,
              let depthStencilState = depthStencilState,
              let meshBuffer = meshBuffer,
              let meshIndexBuffer = meshIndexBuffer,
              let meshUniformsBuffer = meshUniformsBuffer,
              indexCount > 0 else {
            return
        }
        
        // Update mesh uniforms
        updateMeshUniforms()
        
        // Select pipeline based on mode
        let pipelineState = wireframeMode ? wireframePipelineState : solidPipelineState
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
        
        // Draw triangles
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: meshIndexBuffer,
            indexBufferOffset: 0
        )
    }
    
    // Utility methods
    func setColor(_ color: SIMD4<Float>) {
        meshColor = color
        // Note: For dynamic color changes, we'd need to pass color as uniform to shader
    }
    
    func setWireframeMode(_ enabled: Bool) {
        wireframeMode = enabled
        print("Collision mesh wireframe mode: \(enabled ? "ON" : "OFF")")
    }
}
