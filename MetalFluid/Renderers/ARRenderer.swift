//
//  ARRenderer.swift
//  MetalFluid
//
//  ARKit background and mesh rendering with fallback for non-AR environments
//

import Metal
import MetalKit
#if canImport(ARKit)
import ARKit
#endif

class ARRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Pipeline states
    private var cameraBackgroundPipelineState: MTLRenderPipelineState?
    private var arMeshPipelineState: MTLRenderPipelineState?
    private var arMeshWireframePipelineState: MTLRenderPipelineState?
    
    // Camera background resources
    private var cameraBackgroundVertexBuffer: MTLBuffer?
    private var cameraImageTextureY: MTLTexture?
    private var cameraImageTextureCbCr: MTLTexture?
    private let cameraImageTextureCache: CVMetalTextureCache?
    
    // AR Mesh resources
    private var meshVertexBuffer: MTLBuffer?
    private var meshIndexBuffer: MTLBuffer?
    private var meshUniformBuffer: MTLBuffer?
    private var meshVertexCount: Int = 0
    private var meshIndexCount: Int = 0
    
    // AR capability check
    public let isARSupported: Bool
    
    #if canImport(ARKit)
    private var arSession: ARSession?
    private var currentMeshAnchors: [ARMeshAnchor] = []
    #endif
    
    // SDF generation
    private var sdfGenerator: SDFGenerator?
    private var currentARSDF: MTLTexture?
    private var sdfBoundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)?
    
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        
        // Check AR support
        #if canImport(ARKit)
        if #available(iOS 11.0, macOS 10.13, *) {
            self.isARSupported = ARWorldTrackingConfiguration.isSupported
        } else {
            self.isARSupported = false
        }
        #else
        self.isARSupported = false
        #endif
        
        // Create texture cache
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        self.cameraImageTextureCache = cache
        
        // Initialize SDF generator
        sdfGenerator = SDFGenerator(device: device)
        
        setupPipelineStates()
        setupCameraBackgroundGeometry()
    }
    
    private func setupPipelineStates() {
        guard let library = device.makeDefaultLibrary() else {
            print("❌ Failed to create default library for AR shaders")
            return
        }
        
        // Camera Background Pipeline
        if let vertexFunction = library.makeFunction(name: "cameraBackgroundVertex"),
           let fragmentFunction = library.makeFunction(name: "cameraBackgroundFragment") {
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            
            // Vertex descriptor for camera background
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float2
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.attributes[1].format = .float2
            vertexDescriptor.attributes[1].offset = 8
            vertexDescriptor.attributes[1].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = 16
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            
            do {
                cameraBackgroundPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                print("✅ AR Camera background pipeline state created successfully")
            } catch {
                print("❌ Failed to create camera background pipeline state: \\(error)")
            }
        }
        
        // AR Mesh Pipeline (solid)
        if let vertexFunction = library.makeFunction(name: "arMeshVertex"),
           let fragmentFunction = library.makeFunction(name: "arMeshSolidFragment") {
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            
            // Vertex descriptor for AR mesh
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float3
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.attributes[1].format = .float3
            vertexDescriptor.attributes[1].offset = 12
            vertexDescriptor.attributes[1].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = 24
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            
            do {
                arMeshPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                print("✅ AR Mesh pipeline state created successfully")
            } catch {
                print("❌ Failed to create AR mesh pipeline state: \\(error)")
            }
        }
        
        // AR Mesh Wireframe Pipeline
        if let vertexFunction = library.makeFunction(name: "arMeshVertex"),
           let fragmentFunction = library.makeFunction(name: "arMeshWireframeFragment") {
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            
            // Vertex descriptor for AR mesh
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float3
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.attributes[1].format = .float3
            vertexDescriptor.attributes[1].offset = 12
            vertexDescriptor.attributes[1].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = 24
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            
            do {
                arMeshWireframePipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                print("✅ AR Mesh wireframe pipeline state created successfully")
            } catch {
                print("❌ Failed to create AR mesh wireframe pipeline state: \\(error)")
            }
        }
        
        // Create uniform buffer
        meshUniformBuffer = device.makeBuffer(length: MemoryLayout<ARMeshUniforms>.stride, options: [])
    }
    
    private func setupCameraBackgroundGeometry() {
        // Full-screen quad vertices with texture coordinates
        let vertices: [Float] = [
            -1.0, -1.0,  0.0, 1.0,  // Bottom left
             1.0, -1.0,  1.0, 1.0,  // Bottom right
            -1.0,  1.0,  0.0, 0.0,  // Top left
             1.0,  1.0,  1.0, 0.0   // Top right
        ]
        
        cameraBackgroundVertexBuffer = device.makeBuffer(bytes: vertices, 
                                                        length: vertices.count * MemoryLayout<Float>.stride, 
                                                        options: [])
    }
    
    // MARK: - ARKit Integration
    
    #if canImport(ARKit)
    @available(iOS 11.0, macOS 10.13, *)
    func startARSession() {
        guard isARSupported else {
            print("⚠️ ARKit is not supported on this device")
            return
        }
        
        arSession = ARSession()
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        
        if #available(iOS 13.4, *) {
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
                configuration.sceneReconstruction = .mesh
                print("✅ Scene reconstruction enabled")
            }
        }
        
        arSession?.run(configuration)
        print("✅ AR session started")
    }
    
    @available(iOS 11.0, macOS 10.13, *)
    func stopARSession() {
        arSession?.pause()
        arSession = nil
        print("🛑 AR session stopped")
    }
    
    @available(iOS 11.0, macOS 10.13, *)
    func updateCameraTextures(from frame: ARFrame) {
        let pixelBuffer = frame.capturedImage
        
        // Y texture
        var yTextureRef: CVMetalTexture?
        let yWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                cameraImageTextureCache!,
                                                pixelBuffer,
                                                nil,
                                                .r8Unorm,
                                                yWidth, yHeight,
                                                0,
                                                &yTextureRef)
        cameraImageTextureY = CVMetalTextureGetTexture(yTextureRef!)
        
        // CbCr texture
        var cbCrTextureRef: CVMetalTexture?
        let cbCrWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let cbCrHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
        
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                cameraImageTextureCache!,
                                                pixelBuffer,
                                                nil,
                                                .rg8Unorm,
                                                cbCrWidth, cbCrHeight,
                                                1,
                                                &cbCrTextureRef)
        cameraImageTextureCbCr = CVMetalTextureGetTexture(cbCrTextureRef!)
    }
    
    @available(iOS 13.4, macOS 10.15.4, *)
    func updateMeshGeometry(from meshAnchors: [ARMeshAnchor]) {
        // For now, just use the first mesh anchor
        guard let meshAnchor = meshAnchors.first else { return }
        
        let geometry = meshAnchor.geometry
        let vertices = geometry.vertices
        let normals = geometry.normals
        let faces = geometry.faces
        
        // Create combined vertex data (position + normal)
        var vertexData: [Float] = []
        let vertexCount = vertices.count
        
        // Access buffer data directly
        let vertexBuffer = vertices.buffer
        let normalBuffer = normals.buffer
        let vertexStride = vertices.stride
        let normalStride = normals.stride
        
        for i in 0..<vertexCount {
            let vertexOffset = i * vertexStride
            let normalOffset = i * normalStride
            
            let vertexPointer = vertexBuffer.contents().advanced(by: vertexOffset).assumingMemoryBound(to: Float.self)
            let normalPointer = normalBuffer.contents().advanced(by: normalOffset).assumingMemoryBound(to: Float.self)
            
            // Add vertex position (3 components)
            vertexData.append(vertexPointer[0])
            vertexData.append(vertexPointer[1])
            vertexData.append(vertexPointer[2])
            
            // Add vertex normal (3 components)
            vertexData.append(normalPointer[0])
            vertexData.append(normalPointer[1])
            vertexData.append(normalPointer[2])
        }
        
        // Create vertex buffer
        meshVertexBuffer = device.makeBuffer(bytes: vertexData,
                                           length: vertexData.count * MemoryLayout<Float>.stride,
                                           options: [])
        
        // Create index buffer
        let indexData = faces.buffer
        meshIndexBuffer = device.makeBuffer(bytes: indexData.contents(),
                                          length: indexData.length,
                                          options: [])
        
        meshVertexCount = vertexCount
        meshIndexCount = faces.count * 3 // Triangles
        
        print("📐 Updated mesh: \\(meshVertexCount) vertices, \\(meshIndexCount) indices")
        
        // Update current mesh anchors for SDF generation
        currentMeshAnchors = [meshAnchor]
    }
    
    @available(iOS 13.4, macOS 10.15.4, *)
    func updateMeshGeometryFromSession(meshAnchors: [ARMeshAnchor]) {
        currentMeshAnchors = meshAnchors
        
        // For rendering, use the first mesh anchor (existing behavior)
        if let firstMesh = meshAnchors.first {
            updateMeshGeometry(from: [firstMesh])
        }
    }
    #endif
    
    // MARK: - SDF Generation
    
    #if canImport(ARKit)
    @available(iOS 13.4, macOS 10.15.4, *)
    func generateSDFFromCurrentMeshes(resolution: SIMD3<Int32>? = nil, forceRegenerate: Bool = false) -> MTLTexture? {
        guard let sdfGenerator = sdfGenerator else {
            print("❌ SDF generator not available")
            return nil
        }
        
        guard !currentMeshAnchors.isEmpty else {
            print("⚠️ No AR mesh anchors available for SDF generation")
            return nil
        }
        
        // Return existing SDF if available and not forced to regenerate
        if !forceRegenerate, let existingSDF = currentARSDF {
            return existingSDF
        }
        
        print("🔄 Generating SDF from \(currentMeshAnchors.count) AR mesh anchors...")
        
        let sdfTexture = sdfGenerator.generateCombinedSDFFromARMeshes(currentMeshAnchors, resolution: resolution)
        
        if let sdf = sdfTexture {
            currentARSDF = sdf
            print("✅ AR SDF generation successful")
        } else {
            print("❌ AR SDF generation failed")
        }
        
        return sdfTexture
    }
    
    @available(iOS 13.4, macOS 10.15.4, *)
    func getCurrentARSDF() -> MTLTexture? {
        return currentARSDF
    }
    
    @available(iOS 13.4, macOS 10.15.4, *)
    func getARSDFBoundingBox() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        return sdfBoundingBox
    }
    
    @available(iOS 13.4, macOS 10.15.4, *)
    func hasARMeshData() -> Bool {
        return !currentMeshAnchors.isEmpty
    }
    #endif
    
    // MARK: - Rendering
    
    func renderCameraBackground(commandEncoder: MTLRenderCommandEncoder) {
        guard let pipelineState = cameraBackgroundPipelineState,
              let vertexBuffer = cameraBackgroundVertexBuffer else { return }
        
        // Non-AR fallback: render solid background
        if !isARSupported {
            renderFallbackBackground(commandEncoder: commandEncoder)
            return
        }
        
        #if canImport(ARKit)
        guard let yTexture = cameraImageTextureY,
              let cbCrTexture = cameraImageTextureCbCr else {
            renderFallbackBackground(commandEncoder: commandEncoder)
            return
        }
        
        commandEncoder.setRenderPipelineState(pipelineState)
        commandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        commandEncoder.setFragmentTexture(yTexture, index: 0)
        commandEncoder.setFragmentTexture(cbCrTexture, index: 1)
        commandEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        #else
        renderFallbackBackground(commandEncoder: commandEncoder)
        #endif
    }
    
    private func renderFallbackBackground(commandEncoder: MTLRenderCommandEncoder) {
        // Render a simple dark background when ARKit is not available
        // This could be enhanced with a test pattern or gradient
    }
    
    func renderARMesh(commandEncoder: MTLRenderCommandEncoder, 
                     viewMatrix: simd_float4x4, 
                     projectionMatrix: simd_float4x4,
                     wireframe: Bool = true) {
        
        guard let uniformBuffer = meshUniformBuffer else { return }
        
        let pipelineState = wireframe ? arMeshWireframePipelineState : arMeshPipelineState
        guard let pipeline = pipelineState else { return }
        
        // Non-AR fallback: render demo mesh
        if !isARSupported || meshVertexBuffer == nil {
            renderDemoMesh(commandEncoder: commandEncoder, 
                          viewMatrix: viewMatrix, 
                          projectionMatrix: projectionMatrix,
                          wireframe: wireframe)
            return
        }
        
        guard let vertexBuffer = meshVertexBuffer,
              let indexBuffer = meshIndexBuffer,
              meshIndexCount > 0 else { return }
        
        // Update uniforms
        let modelMatrix = simd_float4x4(1.0) // Identity for now
        let modelViewProjection = projectionMatrix * viewMatrix * modelMatrix
        let normalMatrix = simd_float4x4(1.0) // Should be transpose(inverse(modelMatrix))
        
        var uniforms = ARMeshUniforms(
            modelViewProjectionMatrix: modelViewProjection,
            modelMatrix: modelMatrix,
            normalMatrix: normalMatrix,
            meshColor: SIMD4<Float>(0.0, 1.0, 1.0, 1.0), // Cyan
            opacity: 0.6
        )
        
        uniformBuffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<ARMeshUniforms>.stride)
        
        commandEncoder.setRenderPipelineState(pipeline)
        commandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        commandEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 0)
        commandEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)
        commandEncoder.drawIndexedPrimitives(type: .triangle,
                                           indexCount: meshIndexCount,
                                           indexType: .uint16,
                                           indexBuffer: indexBuffer,
                                           indexBufferOffset: 0)
    }
    
    private func renderDemoMesh(commandEncoder: MTLRenderCommandEncoder,
                               viewMatrix: simd_float4x4,
                               projectionMatrix: simd_float4x4,
                               wireframe: Bool) {
        // Create a simple demo cube when AR is not available
        createDemoCube()
        
        guard let vertexBuffer = meshVertexBuffer,
              let indexBuffer = meshIndexBuffer,
              let uniformBuffer = meshUniformBuffer else { return }
        
        let pipelineState = wireframe ? arMeshWireframePipelineState : arMeshPipelineState
        guard let pipeline = pipelineState else { return }
        
        // Update uniforms for demo cube
        let modelMatrix = simd_float4x4(1.0)
        let modelViewProjection = projectionMatrix * viewMatrix * modelMatrix
        let normalMatrix = simd_float4x4(1.0)
        
        var uniforms = ARMeshUniforms(
            modelViewProjectionMatrix: modelViewProjection,
            modelMatrix: modelMatrix,
            normalMatrix: normalMatrix,
            meshColor: SIMD4<Float>(1.0, 0.5, 0.0, 1.0), // Orange
            opacity: 0.8
        )
        
        uniformBuffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<ARMeshUniforms>.stride)
        
        commandEncoder.setRenderPipelineState(pipeline)
        commandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        commandEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 0)
        commandEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)
        commandEncoder.drawIndexedPrimitives(type: .triangle,
                                           indexCount: meshIndexCount,
                                           indexType: .uint16,
                                           indexBuffer: indexBuffer,
                                           indexBufferOffset: 0)
    }
    
    private func createDemoCube() {
        // Simple cube vertices (position + normal)
        let cubeVertices: [Float] = [
            // Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            // Back face
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0
        ]
        
        let cubeIndices: [UInt16] = [
            0, 1, 2, 2, 3, 0,   // Front
            4, 6, 5, 6, 4, 7,   // Back
            3, 2, 6, 6, 7, 3,   // Top
            0, 4, 5, 5, 1, 0,   // Bottom
            0, 3, 7, 7, 4, 0,   // Left
            1, 5, 6, 6, 2, 1    // Right
        ]
        
        meshVertexBuffer = device.makeBuffer(bytes: cubeVertices,
                                           length: cubeVertices.count * MemoryLayout<Float>.stride,
                                           options: [])
        
        meshIndexBuffer = device.makeBuffer(bytes: cubeIndices,
                                          length: cubeIndices.count * MemoryLayout<UInt16>.stride,
                                          options: [])
        
        meshVertexCount = 8
        meshIndexCount = 36
    }
}