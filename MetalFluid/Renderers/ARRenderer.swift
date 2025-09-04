//
//  ARRenderer.swift
//  MetalFluid
//
//  ARKit background and mesh rendering with fallback for non-AR environments
//

import Metal
import MetalKit
import simd
import UIKit
#if canImport(ARKit)
import ARKit
#endif

class ARRenderer:NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Pipeline states
    private var cameraBackgroundPipelineState: MTLRenderPipelineState?
    private var arMeshWirePipelineState: MTLRenderPipelineState?
    private var arMeshWireDepthStencilState: MTLDepthStencilState?
    private var arMeshSolidPipelineState: MTLRenderPipelineState?
    private var arMeshSolidDepthStencilState: MTLDepthStencilState?
    
    // Camera background resources
    private var cameraBackgroundVertexBuffer: MTLBuffer?
    private var cameraImageTextureY: MTLTexture?
    private var cameraImageTextureCbCr: MTLTexture?
    private let cameraImageTextureCache: CVMetalTextureCache?
    
    // AR capability check
    public let isARSupported: Bool
    
    #if canImport(ARKit)
    private var arSession: ARSession?
    private var currentMeshAnchors: [ARMeshAnchor] = []
    #endif

    // AR mesh wireframe buffers
    private var arMeshVertexBuffer: MTLBuffer?
    private var arMeshLineIndexBuffer: MTLBuffer?
    private var arMeshLineIndexCount: Int = 0
    public var showARMeshWireframe: Bool = true
    
    // AR mesh solid buffers
    private var arMeshSolidVertexBuffer: MTLBuffer?
    private var arMeshSolidNormalBuffer: MTLBuffer?
    private var arMeshSolidIndexBuffer: MTLBuffer?
    private var arMeshSolidIndexCount: Int = 0
    public var showARMeshSolid: Bool = true
    
    // SDF generation
    private var sdfGenerator: SDFGenerator?
    private var currentARSDF: MTLTexture?
    private var sdfBoundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)?
    
    // Raycast and mesh extraction
    public var selectedMeshBounds: (center: SIMD3<Float>, size: SIMD3<Float>)?
    private var lastTapWorldPosition: SIMD3<Float>?
    
    // GPU Raycast compute pipeline
    private var gpuRaycastPipelineState: MTLComputePipelineState?
    private var meshExtractionPipelineState: MTLComputePipelineState?
    
    // GPU Raycast buffers
    private var allTrianglesBuffer: MTLBuffer?
    private var raycastResultBuffer: MTLBuffer?
    private var extractedTrianglesBuffer: MTLBuffer?
    private var outputCountBuffer: MTLBuffer?
    
    
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
        super.init()
        setupPipelineStates()
        setupCameraBackgroundGeometry()
        setupGPURaycastPipelines()
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
//            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            
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

        // AR Mesh Wireframe Pipeline
        if let vertexFunction = library.makeFunction(name: "arMeshWireVertex"),
           let fragmentFunction = library.makeFunction(name: "arMeshWireFragment") {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            do {
                arMeshWirePipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                let depthDesc = MTLDepthStencilDescriptor()
                depthDesc.isDepthWriteEnabled = false
                depthDesc.depthCompareFunction = .always // draw on top as overlay
                arMeshWireDepthStencilState = device.makeDepthStencilState(descriptor: depthDesc)
            } catch {
                print("❌ Failed to create AR mesh wireframe pipeline state: \\(error)")
            }
        }
        
        // AR Mesh Solid Pipeline
        if let vertexFunction = library.makeFunction(name: "arMeshSolidVertex"),
           let fragmentFunction = library.makeFunction(name: "arMeshSolidFragment") {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            do {
                arMeshSolidPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                let depthDesc = MTLDepthStencilDescriptor()
                depthDesc.isDepthWriteEnabled = true
                depthDesc.depthCompareFunction = .less
                arMeshSolidDepthStencilState = device.makeDepthStencilState(descriptor: depthDesc)
                print("✅ AR Mesh solid pipeline state created successfully")
            } catch {
                print("❌ Failed to create AR mesh solid pipeline state: \\(error)")
            }
        }
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
    
    private func setupGPURaycastPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            print("❌ Failed to create default library for GPU raycast shaders")
            return
        }
        
        do {
            // GPU Raycast Pipeline
            if let raycastFunction = library.makeFunction(name: "performGPURaycast") {
                gpuRaycastPipelineState = try device.makeComputePipelineState(function: raycastFunction)
                print("✅ GPU Raycast pipeline state created successfully")
            }
            
            // Mesh Extraction Pipeline
            if let extractFunction = library.makeFunction(name: "extractMeshesInBoundingBox") {
                meshExtractionPipelineState = try device.makeComputePipelineState(function: extractFunction)
                print("✅ Mesh extraction pipeline state created successfully")
            }
        } catch {
            print("❌ Failed to create GPU raycast pipeline states: \(error)")
        }
        
        // Create result buffer (reused across raycast operations)
        raycastResultBuffer = device.makeBuffer(length: MemoryLayout<RaycastResult>.stride, 
                                               options: .storageModeShared)
        raycastResultBuffer?.label = "RaycastResult"
        
        // Create output count buffer for mesh extraction
        outputCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, 
                                             options: .storageModeShared)
        outputCountBuffer?.label = "MeshExtractionOutputCount"
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
        arSession?.delegate = self
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
    
    // Provide AR camera matrices for rendering alignment
    #if canImport(ARKit)
    @available(iOS 11.0, macOS 10.13, *)
    func getCameraMatrices(viewportSize: CGSize,
                           orientation: UIInterfaceOrientation,
                           zNear: Float = 0.001,
                           zFar: Float = 1000.0) -> (projection: float4x4, view: float4x4)? {
        guard let frame = arSession?.currentFrame else { return nil }
        let cam = frame.camera
        let proj = cam.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: CGFloat(zNear), zFar: CGFloat(zFar))
        let view = cam.viewMatrix(for: orientation)
        return (proj, view)
    }
    #endif
    
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
    func updateCollision(force: Bool = false){
        if needUpdate || force {
            // SDF update logic will be added later
        }
    }
    
    var needUpdate: Bool = false
    @available(iOS 13.4, macOS 10.15.4, *)
    func updateMeshGeometry(from meshAnchors: [ARMeshAnchor]) {
        // Update current mesh anchors for SDF generation
        currentMeshAnchors = meshAnchors
        needUpdate = true
        rebuildARMeshWireBuffers()
        rebuildARMeshSolidBuffers()
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
}

// MARK: - AR Mesh Wireframe
#if canImport(ARKit)
@available(iOS 13.4, macOS 10.15.4, *)
private extension ARRenderer {
    func rebuildARMeshWireBuffers() {
        guard !currentMeshAnchors.isEmpty else {
            arMeshVertexBuffer = nil
            arMeshLineIndexBuffer = nil
            arMeshLineIndexCount = 0
            return
        }

        var positions: [SIMD3<Float>] = []
        var lineIndices: [UInt32] = []
        var base: UInt32 = 0

        for anchor in currentMeshAnchors {
            let geom = anchor.geometry
            let vCount = geom.vertices.count
            let vStride = geom.vertices.stride
            let vBuf = geom.vertices.buffer
            // Append transformed positions
            positions.reserveCapacity(positions.count + vCount)
            for i in 0..<vCount {
                let ptr = vBuf.contents().advanced(by: i * vStride).assumingMemoryBound(to: Float.self)
                let local = SIMD3<Float>(ptr[0], ptr[1], ptr[2])
                let world = anchor.transform.transformPoint(local)
                positions.append(world)
            }

            // Build line indices from triangle faces
            let fCount = geom.faces.count
            let fBuf = geom.faces.buffer
            let fStride = MemoryLayout<UInt32>.stride * 3
            for fi in 0..<fCount {
                let fptr = fBuf.contents().advanced(by: fi * fStride).assumingMemoryBound(to: UInt32.self)
                let i0 = base + fptr[0]
                let i1 = base + fptr[1]
                let i2 = base + fptr[2]
                lineIndices.append(contentsOf: [i0, i1, i1, i2, i2, i0])
            }
            base += UInt32(vCount)
        }

        // Create/Update Metal buffers
        if !positions.isEmpty {
            arMeshVertexBuffer = device.makeBuffer(bytes: positions,
                                                   length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
                                                   options: .storageModeShared)
        } else {
            arMeshVertexBuffer = nil
        }
        if !lineIndices.isEmpty {
            arMeshLineIndexBuffer = device.makeBuffer(bytes: lineIndices,
                                                      length: lineIndices.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared)
            arMeshLineIndexCount = lineIndices.count
        } else {
            arMeshLineIndexBuffer = nil
            arMeshLineIndexCount = 0
        }
    }
    
    func rebuildARMeshSolidBuffers() {
        guard !currentMeshAnchors.isEmpty else {
            arMeshSolidVertexBuffer = nil
            arMeshSolidNormalBuffer = nil
            arMeshSolidIndexBuffer = nil
            arMeshSolidIndexCount = 0
            return
        }

        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var triangleIndices: [UInt32] = []
        var base: UInt32 = 0

        for anchor in currentMeshAnchors {
            let geom = anchor.geometry
            let vCount = geom.vertices.count
            let vStride = geom.vertices.stride
            let vBuf = geom.vertices.buffer
            let nBuf = geom.normals.buffer
            let nStride = geom.normals.stride
            
            // Append transformed positions and normals
            positions.reserveCapacity(positions.count + vCount)
            normals.reserveCapacity(normals.count + vCount)
            
            for i in 0..<vCount {
                // Get vertex position
                let vPtr = vBuf.contents().advanced(by: i * vStride).assumingMemoryBound(to: Float.self)
                let localPos = SIMD3<Float>(vPtr[0], vPtr[1], vPtr[2])
                let worldPos = anchor.transform.transformPoint(localPos)
                positions.append(worldPos)
                
                // Get vertex normal
                let nPtr = nBuf.contents().advanced(by: i * nStride).assumingMemoryBound(to: Float.self)
                let localNormal = SIMD3<Float>(nPtr[0], nPtr[1], nPtr[2])
                // Transform normal using upper-left 3x3 matrix (no translation)
                let rotationMatrix = float3x3(
                    SIMD3<Float>(anchor.transform[0].x, anchor.transform[0].y, anchor.transform[0].z),
                    SIMD3<Float>(anchor.transform[1].x, anchor.transform[1].y, anchor.transform[1].z),
                    SIMD3<Float>(anchor.transform[2].x, anchor.transform[2].y, anchor.transform[2].z)
                )
                let worldNormal = rotationMatrix * localNormal
                normals.append(simd_normalize(worldNormal))
            }

            // Build triangle indices
            let fCount = geom.faces.count
            let fBuf = geom.faces.buffer
            let fStride = MemoryLayout<UInt32>.stride * 3
            for fi in 0..<fCount {
                let fptr = fBuf.contents().advanced(by: fi * fStride).assumingMemoryBound(to: UInt32.self)
                let i0 = base + fptr[0]
                let i1 = base + fptr[1]
                let i2 = base + fptr[2]
                triangleIndices.append(contentsOf: [i0, i1, i2])
            }
            base += UInt32(vCount)
        }

        // Create/Update Metal buffers
        if !positions.isEmpty {
            arMeshSolidVertexBuffer = device.makeBuffer(bytes: positions,
                                                       length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
                                                       options: .storageModeShared)
            arMeshSolidNormalBuffer = device.makeBuffer(bytes: normals,
                                                       length: normals.count * MemoryLayout<SIMD3<Float>>.stride,
                                                       options: .storageModeShared)
        } else {
            arMeshSolidVertexBuffer = nil
            arMeshSolidNormalBuffer = nil
        }
        
        if !triangleIndices.isEmpty {
            arMeshSolidIndexBuffer = device.makeBuffer(bytes: triangleIndices,
                                                      length: triangleIndices.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared)
            arMeshSolidIndexCount = triangleIndices.count
        } else {
            arMeshSolidIndexBuffer = nil
            arMeshSolidIndexCount = 0
        }
    }
}
#endif

// MARK: - Raycast and Mesh Extraction
extension ARRenderer {
    #if canImport(ARKit)
    @available(iOS 11.0, macOS 10.13, *)
    public func performGPURaycast(at screenPoint: CGPoint, 
                                 viewportSize: CGSize, 
                                 orientation: UIInterfaceOrientation) -> SIMD3<Float>? {
        guard let frame = arSession?.currentFrame,
              let raycastPipeline = gpuRaycastPipelineState,
              let resultBuffer = raycastResultBuffer else { 
            print("❌ GPU raycast prerequisites not available")
            return nil 
        }
        
        // Convert screen coordinates to normalized device coordinates
        let normalizedX = Float(screenPoint.x / viewportSize.width) * 2.0 - 1.0
        let normalizedY = Float(1.0 - screenPoint.y / viewportSize.height) * 2.0 - 1.0
        
        // Get camera matrices
        let cam = frame.camera
        let projectionMatrix = cam.projectionMatrix(for: orientation, 
                                                   viewportSize: viewportSize, 
                                                   zNear: 0.001, 
                                                   zFar: 1000.0)
        let viewMatrix = cam.viewMatrix(for: orientation)
        
        // Create ray in world space
        let rayOrigin = SIMD3<Float>(viewMatrix.inverse.columns.3.x,
                                    viewMatrix.inverse.columns.3.y,
                                    viewMatrix.inverse.columns.3.z)
        
        // Calculate ray direction from NDC
        let ndcPoint = SIMD4<Float>(normalizedX, normalizedY, -1.0, 1.0)
        let viewSpacePoint = projectionMatrix.inverse * ndcPoint
        let normalizedViewPoint = SIMD3<Float>(viewSpacePoint.x / viewSpacePoint.w,
                                              viewSpacePoint.y / viewSpacePoint.w,
                                              viewSpacePoint.z / viewSpacePoint.w)
        
        let worldSpaceDirection = viewMatrix.inverse.upperLeft3x3 * simd_normalize(normalizedViewPoint)
        
        // Update all triangles buffer if needed
        updateAllTrianglesBuffer()
        
        guard let trianglesBuffer = allTrianglesBuffer else { 
            print("❌ Triangle buffer not available")
            return nil 
        }
        
        // Setup raycast uniforms
        var uniforms = RaycastUniforms(
            rayOrigin: rayOrigin,
            rayDirection: worldSpaceDirection,
            triangleCount: UInt32(getTotalTriangleCount()),
            boundingBoxCenter: SIMD3<Float>(0, 0, 0), // Not used for raycast
            boundingBoxSize: SIMD3<Float>(0, 0, 0)    // Not used for raycast
        )
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("❌ Failed to create command buffer/encoder for GPU raycast")
            return nil
        }
        
        computeEncoder.setComputePipelineState(raycastPipeline)
        computeEncoder.setBytes(&uniforms, length: MemoryLayout<RaycastUniforms>.stride, index: 0)
        computeEncoder.setBuffer(trianglesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        let threadgroupSize = MTLSize(width: min(raycastPipeline.threadExecutionWidth, getTotalTriangleCount()), 
                                     height: 1, depth: 1)
        let threadgroupCount = MTLSize(width: (getTotalTriangleCount() + threadgroupSize.width - 1) / threadgroupSize.width,
                                      height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read result
        let resultPtr = resultBuffer.contents().bindMemory(to: RaycastResult.self, capacity: 1)
        let result = resultPtr.pointee
        
        if result.hit != 0 {
            print("✅ GPU Raycast hit at distance: \(result.distance)")
            return result.hitPoint
        } else {
            print("❌ GPU Raycast: No hit found")
            return nil
        }
    }
    
    // Helper methods for GPU raycast
    private func updateAllTrianglesBuffer() {
        guard !currentMeshAnchors.isEmpty else { return }
        
        var allTriangles: [Triangle] = []
        
        for anchor in currentMeshAnchors {
            let geometry = anchor.geometry
            let vertices = geometry.vertices
            let faces = geometry.faces
            
            let vertexCount = vertices.count
            let faceCount = faces.count
            
            for faceIndex in 0..<faceCount {
                let facePtr = faces.buffer.contents()
                    .advanced(by: faceIndex * MemoryLayout<UInt32>.stride * 3)
                    .assumingMemoryBound(to: UInt32.self)
                
                let i0 = Int(facePtr[0])
                let i1 = Int(facePtr[1])
                let i2 = Int(facePtr[2])
                
                guard i0 < vertexCount && i1 < vertexCount && i2 < vertexCount else { continue }
                
                // Get triangle vertices in local space
                let v0Ptr = vertices.buffer.contents()
                    .advanced(by: i0 * vertices.stride)
                    .assumingMemoryBound(to: Float.self)
                let v1Ptr = vertices.buffer.contents()
                    .advanced(by: i1 * vertices.stride)
                    .assumingMemoryBound(to: Float.self)
                let v2Ptr = vertices.buffer.contents()
                    .advanced(by: i2 * vertices.stride)
                    .assumingMemoryBound(to: Float.self)
                
                let localV0 = SIMD3<Float>(v0Ptr[0], v0Ptr[1], v0Ptr[2])
                let localV1 = SIMD3<Float>(v1Ptr[0], v1Ptr[1], v1Ptr[2])
                let localV2 = SIMD3<Float>(v2Ptr[0], v2Ptr[1], v2Ptr[2])
                
                // Transform to world space
                let worldV0 = anchor.transform.transformPoint(localV0)
                let worldV1 = anchor.transform.transformPoint(localV1)
                let worldV2 = anchor.transform.transformPoint(localV2)
                
                allTriangles.append(Triangle(v0: worldV0, v1: worldV1, v2: worldV2))
            }
        }
        
        guard !allTriangles.isEmpty else { return }
        
        // Create or update triangles buffer
        let bufferSize = allTriangles.count * MemoryLayout<Triangle>.stride
        if allTrianglesBuffer?.length != bufferSize {
            allTrianglesBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
            allTrianglesBuffer?.label = "AllARTriangles"
        }
        
        if let buffer = allTrianglesBuffer {
            let bufferPtr = buffer.contents().bindMemory(to: Triangle.self, capacity: allTriangles.count)
            for (index, triangle) in allTriangles.enumerated() {
                bufferPtr[index] = triangle
            }
        }
    }
    
    private func getTotalTriangleCount() -> Int {
        var totalTriangles = 0
        for anchor in currentMeshAnchors {
            totalTriangles += anchor.geometry.faces.count
        }
        return totalTriangles
    }
    
    
    
    public func extractMeshesInBoundingBoxGPU(center: SIMD3<Float>, 
                                             size: SIMD3<Float>) -> [Triangle]? {
        guard let extractionPipeline = meshExtractionPipelineState,
              let trianglesBuffer = allTrianglesBuffer,
              let countBuffer = outputCountBuffer else {
            print("❌ GPU mesh extraction prerequisites not available")
            return nil
        }
        
        let totalTriangleCount = getTotalTriangleCount()
        guard totalTriangleCount > 0 else { return [] }
        
        // Create output buffer for extracted triangles (worst case: all triangles)
        let maxOutputSize = totalTriangleCount * MemoryLayout<Triangle>.stride
        if extractedTrianglesBuffer?.length != maxOutputSize {
            extractedTrianglesBuffer = device.makeBuffer(length: maxOutputSize, options: .storageModeShared)
            extractedTrianglesBuffer?.label = "ExtractedTriangles"
        }
        
        guard let outputBuffer = extractedTrianglesBuffer else { return nil }
        
        // Reset output count
        let countPtr = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        countPtr.pointee = 0
        
        // Setup extraction uniforms
        var uniforms = RaycastUniforms(
            rayOrigin: SIMD3<Float>(0, 0, 0),  // Not used for extraction
            rayDirection: SIMD3<Float>(0, 0, 0), // Not used for extraction
            triangleCount: UInt32(totalTriangleCount),
            boundingBoxCenter: center,
            boundingBoxSize: size
        )
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("❌ Failed to create command buffer/encoder for GPU mesh extraction")
            return nil
        }
        
        computeEncoder.setComputePipelineState(extractionPipeline)
        computeEncoder.setBytes(&uniforms, length: MemoryLayout<RaycastUniforms>.stride, index: 0)
        computeEncoder.setBuffer(trianglesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(countBuffer, offset: 0, index: 3)
        
        let threadgroupSize = MTLSize(width: min(extractionPipeline.threadExecutionWidth, totalTriangleCount), 
                                     height: 1, depth: 1)
        let threadgroupCount = MTLSize(width: (totalTriangleCount + threadgroupSize.width - 1) / threadgroupSize.width,
                                      height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let outputCount = Int(countPtr.pointee)
        guard outputCount > 0 else { 
            print("❌ GPU mesh extraction found 0 triangles")
            return [] 
        }
        
        let trianglePtr = outputBuffer.contents().bindMemory(to: Triangle.self, capacity: outputCount)
        var extractedTriangles: [Triangle] = []
        for i in 0..<outputCount {
            extractedTriangles.append(trianglePtr[i])
        }
        
        print("✅ GPU mesh extraction found \(outputCount) triangles")
        return extractedTriangles
    }
    
    
    public func generateSDFFromTapPositionGPU(tapPoint: CGPoint,
                                             viewportSize: CGSize,
                                             orientation: UIInterfaceOrientation,
                                             boundingBoxSize: Float = 0.5) -> MTLTexture? {
        // Use GPU raycast
        guard let hitPosition = performGPURaycast(at: tapPoint,
                                                 viewportSize: viewportSize,
                                                 orientation: orientation) else {
            print("No mesh hit at tap position with GPU raycast")
            return nil
        }
        
        lastTapWorldPosition = hitPosition
        
        // Set up bounding box around hit position
        let size = SIMD3<Float>(boundingBoxSize, boundingBoxSize, boundingBoxSize)
        selectedMeshBounds = (center: hitPosition, size: size)
        
        // Extract triangles within bounding box using GPU
        guard let triangles = extractMeshesInBoundingBoxGPU(center: hitPosition, size: size) else {
            print("Failed to extract triangles with GPU")
            return nil
        }
        
        guard !triangles.isEmpty else {
            print("No triangles found in bounding box")
            return nil
        }
        
        print("GPU extracted \(triangles.count) triangles for SDF generation")
        
        // Calculate tight bounding box for SDF generation
        let halfSize = size * 0.5
        let boundingBox = (min: hitPosition - halfSize, max: hitPosition + halfSize)
        
        // Generate SDF
        let resolution = SIMD3<Int32>(64, 64, 64) // Fixed resolution for consistency
        return sdfGenerator?.generateSDF(triangles: triangles,
                                        resolution: resolution,
                                        boundingBox: boundingBox)
    }
    #endif
}

extension ARRenderer {
    // Draw AR mesh wireframe as overlay within an existing encoder
    func renderARMeshWireframeInEncoder(renderEncoder: MTLRenderCommandEncoder,
                                        viewportSize: CGSize,
                                        orientation: UIInterfaceOrientation,
                                        color: SIMD4<Float> = SIMD4<Float>(0.0, 1.0, 1.0, 1.0)) {
        #if canImport(ARKit)
        guard showARMeshWireframe,
              let pipeline = arMeshWirePipelineState,
              let depthState = arMeshWireDepthStencilState,
              let vbuf = arMeshVertexBuffer,
              let ibuf = arMeshLineIndexBuffer,
              arMeshLineIndexCount > 0 else { return }
        if #available(iOS 11.0, macOS 10.13, *) {
            guard let (proj, view) = getCameraMatrices(viewportSize: viewportSize, orientation: orientation) else { return }
            var projection = proj
            var viewM = view
            var wireColor = color

            renderEncoder.setRenderPipelineState(pipeline)
            renderEncoder.setDepthStencilState(depthState)
            renderEncoder.setVertexBuffer(vbuf, offset: 0, index: 0)
            renderEncoder.setVertexBytes(&projection, length: MemoryLayout<float4x4>.stride, index: 1)
            renderEncoder.setVertexBytes(&viewM, length: MemoryLayout<float4x4>.stride, index: 2)
            renderEncoder.setFragmentBytes(&wireColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            renderEncoder.drawIndexedPrimitives(type: .line,
                                               indexCount: arMeshLineIndexCount,
                                               indexType: .uint32,
                                               indexBuffer: ibuf,
                                               indexBufferOffset: 0)
        }
        #endif
    }
    
    // Draw AR mesh solid as background within an existing encoder
    func renderARMeshSolidInEncoder(renderEncoder: MTLRenderCommandEncoder,
                                   viewportSize: CGSize,
                                   orientation: UIInterfaceOrientation,
                                   color: SIMD4<Float> = SIMD4<Float>(0.8, 0.8, 0.8, 0.8),
                                   lightDirection: SIMD3<Float> = SIMD3<Float>(0.0, -1.0, 0.0)) {
        #if canImport(ARKit)
        guard showARMeshSolid,
              let pipeline = arMeshSolidPipelineState,
              let depthState = arMeshSolidDepthStencilState,
              let vbuf = arMeshSolidVertexBuffer,
              let nbuf = arMeshSolidNormalBuffer,
              let ibuf = arMeshSolidIndexBuffer,
              arMeshSolidIndexCount > 0 else { return }
        
        if #available(iOS 11.0, macOS 10.13, *) {
            guard let (proj, view) = getCameraMatrices(viewportSize: viewportSize, orientation: orientation) else { return }
            var projection = proj
            var viewM = view
            var meshColor = color
            var lightDir = lightDirection

            renderEncoder.setRenderPipelineState(pipeline)
            renderEncoder.setDepthStencilState(depthState)
            renderEncoder.setVertexBuffer(vbuf, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(nbuf, offset: 0, index: 1)
            renderEncoder.setVertexBytes(&projection, length: MemoryLayout<float4x4>.stride, index: 2)
            renderEncoder.setVertexBytes(&viewM, length: MemoryLayout<float4x4>.stride, index: 3)
            renderEncoder.setFragmentBytes(&meshColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            renderEncoder.setFragmentBytes(&lightDir, length: MemoryLayout<SIMD3<Float>>.stride, index: 1)
            renderEncoder.drawIndexedPrimitives(type: .triangle,
                                               indexCount: arMeshSolidIndexCount,
                                               indexType: .uint32,
                                               indexBuffer: ibuf,
                                               indexBufferOffset: 0)
        }
        #endif
    }
}

#if canImport(ARKit)
@available(iOS 11.0, macOS 10.13, *)
extension ARRenderer: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateCameraTextures(from: frame)
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        if #available(iOS 13.4, *) {
            let meshAnchors = anchors.compactMap { $0 as? ARMeshAnchor }
            if !meshAnchors.isEmpty {
                updateMeshGeometry(from: meshAnchors)
            }
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        if #available(iOS 13.4, *) {
            let meshAnchors = anchors.compactMap { $0 as? ARMeshAnchor }
            if !meshAnchors.isEmpty {
                updateMeshGeometry(from: meshAnchors)
            }
        }
    }
}
#endif
