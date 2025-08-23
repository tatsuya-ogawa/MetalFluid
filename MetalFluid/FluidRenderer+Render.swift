import Metal
import MetalKit
import simd

// MARK: - Render Extension
extension MPMFluidRenderer {
    
    // MARK: - Setup Functions
    
    internal func setupRenderPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }
                
        // Setup pressure heatmap pipeline
        guard let pressureVertexFunction = library.makeFunction(name: "pressureHeatmapVertexShader") else {
            fatalError("Could not find pressure heatmap vertex shader function")
        }
        guard let fragmentFunction = library.makeFunction(name: "pressureHeatmapFragmentShader") else {
            fatalError("Could not find shader functions")
        }
        
        let pressurePipelineDescriptor = MTLRenderPipelineDescriptor()
        pressurePipelineDescriptor.vertexFunction = pressureVertexFunction
        pressurePipelineDescriptor.fragmentFunction = fragmentFunction // Reuse same fragment shader
        pressurePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pressurePipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        // Enable blending for smooth particles
        pressurePipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pressurePipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pressurePipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pressurePipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pressurePipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pressurePipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pressurePipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        do {
            pressureHeatmapPipelineState = try device.makeRenderPipelineState(
                descriptor: pressurePipelineDescriptor
            )
        } catch {
            fatalError("Could not create pressure heatmap render pipeline state: \(error)")
        }
        
        // Create depth stencil state for proper depth testing
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .lessEqual  // More lenient than .less
        depthStencilDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
    }

    internal func setupDepthPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }
        
        // Depth rendering pipeline
        guard let depthVertexFunction = library.makeFunction(name: "vs_depth"),
              let depthFragmentFunction = library.makeFunction(name: "fs_depth")
        else {
            fatalError("Could not find depth shader functions")
        }
        
        let depthPipelineDescriptor = MTLRenderPipelineDescriptor()
        depthPipelineDescriptor.vertexFunction = depthVertexFunction
        depthPipelineDescriptor.fragmentFunction = depthFragmentFunction
        depthPipelineDescriptor.colorAttachments[0].pixelFormat = .r32Float
        depthPipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        do {
            depthRenderPipelineState = try device.makeRenderPipelineState(
                descriptor: depthPipelineDescriptor
            )
        } catch {
            fatalError("Could not create depth render pipeline state: \(error)")
        }
        
        // Depth filter pipeline
        guard let depthFilterVertexFunction = library.makeFunction(name: "vs_bilateral"),
              let filterFragmentFunction = library.makeFunction(name: "fs_bilateral")
        else {
            fatalError("Could not find depth filter shader functions")
        }
        
        let depthFilterPipelineDescriptor = MTLRenderPipelineDescriptor()
        depthFilterPipelineDescriptor.vertexFunction = depthFilterVertexFunction
        depthFilterPipelineDescriptor.fragmentFunction = filterFragmentFunction
        depthFilterPipelineDescriptor.colorAttachments[0].pixelFormat = .r32Float
        
        do {
            depthFilterPipelineState = try device.makeRenderPipelineState(
                descriptor: depthFilterPipelineDescriptor
            )
        } catch {
            fatalError("Could not create depth filter pipeline state: \(error)")
        }
    }

    internal func setupThicknessPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }
        
        // Thickness rendering pipeline
        guard let thicknessVertexFunction = library.makeFunction(name: "vs_thickness"),
              let thicknessFragmentFunction = library.makeFunction(name: "fs_thickness")
        else {
            fatalError("Could not find thickness shader functions")
        }
        
        let thicknessPipelineDescriptor = MTLRenderPipelineDescriptor()
        thicknessPipelineDescriptor.vertexFunction = thicknessVertexFunction
        thicknessPipelineDescriptor.fragmentFunction = thicknessFragmentFunction
        thicknessPipelineDescriptor.colorAttachments[0].pixelFormat = .r16Float
        thicknessPipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        // Enable additive blending for thickness accumulation
        thicknessPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        thicknessPipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        thicknessPipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        thicknessPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        thicknessPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        thicknessPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .one
        thicknessPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .one
        
        do {
            thicknessRenderPipelineState = try device.makeRenderPipelineState(
                descriptor: thicknessPipelineDescriptor
            )
        } catch {
            fatalError("Could not create thickness render pipeline state: \(error)")
        }
        
        // Gaussian filter pipeline
        guard let gaussianVertexFunction = library.makeFunction(name: "vs_bilateral"),
              let gaussianFragmentFunction = library.makeFunction(name: "fs_gaussian")
        else {
            fatalError("Could not find Gaussian filter shader functions")
        }
        
        let gaussianPipelineDescriptor = MTLRenderPipelineDescriptor()
        gaussianPipelineDescriptor.vertexFunction = gaussianVertexFunction
        gaussianPipelineDescriptor.fragmentFunction = gaussianFragmentFunction
        gaussianPipelineDescriptor.colorAttachments[0].pixelFormat = .r16Float
        // Note: Gaussian filter doesn't need depth attachment since it's a full-screen quad
        
        do {
            gaussianFilterPipelineState = try device.makeRenderPipelineState(
                descriptor: gaussianPipelineDescriptor
            )
        } catch {
            fatalError("Could not create Gaussian filter pipeline state: \(error)")
        }
    }

    internal func setupFluidSurfacePipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default library")
        }
        
        guard let vertexFunction = library.makeFunction(name: "fluidVertexShader"),
              let fragmentFunction = library.makeFunction(name: "fluidFragmentShader")
        else {
            fatalError("Could not find fluid surface shader functions")
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        // Enable blending for fluid surface
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        do {
            fluidSurfacePipelineState = try device.makeRenderPipelineState(
                descriptor: pipelineDescriptor
            )
        } catch {
            fatalError("Could not create fluid surface pipeline state: \(error)")
        }
    }

    internal func setupDepthTextures(screenSize: SIMD2<Float>) {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: Int(screenSize.x),
            height: Int(screenSize.y),
            mipmapped: false
        )
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        
        // Create depth textures
        depthTexture = device.makeTexture(descriptor: textureDescriptor)!
        tempDepthTexture = device.makeTexture(descriptor: textureDescriptor)!
        filteredDepthTexture = device.makeTexture(descriptor: textureDescriptor)!
        
        self.screenSize = screenSize
    }
    
    internal func setupFluidTextures(screenSize: SIMD2<Float>) {
        // Thickness textures
        let thicknessDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: Int(screenSize.x),
            height: Int(screenSize.y),
            mipmapped: false
        )
        thicknessDescriptor.usage = [.renderTarget, .shaderRead]
        thicknessTexture = device.makeTexture(descriptor: thicknessDescriptor)!
        tempThicknessTexture = device.makeTexture(descriptor: thicknessDescriptor)!
        filteredThicknessTexture = device.makeTexture(descriptor: thicknessDescriptor)!
        
        // Create a simple cube environment texture
        let envDescriptor = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba8Unorm,
            size: 64,
            mipmapped: false
        )
        envDescriptor.usage = .shaderRead
        environmentTexture = device.makeTexture(descriptor: envDescriptor)!
        
        // Fill environment texture with a simple sky color
        let envRegion = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: 64, height: 64, depth: 1))
        let skyData = Array(repeating: UInt8(200), count: 64 * 64 * 4) // Light blue-ish
        for face in 0..<6 {
            environmentTexture.replace(region: envRegion, mipmapLevel: 0, slice: face, withBytes: skyData, bytesPerRow: 64 * 4, bytesPerImage: 64 * 64 * 4)
        }
    }
    
    public func updateScreenSize(_ newSize: SIMD2<Float>) {
        if newSize.x != screenSize.x || newSize.y != screenSize.y {
            setupDepthTextures(screenSize: newSize)
            setupFluidTextures(screenSize: newSize)
        }
    }
        
    // MARK: - Main Render Function
    
    func render(
        renderPassDescriptor: MTLRenderPassDescriptor,
        performCompute: Bool,
        projectionMatrix: float4x4,
        viewMatrix: float4x4
    ) {
        switch currentRenderMode {
        case .particles:
            particleRenderer.render(
                renderPassDescriptor: renderPassDescriptor,
                performCompute: performCompute,
                projectionMatrix: projectionMatrix,
                viewMatrix: viewMatrix
            )
        case .water:
            waterRenderer.render(
                renderPassDescriptor: renderPassDescriptor,
                performCompute: performCompute,
                projectionMatrix: projectionMatrix,
                viewMatrix: viewMatrix
            )
        }
    }
    
    // MARK: - Water Rendering Pipeline
    
    internal func renderDepthMap(commandBuffer: MTLCommandBuffer) {
        let depthPassDescriptor = MTLRenderPassDescriptor()
        depthPassDescriptor.colorAttachments[0].texture = depthTexture
        depthPassDescriptor.colorAttachments[0].loadAction = .clear
        depthPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        depthPassDescriptor.colorAttachments[0].storeAction = .store
        
        // Create depth buffer for depth testing
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: depthTexture.width,
            height: depthTexture.height,
            mipmapped: false
        )
        depthTextureDescriptor.usage = .renderTarget
        let depthBuffer = device.makeTexture(descriptor: depthTextureDescriptor)!
        
        depthPassDescriptor.depthAttachment.texture = depthBuffer
        depthPassDescriptor.depthAttachment.loadAction = .clear
        depthPassDescriptor.depthAttachment.clearDepth = 1.0
        depthPassDescriptor.depthAttachment.storeAction = .dontCare
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: depthPassDescriptor) else {
            return
        }
        
        renderEncoder.setRenderPipelineState(depthRenderPipelineState)
        renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(vertexUniformBuffer, offset: 0, index: 1)
        renderEncoder.drawPrimitives(
            type: .point,
            vertexStart: 0,
            vertexCount: particleCount
        )
        renderEncoder.endEncoding()
    }
    
    internal func applyDepthFilter(
        commandBuffer: MTLCommandBuffer,
        depthThreshold: Float,
        filterRadius: Int
    ) {
        // Update filter uniforms
        let filterPointer = filterUniformBuffer.contents().bindMemory(
            to: FilterUniforms.self,
            capacity: 1
        )
        
        // Calculate parameters like WebGPU
        let radius: Float = 0.1 // particle radius
        let blurdDepthScale: Float = 10.0
        let maxFilterSize: Float = 100.0
        let blurFilterSize: Float = 12.0
        let diameter = 2.0 * radius
        let fov: Float = 60.0 * Float.pi / 180.0 // Field of view in radians
        
        let calculatedDepthThreshold = radius * blurdDepthScale
        let projectedParticleConstant = (blurFilterSize * diameter * 0.05 * (screenSize.y / 2.0)) / tan(fov / 2.0)
        
        // Horizontal pass
        filterPointer[0] = FilterUniforms(
            direction: SIMD2<Float>(1.0, 0.0),
            screenSize: screenSize,
            depthThreshold: calculatedDepthThreshold,
            filterRadius: Int32(filterRadius),
            projectedParticleConstant: projectedParticleConstant,
            maxFilterSize: maxFilterSize
        )
        
        let horizontalPassDescriptor = MTLRenderPassDescriptor()
        horizontalPassDescriptor.colorAttachments[0].texture = tempDepthTexture
        horizontalPassDescriptor.colorAttachments[0].loadAction = .clear
        horizontalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        horizontalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: horizontalPassDescriptor) {
            renderEncoder.setRenderPipelineState(depthFilterPipelineState)
            renderEncoder.setFragmentTexture(depthTexture, index: 0)
            renderEncoder.setFragmentBuffer(filterUniformBuffer, offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
        
        // Vertical pass
        filterPointer[0] = FilterUniforms(
            direction: SIMD2<Float>(0.0, 1.0),
            screenSize: screenSize,
            depthThreshold: calculatedDepthThreshold,
            filterRadius: Int32(filterRadius),
            projectedParticleConstant: projectedParticleConstant,
            maxFilterSize: maxFilterSize
        )
        
        let verticalPassDescriptor = MTLRenderPassDescriptor()
        verticalPassDescriptor.colorAttachments[0].texture = depthTexture
        verticalPassDescriptor.colorAttachments[0].loadAction = .clear
        verticalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        verticalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: verticalPassDescriptor) {
            renderEncoder.setRenderPipelineState(depthFilterPipelineState)
            renderEncoder.setFragmentTexture(tempDepthTexture, index: 0)
            renderEncoder.setFragmentBuffer(filterUniformBuffer, offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
    }
    
    internal func renderThicknessMap(commandBuffer: MTLCommandBuffer) {
        let thicknessPassDescriptor = MTLRenderPassDescriptor()
        thicknessPassDescriptor.colorAttachments[0].texture = thicknessTexture
        thicknessPassDescriptor.colorAttachments[0].loadAction = .clear
        thicknessPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        thicknessPassDescriptor.colorAttachments[0].storeAction = .store
        
        // Create depth buffer for depth testing
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: thicknessTexture.width,
            height: thicknessTexture.height,
            mipmapped: false
        )
        depthTextureDescriptor.usage = .renderTarget
        let depthBuffer = device.makeTexture(descriptor: depthTextureDescriptor)!
        
        thicknessPassDescriptor.depthAttachment.texture = depthBuffer
        thicknessPassDescriptor.depthAttachment.loadAction = .clear
        thicknessPassDescriptor.depthAttachment.clearDepth = 1.0
        thicknessPassDescriptor.depthAttachment.storeAction = .dontCare
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: thicknessPassDescriptor) else {
            return
        }
        
        renderEncoder.setRenderPipelineState(thicknessRenderPipelineState)
        renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(vertexUniformBuffer, offset: 0, index: 1)
        renderEncoder.drawPrimitives(
            type: .point,
            vertexStart: 0,
            vertexCount: particleCount
        )
        renderEncoder.endEncoding()
    }
    
    internal func applyThicknessFilter(commandBuffer: MTLCommandBuffer, filterRadius: Int = 4) {
        // Update Gaussian uniforms
        let gaussianPointer = gaussianUniformBuffer.contents().bindMemory(
            to: GaussianUniforms.self,
            capacity: 1
        )
        
        // Horizontal pass
        gaussianPointer[0] = GaussianUniforms(
            direction: SIMD2<Float>(1.0, 0.0),
            screenSize: screenSize,
            filterRadius: Int32(filterRadius)
        )
        
        let horizontalPassDescriptor = MTLRenderPassDescriptor()
        horizontalPassDescriptor.colorAttachments[0].texture = tempThicknessTexture
        horizontalPassDescriptor.colorAttachments[0].loadAction = .clear
        horizontalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        horizontalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: horizontalPassDescriptor) {
            renderEncoder.setRenderPipelineState(gaussianFilterPipelineState)
            renderEncoder.setFragmentTexture(thicknessTexture, index: 0)
            renderEncoder.setFragmentBuffer(gaussianUniformBuffer, offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
        
        // Vertical pass
        gaussianPointer[0] = GaussianUniforms(
            direction: SIMD2<Float>(0.0, 1.0),
            screenSize: screenSize,
            filterRadius: Int32(filterRadius)
        )
        
        let verticalPassDescriptor = MTLRenderPassDescriptor()
        verticalPassDescriptor.colorAttachments[0].texture = filteredThicknessTexture
        verticalPassDescriptor.colorAttachments[0].loadAction = .clear
        verticalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        verticalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: verticalPassDescriptor) {
            renderEncoder.setRenderPipelineState(gaussianFilterPipelineState)
            renderEncoder.setFragmentTexture(tempThicknessTexture, index: 0)
            renderEncoder.setFragmentBuffer(gaussianUniformBuffer, offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
    }
    
    // MARK: - Public API
    
    public func setRenderMode(_ mode: RenderMode) {
        currentRenderMode = mode
        print("🎨 Render mode switched to: \(mode)")
    }
    
    public func toggleRenderMode() {
        switch currentRenderMode {
        case .particles:
            setRenderMode(.water)
        case .water:
            setRenderMode(.particles)
        }
    }
}
