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
        
        // Create depth stencil state for fluid rendering
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less
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
    
    internal func createTexturesForSize(_ size: SIMD2<Float>) -> FluidRenderTextures {
        // Ensure even dimensions to prevent crashes with certain GPU operations
        #if targetEnvironment(macCatalyst) || os(macOS)
        let maxTextureSize = Int(max(size.x,size.y))
        #else
        let maxTextureForMobile = 1024
        // for prevent ipad hangup
        let maxTextureSize = maxTextureForMobile
        #endif
        let adjustedWidth = min(Int(size.x), maxTextureSize)
        let adjustedHeight = min(Int(size.y), maxTextureSize)
        
        // Create depth textures
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: adjustedWidth,
            height: adjustedHeight,
            mipmapped: false
        )
        depthTextureDescriptor.usage = [.renderTarget, .shaderRead]
        depthTextureDescriptor.storageMode = .private  // Use private storage for GPU performance
        
        let newDepthTexture = device.makeTexture(descriptor: depthTextureDescriptor)!
        newDepthTexture.label = "FluidDepthTexture"
        let newTempDepthTexture = device.makeTexture(descriptor: depthTextureDescriptor)!
        newTempDepthTexture.label = "FluidTempDepthTexture"
        let newFilteredDepthTexture = device.makeTexture(descriptor: depthTextureDescriptor)!
        newFilteredDepthTexture.label = "FluidFilteredDepthTexture"
        
        // Create thickness textures
        let thicknessDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: adjustedWidth,
            height: adjustedHeight,
            mipmapped: false
        )
        thicknessDescriptor.usage = [.renderTarget, .shaderRead]
        thicknessDescriptor.storageMode = .private  // Use private storage for GPU performance
        let newThicknessTexture = device.makeTexture(descriptor: thicknessDescriptor)!
        newThicknessTexture.label = "FluidThicknessTexture"
        let newTempThicknessTexture = device.makeTexture(descriptor: thicknessDescriptor)!
        newTempThicknessTexture.label = "FluidTempThicknessTexture"
        let newFilteredThicknessTexture = device.makeTexture(descriptor: thicknessDescriptor)!
        newFilteredThicknessTexture.label = "FluidFilteredThicknessTexture"
        
        // Create environment texture (size-independent, but included for completeness)
        let envDescriptor = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba8Unorm,
            size: 64,
            mipmapped: false
        )
        envDescriptor.usage = .shaderRead
        let newEnvironmentTexture = device.makeTexture(descriptor: envDescriptor)!
        
        // Fill environment texture with a simple sky color
        let envRegion = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: 64, height: 64, depth: 1))
        let skyData = Array(repeating: UInt8(200), count: 64 * 64 * 4) // Light blue-ish
        for face in 0..<6 {
            newEnvironmentTexture.replace(region: envRegion, mipmapLevel: 0, slice: face, withBytes: skyData, bytesPerRow: 64 * 4, bytesPerImage: 64 * 64 * 4)
        }
        
        // Create offscreen color texture (same size as screen)
        let offscreenColorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: adjustedWidth,
            height: adjustedHeight,
            mipmapped: false
        )
        offscreenColorDescriptor.usage = [.renderTarget, .shaderRead]
        offscreenColorDescriptor.storageMode = .private
        let newOffscreenColorTexture = device.makeTexture(descriptor: offscreenColorDescriptor)!
        newOffscreenColorTexture.label = "OffscreenColorTexture"
        
        // Create offscreen depth texture
        let offscreenDepthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: adjustedWidth,
            height: adjustedHeight,
            mipmapped: false
        )
        offscreenDepthDescriptor.usage = .renderTarget
        offscreenDepthDescriptor.storageMode = .private
        let newOffscreenDepthTexture = device.makeTexture(descriptor: offscreenDepthDescriptor)!
        newOffscreenDepthTexture.label = "OffscreenDepthTexture"
        
        return FluidRenderTextures(
            depthTexture: newDepthTexture,
            tempDepthTexture: newTempDepthTexture,
            filteredDepthTexture: newFilteredDepthTexture,
            thicknessTexture: newThicknessTexture,
            tempThicknessTexture: newTempThicknessTexture,
            filteredThicknessTexture: newFilteredThicknessTexture,
            environmentTexture: newEnvironmentTexture,
            offscreenColorTexture: newOffscreenColorTexture,
            offscreenDepthTexture: newOffscreenDepthTexture,
            screenSize: SIMD2<Float>(Float(adjustedWidth), Float(adjustedHeight)),
            bufferIndex: 0  // Default buffer index
        )
    }
    
    
    // MARK: - Legacy render function - replaced by IntegratedRenderer
    
    // MARK: - AR Rendering Functions
    
    // Deprecated: AR background rendering handled by BackgroundRenderer
    
    // MARK: - Water Rendering Pipeline
    
    internal func renderDepthMap(commandBuffer: MTLCommandBuffer,particleBuffer:MTLBuffer,textures: FluidRenderTextures) {
        let depthPassDescriptor = MTLRenderPassDescriptor()
        depthPassDescriptor.colorAttachments[0].texture = textures.depthTexture
        depthPassDescriptor.colorAttachments[0].loadAction = .clear
        depthPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        depthPassDescriptor.colorAttachments[0].storeAction = .store
        
        // Create depth buffer for depth testing
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: textures.depthTexture.width,
            height: textures.depthTexture.height,
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
        renderEncoder.setDepthStencilState(depthStencilState)
        renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(scene.getVertexUniformBuffer(), offset: 0, index: 1)
        renderEncoder.setFragmentBuffer(scene.getFluidRenderUniformBuffer(), offset: 0, index: 1)
        
        // Draw billboard quads (6 vertices per instance)
        renderEncoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: 6,
            instanceCount: particleCount
        )
        renderEncoder.endEncoding()
    }
    
    internal func applyDepthFilter(
        commandBuffer: MTLCommandBuffer,
        textures: FluidRenderTextures,
        depthThreshold: Float,
        filterRadius: Int
    ) {
        // Update filter uniforms
        let filterPointer = scene.getFilterUniformBuffer().contents().bindMemory(
            to: FilterUniforms.self,
            capacity: 1
        )
        
        // Calculate parameters like WebGPU - use dynamic particle size
        let baseRadius: Float = 0.1 // base particle radius
        let radius = baseRadius * particleSizeMultiplier // scale with slider
        let blurdDepthScale: Float = 10.0
        let maxFilterSize: Float = 100.0
        let blurFilterSize: Float = 12.0
        let diameter = 2.0 * radius
        let fov: Float = 60.0 * Float.pi / 180.0 // Field of view in radians
        
        let calculatedDepthThreshold = radius * blurdDepthScale
        let projectedParticleConstant = (blurFilterSize * diameter * 0.05 * (textures.screenSize.y / 2.0)) / tan(fov / 2.0)
        
        // Horizontal pass
        filterPointer[0] = FilterUniforms(
            direction: SIMD2<Float>(1.0, 0.0),
            screenSize: textures.screenSize,
            depthThreshold: calculatedDepthThreshold,
            filterRadius: Int32(filterRadius),
            projectedParticleConstant: projectedParticleConstant,
            maxFilterSize: maxFilterSize
        )
        
        let horizontalPassDescriptor = MTLRenderPassDescriptor()
        horizontalPassDescriptor.colorAttachments[0].texture = textures.tempDepthTexture
        horizontalPassDescriptor.colorAttachments[0].loadAction = .clear
        horizontalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        horizontalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: horizontalPassDescriptor) {
            renderEncoder.setRenderPipelineState(depthFilterPipelineState)
            renderEncoder.setFragmentTexture(textures.depthTexture, index: 0)
            renderEncoder.setFragmentBuffer(scene.getFilterUniformBuffer(), offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
        
        // Vertical pass
        filterPointer[0] = FilterUniforms(
            direction: SIMD2<Float>(0.0, 1.0),
            screenSize: textures.screenSize,
            depthThreshold: calculatedDepthThreshold,
            filterRadius: Int32(filterRadius),
            projectedParticleConstant: projectedParticleConstant,
            maxFilterSize: maxFilterSize
        )
        
        let verticalPassDescriptor = MTLRenderPassDescriptor()
        verticalPassDescriptor.colorAttachments[0].texture = textures.depthTexture
        verticalPassDescriptor.colorAttachments[0].loadAction = .clear
        verticalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        verticalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: verticalPassDescriptor) {
            renderEncoder.setRenderPipelineState(depthFilterPipelineState)
            renderEncoder.setFragmentTexture(textures.tempDepthTexture, index: 0)
            renderEncoder.setFragmentBuffer(scene.getFilterUniformBuffer(), offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
    }
    
    internal func renderThicknessMap(commandBuffer: MTLCommandBuffer,particleBuffer:MTLBuffer,textures: FluidRenderTextures) {
        let thicknessPassDescriptor = MTLRenderPassDescriptor()
        thicknessPassDescriptor.colorAttachments[0].texture = textures.thicknessTexture
        thicknessPassDescriptor.colorAttachments[0].loadAction = .clear
        thicknessPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        thicknessPassDescriptor.colorAttachments[0].storeAction = .store
        
        // Create depth buffer for depth testing
        let depthTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: textures.thicknessTexture.width,
            height: textures.thicknessTexture.height,
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
        renderEncoder.setVertexBuffer(scene.getVertexUniformBuffer(), offset: 0, index: 1)
        
        // Draw billboard quads (6 vertices per instance)
        renderEncoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: 6,
            instanceCount: particleCount
        )
        renderEncoder.endEncoding()
    }
    
    internal func applyThicknessFilter(commandBuffer: MTLCommandBuffer, textures: FluidRenderTextures, filterRadius: Int = 4) {
        // Update Gaussian uniforms
        let gaussianPointer = scene.getGaussianUniformBuffer().contents().bindMemory(
            to: GaussianUniforms.self,
            capacity: 1
        )
        
        // Horizontal pass
        gaussianPointer[0] = GaussianUniforms(
            direction: SIMD2<Float>(1.0, 0.0),
            screenSize: textures.screenSize,
            filterRadius: Int32(filterRadius)
        )
        
        let horizontalPassDescriptor = MTLRenderPassDescriptor()
        horizontalPassDescriptor.colorAttachments[0].texture = textures.tempThicknessTexture
        horizontalPassDescriptor.colorAttachments[0].loadAction = .clear
        horizontalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        horizontalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: horizontalPassDescriptor) {
            renderEncoder.setRenderPipelineState(gaussianFilterPipelineState)
            renderEncoder.setFragmentTexture(textures.thicknessTexture, index: 0)
            renderEncoder.setFragmentBuffer(scene.getGaussianUniformBuffer(), offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
        
        // Vertical pass
        gaussianPointer[0] = GaussianUniforms(
            direction: SIMD2<Float>(0.0, 1.0),
            screenSize: textures.screenSize,
            filterRadius: Int32(filterRadius)
        )
        
        let verticalPassDescriptor = MTLRenderPassDescriptor()
        verticalPassDescriptor.colorAttachments[0].texture = textures.filteredThicknessTexture
        verticalPassDescriptor.colorAttachments[0].loadAction = .clear
        verticalPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        verticalPassDescriptor.colorAttachments[0].storeAction = .store
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: verticalPassDescriptor) {
            renderEncoder.setRenderPipelineState(gaussianFilterPipelineState)
            renderEncoder.setFragmentTexture(textures.tempThicknessTexture, index: 0)
            renderEncoder.setFragmentBuffer(scene.getGaussianUniformBuffer(), offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
    }
}
