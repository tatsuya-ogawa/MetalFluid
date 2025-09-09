import Metal
import MetalKit
import simd

enum FluidRenderContext {
    case prepared(FluidRenderTextures, () -> Void)
    case none
}


class IntegratedRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private weak var fluidRenderer: MPMFluidRenderer?
    public weak var overlayRenderer: OverlayRenderer?
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, fluidRenderer: MPMFluidRenderer) {
        self.device = device
        self.commandQueue = commandQueue
        self.fluidRenderer = fluidRenderer
    }
    
    // MARK: - Depth Stencil Descriptor Utilities
    func createDepthStencilDescriptor(
        depthCompareFunction: MTLCompareFunction = .less,
        isDepthWriteEnabled: Bool = true
    ) -> MTLDepthStencilDescriptor {
        let descriptor = MTLDepthStencilDescriptor()
        descriptor.depthCompareFunction = depthCompareFunction
        descriptor.isDepthWriteEnabled = isDepthWriteEnabled
        return descriptor
    }
    
    func createFluidDepthStencilDescriptor() -> MTLDepthStencilDescriptor {
        return createDepthStencilDescriptor(
            depthCompareFunction: .less,
            isDepthWriteEnabled: false  // Fluid doesn't write to depth
        )
    }
    
    private func configureDepth(
        _ renderPassDescriptor: MTLRenderPassDescriptor,
        screenSize: SIMD2<Float>,
        depthLoadAction: MTLLoadAction? = nil
    ) {
        guard let fluidRenderer = fluidRenderer else { return }
        
        if let loadAction = depthLoadAction {
            fluidRenderer.ensureDepthBuffer(for: renderPassDescriptor, screenSize: screenSize, label: "IntegratedRenderDepthBuffer")
            renderPassDescriptor.depthAttachment.loadAction = loadAction
            if loadAction == .clear {
                renderPassDescriptor.depthAttachment.clearDepth = 1.0
            }
        } else {
            renderPassDescriptor.depthAttachment.texture = nil
        }
    }
    
    func render(
        renderPassDescriptor: MTLRenderPassDescriptor,
        performCompute: Bool,
        projectionMatrix: float4x4,
        viewMatrix: float4x4,
        cameraProjectionMatrix: float4x4? = nil,
        cameraViewMatrix: float4x4? = nil
    ) {
        guard let fluidRenderer = fluidRenderer,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let colorTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return
        }
        
        // Calculate screen size once for all render methods
        let screenSize = SIMD2<Float>(Float(colorTexture.width), Float(colorTexture.height))
                
        // 1. Command buffer generation (already done above)
        
        // 2. Perform compute if needed
        if performCompute {
            fluidRenderer.compute(commandBuffer: commandBuffer)
        }
        
        // 3. Prepare fluid offscreen textures (before background to prevent flickering)
        let fluidContext = renderFluidPrepare(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix
        )
        
        // 4. Background render (depth disabled)
        renderBackground(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            depthStencilDescriptor: nil
        )

        // 5. Overlay render (depth disabled)
        renderOverlay(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            depthStencilDescriptor: nil
        )

        // Ensure depth buffer and clear it for collision
        let depthBuffer = fluidRenderer.ensureDepthBuffer(for: renderPassDescriptor, screenSize: screenSize, label: "IntegratedRenderDepthBuffer")
        renderPassDescriptor.depthAttachment.texture = depthBuffer
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .store
        renderPassDescriptor.depthAttachment.clearDepth = 1.0
        // 7. Fluid render (uses custom depthWrite setting from facade or fallback)
        // Set depth buffer to load for fluid rendering
        let fluidDepthStencilDescriptor = createDepthStencilDescriptor(
            depthCompareFunction: .less,
            isDepthWriteEnabled: true
        )
        renderFluid(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            context: fluidContext,
            depthStencilDescriptor: fluidDepthStencilDescriptor
        )
        renderPassDescriptor.depthAttachment.texture = depthBuffer
        renderPassDescriptor.depthAttachment.loadAction = .load
        renderPassDescriptor.depthAttachment.storeAction = .store
        // 6. Collision render (depth clear + write enabled)
        // Create collision depth stencil descriptor (depth clear + write enabled)
        let collisionDepthStencilDescriptor = createDepthStencilDescriptor(
            depthCompareFunction: .less,
            isDepthWriteEnabled: true
        )
        renderCollision(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            cameraProjectionMatrix: cameraProjectionMatrix,
            cameraViewMatrix: cameraViewMatrix,
            depthStencilDescriptor: collisionDepthStencilDescriptor
        )
        // 8. Commit
        commandBuffer.commit()
    }
    
    private func renderBackground(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        depthStencilDescriptor: MTLDepthStencilDescriptor? = nil
    ) {
        guard let fluidRenderer = fluidRenderer,
              let backgroundRenderer = fluidRenderer.backgroundRenderer,
              let colorAttachmentTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return
        }
        
        _ = backgroundRenderer.renderBackground(commandBuffer: commandBuffer, targetTexture: colorAttachmentTexture)
        backgroundRenderer.updateCollisionSDFIfNeeded()
        renderPassDescriptor.colorAttachments[0].loadAction = .load
    }
    
    private func renderFluidPrepare(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        projectionMatrix: float4x4,
        viewMatrix: float4x4,
        fluidDepthStencilDescriptor: MTLDepthStencilDescriptor? = nil
    ) -> FluidRenderContext {
        guard let fluidRenderer = fluidRenderer else { return .none }
        
        // Only prepare offscreen textures for water rendering mode
        guard fluidRenderer.currentRenderMode == .water else { return .none }
        
        // Get screen size from render pass descriptor
        guard let colorTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return .none
        }
        let screenSize = SIMD2<Float>(Float(colorTexture.width), Float(colorTexture.height))
        
        
        let particleBuffer = fluidRenderer.scene.getRenderParticleBuffer()
        
        guard let (textures, release) = fluidRenderer.acquireTextures(for: screenSize) else {
            return .none
        }
        
        // Render depth map (this renders to its own offscreen depth texture)
        fluidRenderer.renderDepthMap(commandBuffer: commandBuffer, particleBuffer: particleBuffer, textures: textures)
        
        // Apply bilateral filter to depth
        for _ in 0..<4 {
            fluidRenderer.applyDepthFilter(
                commandBuffer: commandBuffer,
                textures: textures,
                depthThreshold: 0.01,
                filterRadius: 3
            )
        }
        
        // Render thickness map
        fluidRenderer.renderThicknessMap(commandBuffer: commandBuffer, particleBuffer: particleBuffer, textures: textures)
        
        // Apply Gaussian filter to thickness
        fluidRenderer.applyThicknessFilter(commandBuffer: commandBuffer, textures: textures, filterRadius: 4)
        
        return .prepared(textures, release)
    }
    
    private func renderFluid(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        projectionMatrix: float4x4,
        viewMatrix: float4x4,
        context: FluidRenderContext = .none,
        depthStencilDescriptor: MTLDepthStencilDescriptor?
    ) {
        guard let fluidRenderer = fluidRenderer else { return }
        
        // Get screen size from render pass descriptor
        guard let colorTexture = renderPassDescriptor.colorAttachments[0].texture else { return }
        let screenSize = SIMD2<Float>(Float(colorTexture.width), Float(colorTexture.height))
        
        switch fluidRenderer.currentRenderMode {
        case .particles:
            renderParticles(
                commandBuffer: commandBuffer,
                renderPassDescriptor: renderPassDescriptor,
                screenSize: screenSize,
                depthStencilDescriptor: depthStencilDescriptor
            )
        case .water:
            renderWater(
                commandBuffer: commandBuffer,
                renderPassDescriptor: renderPassDescriptor,
                projectionMatrix: projectionMatrix,
                viewMatrix: viewMatrix,
                screenSize: screenSize,
                context: context,
                depthStencilDescriptor: depthStencilDescriptor
            )
        }
    }
    
    private func renderParticles(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        screenSize: SIMD2<Float>,
        depthStencilDescriptor: MTLDepthStencilDescriptor?
    ) {
        guard let fluidRenderer = fluidRenderer,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        let particleBuffer = fluidRenderer.scene.getRenderParticleBuffer()
        
        
        guard let pipelineState = fluidRenderer.pressureHeatmapPipelineState else {
            renderEncoder.endEncoding()
            return
        }
        
        renderEncoder.setRenderPipelineState(pipelineState)
        if let depthStencilDescriptor{
            // Create depth stencil state for particles
            let particleDepthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)
            renderEncoder.setDepthStencilState(particleDepthStencilState)
        }
        renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(fluidRenderer.scene.getVertexUniformBuffer(), offset: 0, index: 1)
        
        if fluidRenderer.currentParticleRenderMode == .pressureHeatmap {
            renderEncoder.setVertexBuffer(fluidRenderer.scene.getRenderGridBuffer(), offset: 0, index: 2)
        }
        
        renderEncoder.drawPrimitives(
            type: .point,
            vertexStart: 0,
            vertexCount: fluidRenderer.particleCount
        )
        
        renderEncoder.endEncoding()
    }
    
    private func renderWater(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        projectionMatrix: float4x4,
        viewMatrix: float4x4,
        screenSize: SIMD2<Float>,
        context: FluidRenderContext,
        depthStencilDescriptor: MTLDepthStencilDescriptor?
    ) {
        guard let fluidRenderer = fluidRenderer else {
            return
        }
        
        // Use pre-prepared textures if available, otherwise fallback to original behavior
        let (textures, release): (FluidRenderTextures, () -> Void)
        
        switch context {
        case .prepared(let preparedTextures, let preparedRelease):
            textures = preparedTextures
            release = preparedRelease
        case .none:
            // Fallback: acquire and process textures inline (original behavior)
            guard let (acquiredTextures, acquiredRelease) = fluidRenderer.acquireTextures(for: screenSize) else {
                return
            }
            textures = acquiredTextures
            release = acquiredRelease
            
            let particleBuffer = fluidRenderer.scene.getRenderParticleBuffer()
            
            // Render depth map
            fluidRenderer.renderDepthMap(commandBuffer: commandBuffer, particleBuffer: particleBuffer, textures: textures)
            
            // Apply bilateral filter to depth
            for _ in 0..<4 {
                fluidRenderer.applyDepthFilter(
                    commandBuffer: commandBuffer,
                    textures: textures,
                    depthThreshold: 0.01,
                    filterRadius: 3
                )
            }
            
            // Render thickness map
            fluidRenderer.renderThicknessMap(commandBuffer: commandBuffer, particleBuffer: particleBuffer, textures: textures)
            
            // Apply Gaussian filter to thickness
            fluidRenderer.applyThicknessFilter(commandBuffer: commandBuffer, textures: textures, filterRadius: 4)
        }
        
        commandBuffer.addCompletedHandler { _ in release() }
        
        // Render final fluid surface
        let fluidUniformPointer = fluidRenderer.scene.getFluidRenderUniformBuffer().contents().bindMemory(
            to: FluidRenderUniforms.self,
            capacity: 1
        )
        
        let texelSize = SIMD2<Float>(1.0 / screenSize.x, 1.0 / screenSize.y)
        let invProjectionMatrix = projectionMatrix.inverse
        let invViewMatrix = viewMatrix.inverse
        
        fluidUniformPointer[0] = FluidRenderUniforms(
            texelSize: texelSize,
            sphereSize: 1.0,
            invProjectionMatrix: invProjectionMatrix,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            invViewMatrix: invViewMatrix
        )
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Use custom depth stencil descriptor if provided, otherwise use default
        
        renderEncoder.setRenderPipelineState(fluidRenderer.fluidSurfacePipelineState)
        if let depthStencilDescriptor{
            let fluidDepthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)
            renderEncoder.setDepthStencilState(fluidDepthStencilState)
        }

        renderEncoder.setVertexBuffer(fluidRenderer.scene.getFluidRenderUniformBuffer(), offset: 0, index: 0)
        renderEncoder.setFragmentTexture(textures.depthTexture, index: 0)
        renderEncoder.setFragmentTexture(textures.filteredThicknessTexture, index: 1)
        renderEncoder.setFragmentTexture(textures.environmentTexture, index: 2)
        renderEncoder.setFragmentBuffer(fluidRenderer.scene.getFluidRenderUniformBuffer(), offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        renderEncoder.endEncoding()
    }
    
    private func renderCollision(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        cameraProjectionMatrix: float4x4? = nil,
        cameraViewMatrix: float4x4? = nil,
        depthStencilDescriptor: MTLDepthStencilDescriptor? = nil
    ) {
        
        guard let fluidRenderer = fluidRenderer,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Set depth stencil state if descriptor is provided
        if let depthStencilDescriptor = depthStencilDescriptor {
            let depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
            renderEncoder.setDepthStencilState(depthStencilState)
        }
        
        // Render collision mesh with appropriate matrices
        if let camProjMatrix = cameraProjectionMatrix,
           let camViewMatrix = cameraViewMatrix {
            // Use camera world matrices for AR mode
            fluidRenderer.renderCollisionMeshInEncoder(
                renderEncoder: renderEncoder,
                arProjectionMatrix: camProjMatrix,
                arViewMatrix: camViewMatrix
            )
        } else {
            // Use fluid coordinate matrices for normal mode
            fluidRenderer.renderCollisionMeshInEncoder(renderEncoder: renderEncoder)
        }
        
        renderEncoder.endEncoding()
    }
    
    private func renderOverlay(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        depthStencilDescriptor: MTLDepthStencilDescriptor? = nil
    ) {
        guard let colorAttachmentTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return
        }
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Set depth stencil state if descriptor is provided
        if let depthStencilDescriptor = depthStencilDescriptor {
            let depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
            renderEncoder.setDepthStencilState(depthStencilState)
        }
        
        // Render additional overlay (independent of background renderer type)
        overlayRenderer?.renderOverlay(renderEncoder: renderEncoder, targetTexture: colorAttachmentTexture)
        
        renderEncoder.endEncoding()
    }
}
