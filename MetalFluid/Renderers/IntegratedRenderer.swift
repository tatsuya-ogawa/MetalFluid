import Metal
import MetalKit
import simd

class IntegratedRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private weak var fluidRenderer: MPMFluidRenderer?
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, fluidRenderer: MPMFluidRenderer) {
        self.device = device
        self.commandQueue = commandQueue
        self.fluidRenderer = fluidRenderer
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
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // 1. Command buffer generation (already done above)
        
        // 2. Perform compute if needed
        if performCompute {
            fluidRenderer.compute(commandBuffer: commandBuffer)
        }
        
        // 3. Background render
        renderBackground(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor
        )
        
        // 4. Fluid render
        renderFluid(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix
        )
        
        // 5. Collision render
        renderCollision(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            cameraProjectionMatrix: cameraProjectionMatrix,
            cameraViewMatrix: cameraViewMatrix
        )
        
        // 6. Overlay render (AR wireframe, etc.)
        renderOverlay(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor
        )
        
        // 7. Commit
        commandBuffer.commit()
    }
    
    private func renderBackground(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor
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
    
    private func renderFluid(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        projectionMatrix: float4x4,
        viewMatrix: float4x4
    ) {
        guard let fluidRenderer = fluidRenderer else { return }
        
        // Get screen size from render pass descriptor
        guard let colorTexture = renderPassDescriptor.colorAttachments[0].texture else {
            return
        }
        let screenSize = SIMD2<Float>(Float(colorTexture.width), Float(colorTexture.height))
        
        // Ensure depth buffer
        fluidRenderer.ensureDepthBuffer(for: renderPassDescriptor, screenSize: screenSize, label: "IntegratedRenderDepthBuffer")
        
        switch fluidRenderer.currentRenderMode {
        case .particles:
            renderParticles(
                commandBuffer: commandBuffer,
                renderPassDescriptor: renderPassDescriptor,
                screenSize: screenSize
            )
        case .water:
            renderWater(
                commandBuffer: commandBuffer,
                renderPassDescriptor: renderPassDescriptor,
                projectionMatrix: projectionMatrix,
                viewMatrix: viewMatrix,
                screenSize: screenSize
            )
        }
    }
    
    private func renderParticles(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        screenSize: SIMD2<Float>
    ) {
        guard let fluidRenderer = fluidRenderer,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        let particleBuffer = fluidRenderer.scene.getRenderParticleBuffer()
        
        // Create depth stencil state for particles
        let particleDepthStencilDescriptor = MTLDepthStencilDescriptor()
        particleDepthStencilDescriptor.depthCompareFunction = .less
        particleDepthStencilDescriptor.isDepthWriteEnabled = false
        let particleDepthStencilState = device.makeDepthStencilState(descriptor: particleDepthStencilDescriptor)!
        
        guard let pipelineState = fluidRenderer.pressureHeatmapPipelineState else {
            renderEncoder.endEncoding()
            return
        }
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setDepthStencilState(particleDepthStencilState)
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
        screenSize: SIMD2<Float>
    ) {
        guard let fluidRenderer = fluidRenderer else {
            return
        }
        
        let particleBuffer = fluidRenderer.scene.getRenderParticleBuffer()
        
        guard let (textures, release) = fluidRenderer.acquireTextures(for: screenSize) else {
            return
        }
        
        commandBuffer.addCompletedHandler { _ in release() }
        
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
        
        // Render final fluid surface
        let fluidUniformPointer = fluidRenderer.scene.getFluidRenderUniformBuffer().contents().bindMemory(
            to: FluidRenderUniforms.self,
            capacity: 1
        )
        
        let texelSize = SIMD2<Float>(1.0 / textures.screenSize.x, 1.0 / textures.screenSize.y)
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
        
        let fluidDepthStencilDescriptor = MTLDepthStencilDescriptor()
        fluidDepthStencilDescriptor.depthCompareFunction = .less
        fluidDepthStencilDescriptor.isDepthWriteEnabled = true
        let fluidDepthStencilState = device.makeDepthStencilState(descriptor: fluidDepthStencilDescriptor)!
        
        renderEncoder.setRenderPipelineState(fluidRenderer.fluidSurfacePipelineState)
        renderEncoder.setDepthStencilState(fluidDepthStencilState)
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
        cameraViewMatrix: float4x4? = nil
    ) {
        guard let fluidRenderer = fluidRenderer,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
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
        renderPassDescriptor: MTLRenderPassDescriptor
    ) {
        guard let fluidRenderer = fluidRenderer,
              let backgroundRenderer = fluidRenderer.backgroundRenderer,
              let colorAttachmentTexture = renderPassDescriptor.colorAttachments[0].texture,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Render overlay (AR wireframe, etc.)
        backgroundRenderer.renderOverlay(renderEncoder: renderEncoder, targetTexture: colorAttachmentTexture)
        
        renderEncoder.endEncoding()
    }
}