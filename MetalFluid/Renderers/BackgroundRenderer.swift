import Metal
import UIKit

// MARK: - Background Renderer Abstraction
protocol BackgroundRenderer {
    // Whether the background should be considered transparent for subsequent passes
    var isTransparent: Bool { get }
    // Render background to specific target texture (for offscreen rendering) using the given command buffer
    func renderBackground(commandBuffer: MTLCommandBuffer, targetTexture: MTLTexture) -> MTLTexture?
    // Optional: update collision SDF if the background source provides it (e.g., AR)
    func updateCollisionSDFIfNeeded()
    // Optional: overlay rendering into an existing render encoder (drawn after scene)
    func renderOverlay(renderEncoder: MTLRenderCommandEncoder, targetTexture: MTLTexture)
}

extension BackgroundRenderer {
    func updateCollisionSDFIfNeeded() {}
    func renderOverlay(renderEncoder: MTLRenderCommandEncoder, targetTexture: MTLTexture) {}
}

// MARK: - AR Adapter
final class ARBackgroundRendererAdapter: BackgroundRenderer {
    private let arRenderer: ARRenderer
    private weak var fluidRenderer: MPMFluidRenderer?
    var isTransparent: Bool
    
    init(arRenderer: ARRenderer, fluidRenderer: MPMFluidRenderer? = nil, isTransparent: Bool) {
        self.arRenderer = arRenderer
        self.fluidRenderer = fluidRenderer
        self.isTransparent = isTransparent
    }
    
    func renderBackground(commandBuffer: MTLCommandBuffer, targetTexture: MTLTexture) -> MTLTexture? {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = targetTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        renderPassDescriptor.colorAttachments[0].storeAction = .store
                
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return nil
        }
        
        arRenderer.renderCameraBackground(commandEncoder: renderEncoder)
        renderEncoder.endEncoding()
        return targetTexture
    }

    func renderOverlay(renderEncoder: MTLRenderCommandEncoder, targetTexture: MTLTexture) {
        // Compute viewport size and current interface orientation for AR matrices
        let viewportSize = CGSize(width: targetTexture.width, height: targetTexture.height)
        let orientation: UIInterfaceOrientation = {
            if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
                return scene.interfaceOrientation
            }
            return .portrait
        }()
        
        // AR frame matrices are now handled by IntegratedRenderer
        
        arRenderer.renderARMeshWireframeInEncoder(renderEncoder: renderEncoder, viewportSize: viewportSize, orientation: orientation)
    }
}

// MARK: - Solid Color Background
final class SolidColorBackgroundRenderer: BackgroundRenderer {
    var isTransparent: Bool
    var clearColor: MTLClearColor
    
    init(color: MTLClearColor, transparent: Bool = false) {
        self.clearColor = color
        self.isTransparent = transparent
    }
    
    func renderBackground(commandBuffer: MTLCommandBuffer, targetTexture: MTLTexture) -> MTLTexture? {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = targetTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return nil
        }
        
        renderEncoder.endEncoding()
        return targetTexture;
    }
}
