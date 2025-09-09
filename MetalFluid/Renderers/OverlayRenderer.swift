//
//  OverlayRenderer.swift
//  MetalFluid
//
//  Protocol for rendering overlays that appear on top of the main scene
//

import Metal
import MetalKit

/// Protocol for objects that can render overlays into an existing render encoder
protocol OverlayRenderer: AnyObject {
    /// Render overlay content into the provided render encoder
    /// - Parameters:
    ///   - renderEncoder: The active render command encoder
    ///   - targetTexture: The target texture being rendered to (for viewport size calculation)
    func renderOverlay(renderEncoder: MTLRenderCommandEncoder, targetTexture: MTLTexture)
}