import Foundation
import simd
import Metal
import UIKit
// MARK: - Debug Animation Controller
class AnimationController {
    // Animation control properties
    private var animationTimer: Timer?
    private var animationStartTime: CFTimeInterval = 0
    
    // Animation parameters
    private let orbitSpeed: Float = 0.5      // radians per second
    private let orbitRadius: Float = 0.1     // orbit radius
    private let rotationSpeed: Float = 1.0   // radians per second (faster than orbit)
    private let scaleSpeed: Float = 0.8      // radians per second for scale oscillation
    private let scaleAmplitude: Float = 0.3  // scale variation amplitude (±30%)
    private let baseScale: Float = 1.0       // base scale factor
    private let updateInterval: TimeInterval = 1.0/60.0  // 60 FPS
    
    // References
    private weak var fluidRenderer: MPMFluidRenderer?
    private weak var viewController: ViewController?
    
    init(fluidRenderer: MPMFluidRenderer, viewController: ViewController) {
        self.fluidRenderer = fluidRenderer
        self.viewController = viewController
    }
    
    deinit {
        stopAnimation()
    }
    
    // MARK: - Animation Control
    func startAnimation() {
        stopAnimation() // Stop any existing animation first
        animationStartTime = CACurrentMediaTime()
        animationTimer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] _ in
            self?.updateAnimation()
        }
    }
    
    func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
        // Reset to identity transform
        fluidRenderer?.setWorldTransform(matrix_identity_float4x4)
    }
    
    var isAnimating: Bool {
        return animationTimer != nil
    }
    
    // MARK: - Animation Update
    private func updateAnimation() {
        guard let fluidRenderer = fluidRenderer,
              let viewController = viewController else { return }
        
        let currentTime = CACurrentMediaTime()
        let animationTime = currentTime - animationStartTime
        
        // Save slider values before applying world transform
        let sdfScale = viewController.sdfScaleSlider.value
        let sdfYOffset = viewController.sdfYOffsetSlider.value  
        let sdfYRotation = viewController.sdfYRotationSlider.value
        
        // Revolution (orbital motion around center)
        let orbitAngle = Float(animationTime) * orbitSpeed
        let orbitX = orbitRadius * cos(orbitAngle)
        let orbitZ = orbitRadius * sin(orbitAngle)
        let orbitTranslation = SIMD3<Float>(orbitX, 0, orbitZ)
        let orbitMatrix = float4x4(translation: orbitTranslation)
        
        // Rotation (spinning around own axis)
        let rotationAngle = Float(animationTime) * rotationSpeed
        let rotationMatrix = float4x4(rotationY: rotationAngle)
        
        // Scale (breathing/pulsing effect)
        let scaleAngle = Float(animationTime) * scaleSpeed
        let scaleFactor = baseScale + scaleAmplitude * sin(scaleAngle)
        let scaleMatrix = float4x4(scaling: SIMD3<Float>(scaleFactor, scaleFactor, scaleFactor))
        
        // Combine orbital motion, rotation, and scaling
        let combinedTransform = orbitMatrix * scaleMatrix * rotationMatrix
        
        // Apply to fluid renderer using setWorldTransform
        fluidRenderer.setWorldTransform(combinedTransform)
        
        // Immediately restore slider values after setWorldTransform
        DispatchQueue.main.async {
            fluidRenderer.collisionManager?.representativeItem.updateSdfTransform(scale: SIMD3<Float>(sdfScale, sdfScale, sdfScale), translate: SIMD3<Float>(0.0, sdfYOffset, 0.0), rotate: SIMD3<Float>(0.0,sdfYRotation * Float.pi / 180.0,0.0))
        }
    }
}
