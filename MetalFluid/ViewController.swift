import MetalKit
import UIKit
import simd

class ViewController: UIViewController {
    // For 10fps control
    private var lastComputeTime: CFTimeInterval = 0

    private var metalView: MTKView!
    private var fluidRenderer: MPMFluidRenderer!
    private var lastFrameTime: CFTimeInterval = 0

    // Camera properties for 3D viewing
    private var cameraDistance: Float = 3.0
    private var cameraAngleX: Float = 0.3  // Pitch
    private var cameraAngleY: Float = 0.5  // Yaw
    private var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, 0)

    // Interaction
    private var lastPanPoint: CGPoint = .zero
    private var isInteracting = false

    // Simulation control
    private var isAutoMode: Bool = true
    private var shouldStep: Bool = false

    // UI Elements
    private var controlPanel: UIView!
    private var modeButton: UIButton!
    private var stepButton: UIButton!
    private var resetButton: UIButton!
    private var dumpButton: UIButton!
    private var renderModeButton: UIButton!
    private var particleSizeSlider: UISlider!
    private var particleSizeLabel: UILabel!
    private var massScaleSlider: UISlider!
    private var massScaleLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        let env = ProcessInfo.processInfo.environment
        if let mode = env["SKIP_RENDER"], mode == "1" {
            print("Skipping rendering for testing environment")
            return
        }
        setupMetalView()
        setupRenderer()
        setupGestures()
        setupControlPanel()
    }

    private func setupMetalView() {
        metalView = MTKView(
            frame: view.bounds,
            device: MTLCreateSystemDefaultDevice()
        )
        metalView.delegate = self
        metalView.preferredFramesPerSecond = 60
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.depthStencilPixelFormat = .depth32Float
        metalView.clearColor = MTLClearColor(
            red: 0.05,
            green: 0.05,
            blue: 0.1,
            alpha: 1.0
        )

        view.addSubview(metalView)

        // Auto layout constraints
        metalView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            metalView.topAnchor.constraint(equalTo: view.topAnchor),
            metalView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            metalView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    private func setupRenderer() {
        fluidRenderer = MPMFluidRenderer()
    }

    private func setupGestures() {
        // Pan gesture for orbit control
        let panGesture = UIPanGestureRecognizer(
            target: self,
            action: #selector(handlePanGesture(_:))
        )
        metalView.addGestureRecognizer(panGesture)

        // Pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(
            target: self,
            action: #selector(handlePinchGesture(_:))
        )
        metalView.addGestureRecognizer(pinchGesture)

        // Tap gesture for fluid interaction
        let tapGesture = UITapGestureRecognizer(
            target: self,
            action: #selector(handleTapGesture(_:))
        )
        metalView.addGestureRecognizer(tapGesture)

        // Double tap to reset camera
        let doubleTapGesture = UITapGestureRecognizer(
            target: self,
            action: #selector(handleDoubleTapGesture(_:))
        )
        doubleTapGesture.numberOfTapsRequired = 2
        metalView.addGestureRecognizer(doubleTapGesture)

        tapGesture.require(toFail: doubleTapGesture)
    }

    private func setupControlPanel() {
        // Create control panel
        controlPanel = UIView()
        controlPanel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        controlPanel.layer.cornerRadius = 10
        view.addSubview(controlPanel)

        // Mode toggle button
        modeButton = UIButton(type: .system)
        modeButton.setTitle("Auto Mode", for: .normal)
        modeButton.setTitleColor(.white, for: .normal)
        modeButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        modeButton.layer.cornerRadius = 8
        modeButton.addTarget(
            self,
            action: #selector(toggleMode),
            for: .touchUpInside
        )

        // Step button
        stepButton = UIButton(type: .system)
        stepButton.setTitle("Step", for: .normal)
        stepButton.setTitleColor(.white, for: .normal)
        stepButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        stepButton.layer.cornerRadius = 8
        stepButton.addTarget(
            self,
            action: #selector(stepSimulation),
            for: .touchUpInside
        )
        stepButton.isEnabled = false
        stepButton.alpha = 0.5

        // Reset button
        resetButton = UIButton(type: .system)
        resetButton.setTitle("Reset", for: .normal)
        resetButton.setTitleColor(.white, for: .normal)
        resetButton.backgroundColor = UIColor.systemRed.withAlphaComponent(0.8)
        resetButton.layer.cornerRadius = 8
        resetButton.addTarget(
            self,
            action: #selector(resetSimulation),
            for: .touchUpInside
        )

        // Dump button
        dumpButton = UIButton(type: .system)
        dumpButton.setTitle("Dump", for: .normal)
        dumpButton.setTitleColor(.white, for: .normal)
        dumpButton.backgroundColor = UIColor.systemCyan.withAlphaComponent(0.8)
        dumpButton.layer.cornerRadius = 8
        dumpButton.addTarget(
            self,
            action: #selector(dumpParticles),
            for: .touchUpInside
        )
        
        // Render mode button
        renderModeButton = UIButton(type: .system)
        renderModeButton.setTitle("Particles", for: .normal)
        renderModeButton.setTitleColor(.white, for: .normal)
        renderModeButton.backgroundColor = UIColor.systemPurple.withAlphaComponent(0.8)
        renderModeButton.layer.cornerRadius = 8
        renderModeButton.addTarget(
            self,
            action: #selector(toggleRenderMode),
            for: .touchUpInside
        )
        
        // Particle size slider
        particleSizeSlider = UISlider()
        particleSizeSlider.minimumValue = 0.5
        particleSizeSlider.maximumValue = 5.0
        particleSizeSlider.value = 1.0
        particleSizeSlider.addTarget(
            self,
            action: #selector(particleSizeChanged),
            for: .valueChanged
        )
        
        // Particle size label
        particleSizeLabel = UILabel()
        particleSizeLabel.text = "Size: 1.0x"
        particleSizeLabel.textColor = .white
        particleSizeLabel.font = UIFont.systemFont(ofSize: 14)
        particleSizeLabel.textAlignment = .center
        
        // Mass scale slider
        massScaleSlider = UISlider()
        massScaleSlider.minimumValue = 0.1
        massScaleSlider.maximumValue = 3.0
        massScaleSlider.value = 1.0
        massScaleSlider.addTarget(
            self,
            action: #selector(massScaleChanged),
            for: .valueChanged
        )
        
        // Mass scale label
        massScaleLabel = UILabel()
        massScaleLabel.text = "Mass: 1.0x"
        massScaleLabel.textColor = .white
        massScaleLabel.font = UIFont.systemFont(ofSize: 14)
        massScaleLabel.textAlignment = .center

        // Add buttons to control panel
        controlPanel.addSubview(modeButton)
        controlPanel.addSubview(stepButton)
        controlPanel.addSubview(resetButton)
        controlPanel.addSubview(dumpButton)
        controlPanel.addSubview(renderModeButton)
        controlPanel.addSubview(particleSizeSlider)
        controlPanel.addSubview(particleSizeLabel)
        controlPanel.addSubview(massScaleSlider)
        controlPanel.addSubview(massScaleLabel)

        // Setup constraints
        controlPanel.translatesAutoresizingMaskIntoConstraints = false
        modeButton.translatesAutoresizingMaskIntoConstraints = false
        stepButton.translatesAutoresizingMaskIntoConstraints = false
        resetButton.translatesAutoresizingMaskIntoConstraints = false
        dumpButton.translatesAutoresizingMaskIntoConstraints = false
        renderModeButton.translatesAutoresizingMaskIntoConstraints = false
        particleSizeSlider.translatesAutoresizingMaskIntoConstraints = false
        particleSizeLabel.translatesAutoresizingMaskIntoConstraints = false
        massScaleSlider.translatesAutoresizingMaskIntoConstraints = false
        massScaleLabel.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            // Control panel constraints
            controlPanel.topAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.topAnchor,
                constant: 20
            ),
            controlPanel.leadingAnchor.constraint(
                equalTo: view.leadingAnchor,
                constant: 20
            ),
            controlPanel.widthAnchor.constraint(equalToConstant: 200),
            controlPanel.heightAnchor.constraint(equalToConstant: 390),

            // Mode button constraints
            modeButton.topAnchor.constraint(
                equalTo: controlPanel.topAnchor,
                constant: 10
            ),
            modeButton.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            modeButton.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            modeButton.heightAnchor.constraint(equalToConstant: 40),

            // Step button constraints
            stepButton.topAnchor.constraint(
                equalTo: modeButton.bottomAnchor,
                constant: 10
            ),
            stepButton.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            stepButton.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            stepButton.heightAnchor.constraint(equalToConstant: 40),

            // Reset button constraints
            resetButton.topAnchor.constraint(
                equalTo: stepButton.bottomAnchor,
                constant: 10
            ),
            resetButton.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            resetButton.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            resetButton.heightAnchor.constraint(equalToConstant: 40),
            // Dump button constraints
            dumpButton.topAnchor.constraint(
                equalTo: resetButton.bottomAnchor,
                constant: 10
            ),
            dumpButton.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            dumpButton.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            dumpButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Render mode button constraints
            renderModeButton.topAnchor.constraint(
                equalTo: dumpButton.bottomAnchor,
                constant: 10
            ),
            renderModeButton.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            renderModeButton.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            renderModeButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Particle size label constraints
            particleSizeLabel.topAnchor.constraint(
                equalTo: renderModeButton.bottomAnchor,
                constant: 10
            ),
            particleSizeLabel.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            particleSizeLabel.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            particleSizeLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // Particle size slider constraints
            particleSizeSlider.topAnchor.constraint(
                equalTo: particleSizeLabel.bottomAnchor,
                constant: 5
            ),
            particleSizeSlider.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            particleSizeSlider.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            particleSizeSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // Mass scale label constraints
            massScaleLabel.topAnchor.constraint(
                equalTo: particleSizeSlider.bottomAnchor,
                constant: 10
            ),
            massScaleLabel.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            massScaleLabel.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            massScaleLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // Mass scale slider constraints
            massScaleSlider.topAnchor.constraint(
                equalTo: massScaleLabel.bottomAnchor,
                constant: 5
            ),
            massScaleSlider.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            massScaleSlider.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            massScaleSlider.heightAnchor.constraint(equalToConstant: 30),
        ])
    }
    @objc private func dumpParticles() {
        guard let renderer = fluidRenderer else { return }
        renderer.requestDebugDump()
    }
    
    @objc private func particleSizeChanged(_ slider: UISlider) {
        let size = slider.value
        particleSizeLabel.text = String(format: "Size: %.1fx", size)
        fluidRenderer.setParticleSizeMultiplier(size)
    }
    
    @objc private func massScaleChanged(_ slider: UISlider) {
        let scale = slider.value
        massScaleLabel.text = String(format: "Mass: %.1fx", scale)
        fluidRenderer.setMassScale(scale)
    }
    
    @objc private func toggleRenderMode() {
        guard let renderer = fluidRenderer else { return }
        renderer.toggleRenderMode()
        
        // Update button title based on current mode
        switch renderer.currentRenderMode {
        case .particles:
            renderModeButton.setTitle("Particles", for: .normal)
            renderModeButton.backgroundColor = UIColor.systemPurple.withAlphaComponent(0.8)
        case .water:
            renderModeButton.setTitle("Water", for: .normal)
            renderModeButton.backgroundColor = UIColor.systemTeal.withAlphaComponent(0.8)
        }
    }

    @objc private func toggleMode() {
        isAutoMode.toggle()

        if isAutoMode {
            modeButton.setTitle("Auto Mode", for: .normal)
            modeButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(
                0.8
            )
            stepButton.isEnabled = false
            stepButton.alpha = 0.5
        } else {
            modeButton.setTitle("Manual Mode", for: .normal)
            modeButton.backgroundColor = UIColor.systemOrange
                .withAlphaComponent(0.8)
            stepButton.isEnabled = true
            stepButton.alpha = 1.0
        }
    }

    @objc private func stepSimulation() {
        if !isAutoMode {
            shouldStep = true
        }
    }

    @objc private func resetSimulation() {
        fluidRenderer.reset()
        // Reset camera position for better overview
        cameraDistance = 5.0
        cameraAngleX = 0.2
        cameraAngleY = 0.3
    }

    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: metalView)

        switch gesture.state {
        case .began:
            lastPanPoint = gesture.location(in: metalView)

        case .changed:
            // Use translation directly for orbit control
            let deltaX = Float(translation.x) * 0.01
            let deltaY = Float(translation.y) * 0.01

            // Apply rotation: horizontal drag rotates around Y axis, vertical drag rotates around X axis
            cameraAngleY -= deltaX  // Yaw (horizontal rotation, inverted for natural feel)
            cameraAngleX += deltaY  // Pitch (vertical rotation)

            // Clamp vertical rotation to prevent flipping
            cameraAngleX = max(
                -Float.pi / 2 + 0.1,
                min(Float.pi / 2 - 0.1, cameraAngleX)
            )

            // Reset translation for next frame
            gesture.setTranslation(.zero, in: metalView)

        default:
            break
        }
    }

    @objc private func handlePinchGesture(_ gesture: UIPinchGestureRecognizer) {
        switch gesture.state {
        case .changed:
            cameraDistance /= Float(gesture.scale)
            cameraDistance = max(1.0, min(20.0, cameraDistance))  // Expand range
            gesture.scale = 1.0

        default:
            break
        }
    }

    @objc private func handleTapGesture(_ gesture: UITapGestureRecognizer) {
        let tapPoint = gesture.location(in: metalView)

        // Convert screen coordinates to world coordinates
        let normalizedX = Float(tapPoint.x / metalView.bounds.width) * 2.0 - 1.0
        let normalizedY =
            Float(1.0 - tapPoint.y / metalView.bounds.height) * 2.0 - 1.0

        let worldPosition = SIMD2<Float>(normalizedX, normalizedY)
        let force = SIMD2<Float>(
            Float.random(in: -5.0...5.0),
            Float.random(in: -2.0...8.0)
        )

        fluidRenderer.addForce(at: worldPosition, force: force)
    }

    @objc private func handleDoubleTapGesture(_ gesture: UITapGestureRecognizer)
    {
        // Reset camera to default position with better overview
        cameraDistance = 5.0
        cameraAngleX = 0.2
        cameraAngleY = 0.3
    }

    private func getCameraPosition() -> SIMD3<Float> {
        let x = cameraDistance * cos(cameraAngleX) * sin(cameraAngleY)
        let y = cameraDistance * sin(cameraAngleX)
        let z = cameraDistance * cos(cameraAngleX) * cos(cameraAngleY)
        return cameraTarget + SIMD3<Float>(x, y, z)
    }

    private func getViewMatrix() -> float4x4 {
        let cameraPosition = getCameraPosition()
        let up = SIMD3<Float>(0, 1, 0)
        return lookAt(eye: cameraPosition, center: cameraTarget, up: up)
    }

    private func getProjectionMatrix(aspectRatio: Float) -> float4x4 {
        let fov: Float = 45.0 * Float.pi / 180.0
        let nearPlane: Float = 0.1
        let farPlane: Float = 100.0
        return perspective(
            fovy: fov,
            aspect: aspectRatio,
            nearZ: nearPlane,
            farZ: farPlane
        )
    }
}
// Translation matrix extension
extension float4x4 {
    init(translation t: SIMD3<Float>) {
        self = matrix_identity_float4x4
        columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)
    }
}

extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle size changes if needed
    }

    func draw(in view: MTKView) {
        let currentTime = CACurrentMediaTime()
        let deltaTime = Float(currentTime - lastFrameTime)
        lastFrameTime = currentTime

        // Only compute at 10fps equivalent
        let computeInterval: CFTimeInterval = 0.1  // 10fps
        var performStep = isAutoMode || shouldStep
        if isAutoMode {
            if currentTime - lastComputeTime < computeInterval {
                performStep = false
            } else {
                lastComputeTime = currentTime
            }
        }

        let screenSize = SIMD2<Float>(
            Float(view.bounds.width),
            Float(view.bounds.height)
        )
        let aspectRatio = screenSize.x / screenSize.y

        // Create transformation matrices
        let baseViewMatrix = getViewMatrix()
        let projectionMatrix = getProjectionMatrix(aspectRatio: aspectRatio)
        // Create scale matrix (example: scale with scaleFactor)
        let scaleFactor: Float = fluidRenderer.getRenderScale(scale: 1.0)
        let scaleMatrix = float4x4(
            SIMD4<Float>(scaleFactor, 0, 0, 0),
            SIMD4<Float>(0, scaleFactor, 0, 0),
            SIMD4<Float>(0, 0, scaleFactor, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
        // Translate in opposite direction of domainOrigin
        let translateMatrix = float4x4(translation: -fluidRenderer.getDomainOriginTranslation())
        
        // Complete view matrix including scale and translation
        let viewMatrix = baseViewMatrix * scaleMatrix * translateMatrix
        let mvpMatrix = projectionMatrix * viewMatrix

        // Update simulation based on mode
        if performStep {
            fluidRenderer.update(
                deltaTime: deltaTime,
                screenSize: screenSize,
                mvpMatrix: mvpMatrix,
                projectionMatrix: projectionMatrix,
                viewMatrix: viewMatrix
            )
        }

        guard let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        // Use the new render mode switching system with proper matrices
        fluidRenderer.render(
            renderPassDescriptor: renderPassDescriptor,
            performCompute: performStep,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix
        )

        // reset manual step after issuing one compute pass
        shouldStep = false

        guard let drawable = view.currentDrawable else { return }
        drawable.present()
    }
}

// Matrix utility functions
func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>)
    -> float4x4
{
    let z = normalize(eye - center)
    let x = normalize(cross(up, z))
    let y = cross(z, x)

    return float4x4(
        SIMD4<Float>(x.x, y.x, z.x, 0),
        SIMD4<Float>(x.y, y.y, z.y, 0),
        SIMD4<Float>(x.z, y.z, z.z, 0),
        SIMD4<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
    )
}

func perspective(fovy: Float, aspect: Float, nearZ: Float, farZ: Float)
    -> float4x4
{
    let yScale = 1 / tan(fovy * 0.5)
    let xScale = yScale / aspect
    let zRange = farZ - nearZ
    let zScale = -(farZ + nearZ) / zRange
    let wzScale = -2 * farZ * nearZ / zRange

    return float4x4(
        SIMD4<Float>(xScale, 0, 0, 0),
        SIMD4<Float>(0, yScale, 0, 0),
        SIMD4<Float>(0, 0, zScale, -1),
        SIMD4<Float>(0, 0, wzScale, 0)
    )
}
