import MetalKit
import UIKit
import simd
import ReplayKit
#if canImport(ARKit)
import ARKit
#endif

// Helper to get RGBA components from UIColor
private extension UIColor {
    var rgba: (red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat) {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        getRed(&r, green: &g, blue: &b, alpha: &a)
        return (r, g, b, a)
    }
}

class ViewController: UIViewController {
    private let recordKey = "r"
    // For 10fps control
    private var lastComputeTime: CFTimeInterval = 0

    private var metalView: MTKView!
    private var fluidRenderer: MPMFluidRenderer!
    private var lastFrameTime: CFTimeInterval = 0
    
    // Camera properties for 3D viewing
    private var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
    
    // Interaction
    private var lastPanPoint: CGPoint = .zero
    private var isInteracting = false

    // Simulation control
    private var isAutoMode: Bool = true
    private var shouldStep: Bool = false
    
    // Animation control (for debug purposes)
    private var animationController: AnimationController?
    
    // Wireframe state tracking
    private var isWireframeMode: Bool = false
    
    // Transform coefficients (managed in VC, applied when needed)
    private var worldTranslation: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, -3.0)
    private var worldYaw: Float = 0.0
    private var worldPitch: Float = 0.0
    private var worldScale: Float = 1.0

    // Orbit control tuning
    private let rotateSpeed: Float = 0.01
    private let worldPanSpeed: Float = 0.003
    private let minPitch: Float = -Float.pi/2 + 0.1
    private let maxPitch: Float =  Float.pi/2 - 0.1

    // Initial values (shared between sliders and FluidRenderer)
    private let initialParticleCount: Float = 40000
    private let initialGridSize: Float = 64
    private let initialGridHeightMultiplier: Float = 1.5

    // UI Elements
    private var controlPanel: UIView!
    private var modeButton: UIButton!
    private var stepButton: UIButton!
    private var resetButton: UIButton!
    private var renderModeButton: UIButton!
    private var particleSizeSlider: UISlider!
    private var particleSizeLabel: UILabel!
    private var massScaleSlider: UISlider!
    private var massScaleLabel: UILabel!
    private var particleCountSlider: UISlider!
    private var particleCountLabel: UILabel!
    private var gridSizeSlider: UISlider!
    private var gridSizeLabel: UILabel!
    internal var sdfScaleSlider: UISlider!
    private var sdfScaleLabel: UILabel!
    internal var sdfYOffsetSlider: UISlider!
    private var sdfYOffsetLabel: UILabel!
    internal var sdfYRotationSlider: UISlider!
    private var sdfYRotationLabel: UILabel!
    
    // Transparent background control
    private var transparentBackgroundToggle: UIButton!
    
    // Collision controls panel (right side)
    private var collisionPanel: UIView!
    private var collisionToggleButton: UIButton!
    private var meshVisibilityButton: UIButton!
    private var wireframeButton: UIButton!
    private var materialModeButton: UIButton!
    
    // AR controls panel (bottom right)
    private var arPanel: UIView!
    private var arToggleButton: UIButton!
    private var arPanelHeightConstraint: NSLayoutConstraint!
    
    // ReplayKit recording
    private let screenRecorder = RPScreenRecorder.shared()
    private var isRecording = false
    private let meshLoader: MeshLoader = MeshLoader(scaleFactor: 1.0)
    
    // MARK: - Transform Management
    
    func setInitialWorldTransform() {
        // Just set the initial coefficients, don't apply to renderer yet
        worldTranslation = SIMD3<Float>(0.0, 0.0, -3.0)
        worldYaw = 0.0
        worldPitch = 0.0
        worldScale = 1.0
    }
    
    private func computeWorldTransform() -> float4x4 {
        // Compute transform from current coefficients: T * R * S order
        let S = float4x4(scale: SIMD3<Float>(worldScale, worldScale, worldScale))
        let R = float4x4(rotationY: worldYaw) * float4x4(rotationX: worldPitch)
        let T = float4x4(translation: worldTranslation)
        return T * R * S
    }
    
    private func applyWorldTransformToRenderer() {
        // Apply current coefficients to the renderer only when needed
        let transform = computeWorldTransform()
        fluidRenderer.setWorldTransform(transform)
    }
    
    private func updateRendererTransformIfNeeded() {
        // Apply transform to renderer when rendering or when needed
        applyWorldTransformToRenderer()
    }
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
        setupCollisionPanel()
        setupARPanel()
        setupKeyboardHandling()
        
        // Set up delegate for AR state changes
        fluidRenderer.viewController = self
        
        // Initialize debug animation controller (disabled by default)
        animationController = AnimationController(fluidRenderer: fluidRenderer, viewController: self)
        setInitialWorldTransform()
        
        // Apply initial transform
        updateSdfTransfom()
        applyWorldTransformToRenderer()
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
        // Use predefined initial values for FluidRenderer constructor
        fluidRenderer = MPMFluidRenderer(
            particleCount: Int(initialParticleCount), 
            gridSize: Int(initialGridSize), 
            gridHeightMultiplier: initialGridHeightMultiplier
        )
        
        // Install default background renderer
        installDefaultBackgroundRenderer()
        
        // Load Stanford Bunny for collision detection
        setupCollisionMesh()
    }
    // MARK: - Mesh Loading
    /// Load Stanford Bunny asynchronously
    private func loadStanfordBunnyAsync(resolution: SIMD3<Int32>, gridBoundaryMin: SIMD3<Float>? = nil, gridBoundaryMax: SIMD3<Float>? = nil, completion: @escaping (Bool) -> Void) {
        meshLoader.loadStanfordBunnyAsync(offsetToBottom: nil) { [weak self] triangles in
            guard let self = self else {
                completion(false)
                return
            }
            
            if triangles.isEmpty {
                print("No triangles loaded from Stanford Bunny")
                completion(false)
                return
            }
            if let collisionManager = fluidRenderer.collisionManager{
                collisionManager.representativeItem.processAndGenerateSDF(sdfGenerator: collisionManager.sdfGenerator, triangles: triangles, resolution: resolution, gridBoundaryMin: gridBoundaryMin, gridBoundaryMax: gridBoundaryMax)
            }
            completion(true)
        }
    }
    
    private func setupCollisionMesh() {
        // Get grid boundary to position bunny correctly
        let (boundaryMin, boundaryMax) = fluidRenderer.getBoundaryMinMax()
        
        // Use fixed SDF resolution (64x64x64) instead of world resolution
        // The collision system will handle scaling and transformation automatically
        let fixedSDFResolution = SIMD3<Int32>(64, 64, 64)
        
        // Load Stanford Bunny asynchronously (with caching)
        loadStanfordBunnyAsync(
            resolution: fixedSDFResolution,
            gridBoundaryMin: boundaryMin,
            gridBoundaryMax: boundaryMax
        ) { [weak self] success in
            if success {
                print("✅ Stanford Bunny loaded successfully!")
                // Configure collision visualization
                self?.fluidRenderer.collisionManager?.setMeshVisible(true)
                self?.fluidRenderer.collisionManager?.representativeItem.setMeshColor(SIMD4<Float>(1.0, 1.0, 1.0, 0.8)) // Semi-transparent white
                print("🐰 Stanford Bunny collision mesh configured!")
            } else {
                print("❌ Failed to load Stanford Bunny. Collision detection disabled.")
            }
        }
    }

    private func setupGestures() {
        // Pan gesture for orbit control
        let panGesture = UIPanGestureRecognizer(
            target: self,
            action: #selector(handlePanGesture(_:))
        )
        panGesture.maximumNumberOfTouches = 1
        panGesture.delegate = self
        metalView.addGestureRecognizer(panGesture)

        // Two-finger pan for world translation (screen-plane)
        let twoFingerPan = UIPanGestureRecognizer(
            target: self,
            action: #selector(handleTwoFingerPanGesture(_:))
        )
        twoFingerPan.minimumNumberOfTouches = 2
        twoFingerPan.maximumNumberOfTouches = 2
        twoFingerPan.delegate = self
        metalView.addGestureRecognizer(twoFingerPan)

        // Pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(
            target: self,
            action: #selector(handlePinchGesture(_:))
        )
        pinchGesture.delegate = self
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

        // Material mode button
        materialModeButton = UIButton(type: .system)
        materialModeButton.setTitle("Fluid", for: .normal)
        materialModeButton.setTitleColor(.white, for: .normal)
        materialModeButton.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.8)
        materialModeButton.layer.cornerRadius = 8
        materialModeButton.addTarget(
            self,
            action: #selector(toggleMaterialMode),
            for: .touchUpInside
        )
        
        // World orbit toggle button
        // Moved to AR panel (bottom-right) in setupARPanel
        
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
        
        // Particle count slider
        particleCountSlider = UISlider()
        particleCountSlider.minimumValue = 1000
        particleCountSlider.maximumValue = 400000
        particleCountSlider.value = initialParticleCount
        particleCountSlider.addTarget(
            self,
            action: #selector(particleCountChanged),
            for: .valueChanged
        )
        
        // Particle count label
        particleCountLabel = UILabel()
        particleCountLabel.text = "Particles: \(particleCountSlider.value)"
        particleCountLabel.textColor = .white
        particleCountLabel.font = UIFont.systemFont(ofSize: 14)
        particleCountLabel.textAlignment = .center
        
        // Grid size slider
        gridSizeSlider = UISlider()
        gridSizeSlider.minimumValue = 32
        gridSizeSlider.maximumValue = 128
        gridSizeSlider.value = initialGridSize
        gridSizeSlider.addTarget(
            self,
            action: #selector(gridSizeChanged),
            for: .valueChanged
        )
        
        // Grid size label
        gridSizeLabel = UILabel()
        gridSizeLabel.text = "Grid: \(gridSizeSlider.value)³"
        gridSizeLabel.textColor = .white
        gridSizeLabel.font = UIFont.systemFont(ofSize: 14)
        gridSizeLabel.textAlignment = .center
        
        // SDF scale slider
        sdfScaleSlider = UISlider()
        sdfScaleSlider.minimumValue = 0.01
        sdfScaleSlider.maximumValue = 3.0
        sdfScaleSlider.value = 1.0
        sdfScaleSlider.addTarget(
            self,
            action: #selector(sdfScaleChanged),
            for: .valueChanged
        )
        
        // SDF scale label
        sdfScaleLabel = UILabel()
        sdfScaleLabel.text = String(format: "SDF Scale: %.2fx", sdfScaleSlider.value)
        sdfScaleLabel.textColor = .white
        sdfScaleLabel.font = UIFont.systemFont(ofSize: 14)
        sdfScaleLabel.textAlignment = .center
        
        // SDF Y offset slider
        sdfYOffsetSlider = UISlider()
        sdfYOffsetSlider.minimumValue = -5.0
        sdfYOffsetSlider.maximumValue = 5.0
        sdfYOffsetSlider.value = 0.0
        sdfYOffsetSlider.addTarget(
            self,
            action: #selector(sdfYOffsetChanged),
            for: .valueChanged
        )
        
        // SDF Y offset label
        sdfYOffsetLabel = UILabel()
        sdfYOffsetLabel.text = String(format: "SDF Y Offset: %.1f", sdfYOffsetSlider.value)
        sdfYOffsetLabel.textColor = .white
        sdfYOffsetLabel.font = UIFont.systemFont(ofSize: 14)
        sdfYOffsetLabel.textAlignment = .center
        
        // SDF Y rotation slider
        sdfYRotationSlider = UISlider()
        sdfYRotationSlider.minimumValue = 0.0
        sdfYRotationSlider.maximumValue = 360.0
        sdfYRotationSlider.value = 0.0
        sdfYRotationSlider.addTarget(
            self,
            action: #selector(sdfYRotationChanged),
            for: .valueChanged
        )
        
        // Transparent background toggle
        transparentBackgroundToggle = UIButton(type: .system)
        transparentBackgroundToggle.setTitle("Transparent: OFF", for: .normal)
        transparentBackgroundToggle.setTitleColor(.white, for: .normal)
        transparentBackgroundToggle.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        transparentBackgroundToggle.layer.cornerRadius = 8
        transparentBackgroundToggle.addTarget(
            self,
            action: #selector(toggleTransparentBackground),
            for: .touchUpInside
        )
        
        // SDF Y rotation label
        sdfYRotationLabel = UILabel()
        sdfYRotationLabel.text = String(format: "SDF Y Rotation: %.0f°", sdfYRotationSlider.value)
        sdfYRotationLabel.textColor = .white
        sdfYRotationLabel.font = UIFont.systemFont(ofSize: 14)
        sdfYRotationLabel.textAlignment = .center

        // Add buttons to control panel
        controlPanel.addSubview(modeButton)
        controlPanel.addSubview(stepButton)
        controlPanel.addSubview(resetButton)
        controlPanel.addSubview(renderModeButton)
        controlPanel.addSubview(materialModeButton)
        // worldOrbitButton lives in arPanel, not controlPanel
        controlPanel.addSubview(particleSizeSlider)
        controlPanel.addSubview(particleSizeLabel)
        controlPanel.addSubview(massScaleSlider)
        controlPanel.addSubview(massScaleLabel)
        controlPanel.addSubview(particleCountSlider)
        controlPanel.addSubview(particleCountLabel)
        controlPanel.addSubview(gridSizeSlider)
        controlPanel.addSubview(gridSizeLabel)

        // Setup constraints
        controlPanel.translatesAutoresizingMaskIntoConstraints = false
        modeButton.translatesAutoresizingMaskIntoConstraints = false
        stepButton.translatesAutoresizingMaskIntoConstraints = false
        resetButton.translatesAutoresizingMaskIntoConstraints = false
        renderModeButton.translatesAutoresizingMaskIntoConstraints = false
        materialModeButton.translatesAutoresizingMaskIntoConstraints = false
        // worldOrbitButton constraints are defined in setupARPanel
        particleSizeSlider.translatesAutoresizingMaskIntoConstraints = false
        particleSizeLabel.translatesAutoresizingMaskIntoConstraints = false
        massScaleSlider.translatesAutoresizingMaskIntoConstraints = false
        massScaleLabel.translatesAutoresizingMaskIntoConstraints = false
        particleCountSlider.translatesAutoresizingMaskIntoConstraints = false
        particleCountLabel.translatesAutoresizingMaskIntoConstraints = false
        gridSizeSlider.translatesAutoresizingMaskIntoConstraints = false
        gridSizeLabel.translatesAutoresizingMaskIntoConstraints = false
        sdfScaleSlider.translatesAutoresizingMaskIntoConstraints = false
        sdfScaleLabel.translatesAutoresizingMaskIntoConstraints = false
        sdfYOffsetSlider.translatesAutoresizingMaskIntoConstraints = false
        sdfYOffsetLabel.translatesAutoresizingMaskIntoConstraints = false
        sdfYRotationSlider.translatesAutoresizingMaskIntoConstraints = false
        sdfYRotationLabel.translatesAutoresizingMaskIntoConstraints = false

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
            
            // Render mode button constraints
            renderModeButton.topAnchor.constraint(
                equalTo: resetButton.bottomAnchor,
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
            
            // Material mode button constraints
            materialModeButton.topAnchor.constraint(
                equalTo: renderModeButton.bottomAnchor,
                constant: 10
            ),
            materialModeButton.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            materialModeButton.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            materialModeButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Particle size label constraints
            particleSizeLabel.topAnchor.constraint(
                equalTo: materialModeButton.bottomAnchor,
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
            
            // Particle count label constraints
            particleCountLabel.topAnchor.constraint(
                equalTo: massScaleSlider.bottomAnchor,
                constant: 10
            ),
            particleCountLabel.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            particleCountLabel.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            particleCountLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // Particle count slider constraints
            particleCountSlider.topAnchor.constraint(
                equalTo: particleCountLabel.bottomAnchor,
                constant: 5
            ),
            particleCountSlider.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            particleCountSlider.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            particleCountSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // Grid size label constraints
            gridSizeLabel.topAnchor.constraint(
                equalTo: particleCountSlider.bottomAnchor,
                constant: 10
            ),
            gridSizeLabel.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            gridSizeLabel.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            gridSizeLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // Grid size slider constraints
            gridSizeSlider.topAnchor.constraint(
                equalTo: gridSizeLabel.bottomAnchor,
                constant: 5
            ),
            gridSizeSlider.leadingAnchor.constraint(
                equalTo: controlPanel.leadingAnchor,
                constant: 10
            ),
            gridSizeSlider.trailingAnchor.constraint(
                equalTo: controlPanel.trailingAnchor,
                constant: -10
            ),
            gridSizeSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // Bottom constraint to define controlPanel height
            controlPanel.bottomAnchor.constraint(
                equalTo: gridSizeSlider.bottomAnchor,
                constant: 10
            ),
        ])
    }
    
    private func setupCollisionPanel() {
        // Create collision control panel (right side)
        collisionPanel = UIView()
        collisionPanel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        collisionPanel.layer.cornerRadius = 10
        view.addSubview(collisionPanel)
        
        // Collision toggle button
        collisionToggleButton = UIButton(type: .system)
        collisionToggleButton.setTitle("Collision: ON", for: .normal)
        collisionToggleButton.setTitleColor(.white, for: .normal)
        collisionToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        collisionToggleButton.layer.cornerRadius = 8
        collisionToggleButton.addTarget(
            self,
            action: #selector(toggleCollision),
            for: .touchUpInside
        )
                
        // Mesh visibility button
        meshVisibilityButton = UIButton(type: .system)
        meshVisibilityButton.setTitle("Mesh: ON", for: .normal)
        meshVisibilityButton.setTitleColor(.white, for: .normal)
        meshVisibilityButton.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.8)
        meshVisibilityButton.layer.cornerRadius = 8
        meshVisibilityButton.addTarget(
            self,
            action: #selector(toggleMeshVisibility),
            for: .touchUpInside
        )
        
        // Wireframe toggle button
        wireframeButton = UIButton(type: .system)
        wireframeButton.setTitle("Wireframe", for: .normal)
        wireframeButton.setTitleColor(.white, for: .normal)
        wireframeButton.backgroundColor = UIColor.systemPurple.withAlphaComponent(0.8)
        wireframeButton.layer.cornerRadius = 8
        wireframeButton.addTarget(
            self,
            action: #selector(toggleWireframe),
            for: .touchUpInside
        )
        
        // Add buttons to collision panel
        collisionPanel.addSubview(collisionToggleButton)
        collisionPanel.addSubview(meshVisibilityButton)
        collisionPanel.addSubview(wireframeButton)
        collisionPanel.addSubview(sdfScaleSlider)
        collisionPanel.addSubview(sdfScaleLabel)
        collisionPanel.addSubview(sdfYOffsetSlider)
        collisionPanel.addSubview(sdfYOffsetLabel)
        collisionPanel.addSubview(sdfYRotationSlider)
        collisionPanel.addSubview(sdfYRotationLabel)
        
        // Setup constraints
        collisionPanel.translatesAutoresizingMaskIntoConstraints = false
        collisionToggleButton.translatesAutoresizingMaskIntoConstraints = false
        meshVisibilityButton.translatesAutoresizingMaskIntoConstraints = false
        wireframeButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Collision panel constraints (right side)
            collisionPanel.topAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.topAnchor,
                constant: 20
            ),
            collisionPanel.trailingAnchor.constraint(
                equalTo: view.trailingAnchor,
                constant: -20
            ),
            collisionPanel.widthAnchor.constraint(equalToConstant: 200),
            
            // Collision toggle button constraints
            collisionToggleButton.topAnchor.constraint(
                equalTo: collisionPanel.topAnchor,
                constant: 10
            ),
            collisionToggleButton.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            collisionToggleButton.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            collisionToggleButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Mesh visibility button constraints
            meshVisibilityButton.topAnchor.constraint(
                equalTo: collisionToggleButton.bottomAnchor,
                constant: 10
            ),
            meshVisibilityButton.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            meshVisibilityButton.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            meshVisibilityButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Wireframe button constraints
            wireframeButton.topAnchor.constraint(
                equalTo: meshVisibilityButton.bottomAnchor,
                constant: 10
            ),
            wireframeButton.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            wireframeButton.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            wireframeButton.heightAnchor.constraint(equalToConstant: 40),
            
            // SDF scale label constraints
            sdfScaleLabel.topAnchor.constraint(
                equalTo: wireframeButton.bottomAnchor,
                constant: 10
            ),
            sdfScaleLabel.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            sdfScaleLabel.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            sdfScaleLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // SDF scale slider constraints
            sdfScaleSlider.topAnchor.constraint(
                equalTo: sdfScaleLabel.bottomAnchor,
                constant: 5
            ),
            sdfScaleSlider.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            sdfScaleSlider.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            sdfScaleSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // SDF Y offset label constraints
            sdfYOffsetLabel.topAnchor.constraint(
                equalTo: sdfScaleSlider.bottomAnchor,
                constant: 10
            ),
            sdfYOffsetLabel.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            sdfYOffsetLabel.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            sdfYOffsetLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // SDF Y offset slider constraints
            sdfYOffsetSlider.topAnchor.constraint(
                equalTo: sdfYOffsetLabel.bottomAnchor,
                constant: 5
            ),
            sdfYOffsetSlider.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            sdfYOffsetSlider.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            sdfYOffsetSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // SDF Y rotation label constraints
            sdfYRotationLabel.topAnchor.constraint(
                equalTo: sdfYOffsetSlider.bottomAnchor,
                constant: 10
            ),
            sdfYRotationLabel.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            sdfYRotationLabel.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            sdfYRotationLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // SDF Y rotation slider constraints
            sdfYRotationSlider.topAnchor.constraint(
                equalTo: sdfYRotationLabel.bottomAnchor,
                constant: 5
            ),
            sdfYRotationSlider.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            sdfYRotationSlider.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            sdfYRotationSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // Bottom constraint to define collision panel height
            collisionPanel.bottomAnchor.constraint(
                equalTo: sdfYRotationSlider.bottomAnchor,
                constant: 10
            ),
        ])
    }
    
    private func setupARPanel() {
        // Create AR control panel (bottom right)
        arPanel = UIView()
        arPanel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        arPanel.layer.cornerRadius = 10
        arPanel.clipsToBounds = true // Ensure content doesn't get clipped
        view.addSubview(arPanel)
        
        // AR toggle button
        arToggleButton = UIButton(type: .system)
        arToggleButton.setTitle("AR: OFF", for: .normal)
        arToggleButton.setTitleColor(.white, for: .normal)
        arToggleButton.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        arToggleButton.layer.cornerRadius = 8
        arToggleButton.addTarget(
            self,
            action: #selector(toggleAR),
            for: .touchUpInside
        )
        
        // Add controls to AR panel (bottom right)
        arPanel.addSubview(arToggleButton)
        arPanel.addSubview(transparentBackgroundToggle)
        
        
        // Setup constraints
        arPanel.translatesAutoresizingMaskIntoConstraints = false
        arToggleButton.translatesAutoresizingMaskIntoConstraints = false
        transparentBackgroundToggle.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // AR panel constraints (bottom right with fixed bottom anchor)
            arPanel.bottomAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.bottomAnchor,
                constant: -20
            ),
            arPanel.trailingAnchor.constraint(
                equalTo: view.trailingAnchor,
                constant: -20
            ),
            arPanel.widthAnchor.constraint(equalToConstant: 200),
            
            // AR toggle button constraints (top of panel)
            arToggleButton.topAnchor.constraint(
                equalTo: arPanel.topAnchor,
                constant: 10
            ),
            arToggleButton.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arToggleButton.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arToggleButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Transparent background toggle constraints (below AR toggle)
            transparentBackgroundToggle.topAnchor.constraint(
                equalTo: arToggleButton.bottomAnchor,
                constant: 10
            ),
            transparentBackgroundToggle.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            transparentBackgroundToggle.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            transparentBackgroundToggle.heightAnchor.constraint(equalToConstant: 40),
        ])
        
        // Create and manage panel height constraint (adjusted dynamically)
        arPanelHeightConstraint = arPanel.heightAnchor.constraint(equalToConstant: 60)
        arPanelHeightConstraint.isActive = true
        
        // Initialize AR panel visibility
        updateTransparentBackgroundToggleVisibility()
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
    
    @objc private func particleCountChanged(_ slider: UISlider) {
        let count = Int(slider.value)
        particleCountLabel.text = "Particles: \(count)"
        fluidRenderer.setParticleCount(count)
    }
    
    @objc private func gridSizeChanged(_ slider: UISlider) {
        let size = Int(slider.value)
        gridSizeLabel.text = "Grid: \(size)³"
        fluidRenderer.setGridSize(size)
    }
    
    @objc private func sdfScaleChanged(_ slider: UISlider) {
        let scale = slider.value
        sdfScaleLabel.text = String(format: "SDF Scale: %.2fx", scale)
        updateSdfTransfom()
    }
    
    @objc private func sdfYOffsetChanged(_ slider: UISlider) {
        let offset = slider.value
        sdfYOffsetLabel.text = String(format: "SDF Y Offset: %.1f", offset)
        updateSdfTransfom()
    }
    
    @objc private func sdfYRotationChanged(_ slider: UISlider) {
        let rotation = slider.value
        sdfYRotationLabel.text = String(format: "SDF Y Rotation: %.0f°", rotation)
        updateSdfTransfom()
    }
    private func updateSdfTransfom(){
        let scale = sdfScaleSlider.value
        let offset = sdfYOffsetSlider.value
        let rotation = sdfYRotationSlider.value
        fluidRenderer.collisionManager?.representativeItem.updateSdfTransform(scale: SIMD3<Float>(scale, scale, scale), translate:SIMD3<Float>(0.0, offset, 0.0),rotate: SIMD3<Float>(0.0, rotation * Float.pi / 180.0, 0.0))
    }
    
    @objc private func toggleRenderMode() {
        guard let renderer = fluidRenderer else { return }
        renderer.toggleRenderMode()
        
        // If using default solid background, update color to mode-appropriate
        if renderer.backgroundRenderer == nil || renderer.backgroundRenderer is SolidColorBackgroundRenderer {
            installDefaultBackgroundRenderer()
        }
        
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

    // MARK: - Background Color Controls
    public var defaultBackgroundColorParticles: MTLClearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
    public var defaultBackgroundColorWater: MTLClearColor = MTLClearColor(red: 0.8, green: 0.8, blue: 0.8, alpha: 1.0)
    private func installDefaultBackgroundRenderer() {
        guard let fluidRenderer = fluidRenderer else { return }
        let color = (fluidRenderer.currentRenderMode == .water) ?
            defaultBackgroundColorWater :
            defaultBackgroundColorParticles
        // Update in place if already using a solid color background
        if let solid = fluidRenderer.backgroundRenderer as? SolidColorBackgroundRenderer {
            solid.clearColor = color
        } else {
            fluidRenderer.backgroundRenderer = SolidColorBackgroundRenderer(color: color, transparent: false)
        }
        // Also reflect on MTKView as a fallback when no background renderer is active
        metalView?.clearColor = color
    }

    @objc private func toggleMaterialMode() {
        guard let renderer = fluidRenderer else { return }
        
        // Cycle through material modes
        switch renderer.materialParameters.currentMaterialMode {
        case .fluid:
            renderer.materialParameters.currentMaterialMode = .neoHookeanElastic
            materialModeButton.setTitle("Elastic", for: .normal)
            materialModeButton.backgroundColor = UIColor.systemYellow.withAlphaComponent(0.8)
        case .neoHookeanElastic:
            renderer.materialParameters.currentMaterialMode = .fluid
            materialModeButton.setTitle("Fluid", for: .normal)
            materialModeButton.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.8)
        }
        
        // Reset simulation when material mode changes
        renderer.reset()
        print("🔄 Material mode changed to: \(renderer.materialParameters.currentMaterialMode)")
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
        // Reset simulation to clean state first
        fluidRenderer.reset()
        
        // Then re-apply current slider values by calling their handlers
        sdfScaleChanged(sdfScaleSlider)
        sdfYOffsetChanged(sdfYOffsetSlider)
        sdfYRotationChanged(sdfYRotationSlider)
    }
    
    // MARK: - Collision Control Actions
    
    @objc private func toggleCollision() {
        guard let isEnabled = fluidRenderer.collisionManager?.representativeItem.isEnabled() else {
            return
        }
        fluidRenderer.collisionManager?.representativeItem.setEnabled(!isEnabled)
        
        if !isEnabled{
            collisionToggleButton.setTitle("Collision: ON", for: .normal)
            collisionToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            collisionToggleButton.setTitle("Collision: OFF", for: .normal)
            collisionToggleButton.backgroundColor = UIColor.systemRed.withAlphaComponent(0.8)
        }
    }
        
    @objc private func toggleMeshVisibility() {
        guard let isVisible = fluidRenderer.collisionManager?.isMeshVisible() else {
            return
        }
        fluidRenderer.collisionManager?.setMeshVisible(!isVisible)
        
        if !isVisible {
            meshVisibilityButton.setTitle("Mesh: ON", for: .normal)
            meshVisibilityButton.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.8)
        } else {
            meshVisibilityButton.setTitle("Mesh: OFF", for: .normal)
            meshVisibilityButton.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        }
    }
    
    @objc private func toggleWireframe() {
        // Toggle between wireframe and solid rendering
        isWireframeMode.toggle()
        
        fluidRenderer.collisionManager?.representativeItem.setMeshWireframe(isWireframeMode)
        
        if isWireframeMode {
            wireframeButton.setTitle("Solid", for: .normal)
            wireframeButton.backgroundColor = UIColor.systemIndigo.withAlphaComponent(0.8)
        } else {
            wireframeButton.setTitle("Wireframe", for: .normal)
            wireframeButton.backgroundColor = UIColor.systemPurple.withAlphaComponent(0.8)
        }
    }
   

    // MARK: - World Transform Controls (for AR placement)
    public func setFluidWorldTranslation(_ t: SIMD3<Float>) {
        // Update only the translation coefficients
        worldTranslation = t
    }
    
    public func setFluidWorldRotationEuler(yaw: Float, pitch: Float, roll: Float) {
        // Update rotation coefficients (ignoring roll for simplicity in orbit controls)
        worldYaw = yaw
        worldPitch = pitch
    }
    
    public func setFluidWorldTransform(_ transform: float4x4) {
        // Extract translation and rotation from transform matrix
        worldTranslation = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        
        // Extract scale (magnitude of basis vectors)
        let scaleX = simd_length(SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z))
        let scaleY = simd_length(SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z))
        let scaleZ = simd_length(SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z))
        worldScale = (scaleX + scaleY + scaleZ) / 3.0 // Average scale
        
        // Extract rotation (normalized after removing scale)
        let normalizedForward = simd_normalize(SIMD3<Float>(-transform.columns.2.x, -transform.columns.2.y, -transform.columns.2.z))
        worldYaw = atan2(normalizedForward.x, normalizedForward.z)
        worldPitch = asin(normalizedForward.y)
    }
    
    public func getFluidWorldTransform() -> float4x4 {
        // Return computed transform without applying to renderer
        return computeWorldTransform()
    }
    @objc private func toggleTransparentBackground() {
        // Swap between AR background and existing background renderer
        guard let fluidRenderer = fluidRenderer else { return }
        if fluidRenderer.backgroundRenderer is ARBackgroundRendererAdapter {
            // Swap back to saved (or default) background
            if let saved = savedBackgroundRenderer {
                fluidRenderer.backgroundRenderer = saved
            } else {
                installDefaultBackgroundRenderer()
            }
            transparentBackgroundToggle.setTitle("AR BG: OFF", for: .normal)
            transparentBackgroundToggle.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        } else {
            // Ensure AR renderer exists
            if arRenderer == nil {
                arRenderer = ARRenderer(device: fluidRenderer.device, commandQueue: fluidRenderer.commandQueue)
            }
            guard let arRenderer = arRenderer else { return }
            // Save current background and swap to AR background
            savedBackgroundRenderer = fluidRenderer.backgroundRenderer
            arRenderer.startARSession()
            fluidRenderer.backgroundRenderer = ARBackgroundRendererAdapter(arRenderer: arRenderer, isTransparent: true)
            transparentBackgroundToggle.setTitle("AR BG: ON", for: .normal)
            transparentBackgroundToggle.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        }
    }
    
    private var isAREnabled: Bool = false
    private var arRenderer: ARRenderer?
    private var savedBackgroundRenderer: BackgroundRenderer?
    

    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: metalView)

        switch gesture.state {
        case .began:
            lastPanPoint = gesture.location(in: metalView)

        case .changed:
            if isAREnabled {
                return
            }
            // One-finger: world yaw + pitch (trackball-like)
            let deltaX = Float(translation.x) * rotateSpeed
            let deltaY = Float(translation.y) * rotateSpeed
            worldYaw -= deltaX
            worldPitch += deltaY
            worldPitch = max(minPitch, min(maxPitch, worldPitch))
            // Just update coefficients, no direct application to renderer
            gesture.setTranslation(.zero, in: metalView)
        default:
            break
        }
    }

    @objc private func handleTwoFingerPanGesture(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: metalView)
        switch gesture.state {
        case .changed:
            if isAREnabled {
                return
            }
            var baseViewMatrix = getViewMatrix()
            let invView = baseViewMatrix.inverse
            let right = SIMD3<Float>(invView.columns.0.x, invView.columns.0.y, invView.columns.0.z)
            let up = SIMD3<Float>(invView.columns.1.x, invView.columns.1.y, invView.columns.1.z)

            // Two-finger: translate world in screen plane
            let dx = Float(translation.x) * worldPanSpeed
            let dy = Float(-translation.y) * worldPanSpeed

            // Update translation coefficients directly
            worldTranslation = worldTranslation + right * dx + up * dy
            gesture.setTranslation(.zero, in: metalView)
        default:
            break
        }
    }

    @objc private func handlePinchGesture(_ gesture: UIPinchGestureRecognizer) {
        switch gesture.state {
        case .began:
            break
        case .changed:
            if isAREnabled {
                return
            }
            // Scale the scene instead of translating
            let scaleSpeed: Float = 0.01
            let scale = Float(gesture.scale)
            let scaleDelta = (scale - 1.0) * scaleSpeed
            
            // Update scale coefficient with limits
            worldScale = max(0.1, min(10.0, worldScale + scaleDelta))
            
            gesture.scale = 1.0
        case .ended, .cancelled, .failed:
            break
        default:
            break
        }
    }

    @objc private func handleTapGesture(_ gesture: UITapGestureRecognizer) {
        let tapPoint = gesture.location(in: metalView)

        if isAREnabled {
            // AR Mode: Perform raycast against AR mesh
            handleARTap(at: tapPoint)
        } else {
            // Non-AR Mode: Original 2D fluid interaction
            let normalizedX = Float(tapPoint.x / metalView.bounds.width) * 2.0 - 1.0
            let normalizedY = Float(1.0 - tapPoint.y / metalView.bounds.height) * 2.0 - 1.0

            let worldPosition = SIMD2<Float>(normalizedX, normalizedY)
            let force = SIMD2<Float>(
                Float.random(in: -5.0...5.0),
                Float.random(in: -2.0...8.0)
            )

            fluidRenderer.addForce(at: worldPosition, force: force)
        }
    }
    
    private func handleARTap(at tapPoint: CGPoint) {
        #if canImport(ARKit)
        guard let arRenderer = arRenderer,
              let orientation = metalView.window?.windowScene?.interfaceOrientation else {
            print("AR renderer or orientation not available")
            return
        }
        
        if #available(iOS 11.0, macOS 10.13, *) {
            // Perform GPU raycast and generate SDF
            if let sdfTexture = arRenderer.generateSDFFromTapPositionGPU(
                tapPoint: tapPoint,
                viewportSize: metalView.bounds.size,
                orientation: orientation,
                boundingBoxSize: 0.3 // 30cm bounding box
            ) {
                print("✅ Successfully generated SDF from AR mesh at tap position")
                
                // Here you can use the SDF texture for collision detection
                // For example, integrate with collision manager:
                // fluidRenderer.collisionManager?.setCustomSDF(sdfTexture)
                
                // Show visual feedback
                showARTapFeedback(at: tapPoint)
            } else {
                print("❌ Failed to generate SDF from tap position")
            }
        }
        #endif
    }
    
    private func showARTapFeedback(at tapPoint: CGPoint) {
        // Create visual feedback for successful AR tap
        let feedbackView = UIView(frame: CGRect(x: tapPoint.x - 25, y: tapPoint.y - 25, width: 50, height: 50))
        feedbackView.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.6)
        feedbackView.layer.cornerRadius = 25
        feedbackView.layer.borderWidth = 2
        feedbackView.layer.borderColor = UIColor.white.cgColor
        metalView.addSubview(feedbackView)
        
        // Animate feedback
        UIView.animate(withDuration: 0.8, animations: {
            feedbackView.alpha = 0
            feedbackView.transform = CGAffineTransform(scaleX: 2.0, y: 2.0)
        }) { _ in
            feedbackView.removeFromSuperview()
        }
    }

    @objc private func handleDoubleTapGesture(_ gesture: UITapGestureRecognizer)
    {
        // Reset world transform rotation and scale (keep translation)
        worldYaw = 0.0
        worldPitch = 0.0
        worldScale = 1.0
        // Coefficients updated, no direct application to renderer
    }

    private func updateTransparentBackgroundToggleVisibility() {
        let isAREnabled = self.isAREnabled
        
        print("🔍 AR State: \(isAREnabled), Transparent toggle hidden: \(!isAREnabled)")
        
        // Show/hide transparent background toggle based on AR state
        transparentBackgroundToggle.isHidden = !isAREnabled
        
        // Update AR panel height based on visibility
        // Elements: AR button (always), Transparent (AR ON only), World Orbit (always)
        // Height formula: top padding (10) + sum(button heights + inner spacings) + bottom padding (10)
        // - AR OFF: AR(40) + spacing(10) + World(40) + paddings(20) = 110
        // - AR ON:  AR(40) + spacing(10) + Transparent(40) + spacing(10) + World(40) + paddings(20) = 160
        arPanelHeightConstraint.constant = isAREnabled ? 160 : 110
        
        print("📏 AR Panel height set to: \(arPanelHeightConstraint.constant)")
        
        // Animate the height change
        UIView.animate(withDuration: 0.3) {
            self.view.layoutIfNeeded()
        }
    }
    
    private func getCameraPosition() -> SIMD3<Float> {
        return SIMD3<Float>(0,0,0)
    }

    private func getViewMatrix() -> float4x4 {
        return matrix_identity_float4x4
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
extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle size changes if needed
    }

    func draw(in view: MTKView) {
        // Apply transform coefficients to renderer only when rendering
        updateRendererTransformIfNeeded()
        
        let currentTime = CACurrentMediaTime()
        let deltaTime = Float(currentTime - lastFrameTime)
        lastFrameTime = currentTime

        // Only compute at 10fps equivalent
        let computeInterval: CFTimeInterval = 0.05  // 20fps
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

        // Create transformation matrices (use AR camera when AR is enabled)
        var baseViewMatrix = getViewMatrix()
        var projectionMatrix = getProjectionMatrix(aspectRatio: aspectRatio)
        if isAREnabled {
            #if canImport(ARKit)
            if #available(iOS 11.0, macOS 10.13, *) {
                let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
                if let (proj, viewM) = arRenderer?.getCameraMatrices(viewportSize: metalView.bounds.size, orientation: orientation) {
                    projectionMatrix = proj
                    baseViewMatrix = viewM
                }
            }
            #endif
        }
        // Complete view matrix: View * Model(world) * Scale(sim units->world) * Centering
        let viewMatrix = baseViewMatrix * fluidRenderer.worldTransform * fluidRenderer.getGridToLocalTransform()
        // Update simulation based on mode
        if performStep {
            fluidRenderer.update(
                deltaTime: deltaTime,
                screenSize: screenSize,
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

// Allow simultaneous recognition for pinch and two-finger pan to improve UX.
extension ViewController: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }
}

// MARK: - Keyboard Handling and ReplayKit Recording
extension ViewController {
    
    private func setupKeyboardHandling() {
        // Make the view controller first responder to receive key events
        view.becomeFirstResponder()
    }
    
    override var canBecomeFirstResponder: Bool {
        return true
    }
    
    override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        for press in presses {
            guard let key = press.key else { continue }
            
            // Check for Shift+G
            if key.charactersIgnoringModifiers == recordKey && key.modifierFlags.contains(.shift) {
                toggleRecording()
                return
            }
        }
        
        super.pressesBegan(presses, with: event)
    }
    
    private func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }
    
    private func startRecording() {
        guard !isRecording else { return }
        fluidRenderer.reset()
        screenRecorder.startRecording { [weak self] error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Failed to start recording: \(error.localizedDescription)")
                    self?.showAlert(title: "Recording Error", 
                                   message: "Failed to start screen recording: \(error.localizedDescription)")
                } else {
                    self?.isRecording = true
                    print("🎥 Screen recording started - Press Shift+G to stop")
                }
            }
        }
    }
    
    private func stopRecording() {
        guard isRecording else { return }
        
        screenRecorder.stopRecording { [weak self] previewController, error in
            DispatchQueue.main.async {
                self?.isRecording = false
                
                if let error = error {
                    print("Failed to stop recording: \(error.localizedDescription)")
                    self?.showAlert(title: "Recording Error", 
                                   message: "Failed to stop recording: \(error.localizedDescription)")
                    return
                }
                
                guard let previewController = previewController else {
                    print("No preview controller available")
                    return
                }
                
                previewController.previewControllerDelegate = self
                self?.present(previewController, animated: true) {
                    print("🎥 Screen recording stopped - Preview presented")
                }
            }
        }
    }
    
    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - RPPreviewViewControllerDelegate
extension ViewController: RPPreviewViewControllerDelegate {
    func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
        previewController.dismiss(animated: true) {
            print("📱 Recording preview dismissed")
        }
    }
}
extension ViewController{
    @objc private func toggleAR() {
        isAREnabled.toggle()
        if isAREnabled {
            // Start AR session and install AR background adapter
            if arRenderer == nil, let fr = fluidRenderer {
                arRenderer = ARRenderer(device: fr.device, commandQueue: fr.commandQueue)
            }
            if let ar = arRenderer {
                ar.startARSession()
                savedBackgroundRenderer = fluidRenderer.backgroundRenderer
                fluidRenderer.backgroundRenderer = ARBackgroundRendererAdapter(arRenderer: ar, isTransparent: true)
            }
            arToggleButton.setTitle("AR: ON", for: .normal)
            arToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            // Stop AR session and detach background renderer
            arRenderer?.stopARSession()
            if let saved = savedBackgroundRenderer {
                fluidRenderer.backgroundRenderer = saved
                savedBackgroundRenderer = nil
            } else {
                installDefaultBackgroundRenderer()
            }
            arToggleButton.setTitle("AR: OFF", for: .normal)
            arToggleButton.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        }
        onARStateChanged()
    }
    
    // Called when AR state changes
    func onARStateChanged() {
        DispatchQueue.main.async {
            self.updateTransparentBackgroundToggleVisibility()
        }
    }
    // MARK: - AR Controls
    public func startAR() {
        #if canImport(ARKit)
        if #available(iOS 11.0, macOS 10.13, *) {
            arRenderer?.startARSession()
        }
        #endif
    }
    
    public func stopAR() {
        #if canImport(ARKit)
        if #available(iOS 11.0, macOS 10.13, *) {
            arRenderer?.stopARSession()
        }
        #endif
    }
}
