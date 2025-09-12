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
    internal var lastComputeTime: CFTimeInterval = 0
    
    internal var metalView: MTKView!
    internal var fluidRenderer: MPMFluidRenderer!
    internal var integratedRenderer: IntegratedRenderer!
    internal var lastFrameTime: CFTimeInterval = 0
    
    // Camera properties for 3D viewing
    internal var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
    
    // Interaction
    internal var lastPanPoint: CGPoint = .zero
    internal var isInteracting = false
    
    // Simulation control
    internal var isAutoMode: Bool = true
    internal var shouldStep: Bool = false
    
    // Wireframe state tracking
    internal var isWireframeMode: Bool = false
    
    // Transform coefficients (managed in VC, applied when needed)
    internal var worldTranslation: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, 0.0)
    internal var worldTranslationOffset: SIMD3<Float> = SIMD3<Float>(0.0, 0.0, 0.0)  // UI offset from base translation
    internal var worldYaw: Float = 0.0
    internal var worldPitch: Float = 0.0
    internal var worldScale: Float = 1.0
    
    // Orbit control tuning
    private let rotateSpeed: Float = 0.01
    private let worldPanSpeed: Float = 0.003
    private let minPitch: Float = -Float.pi/2 + 0.1
    private let maxPitch: Float =  Float.pi/2 - 0.1
    
    // Initial values (shared between sliders and FluidRenderer)
    private let initialParticleCount: Float = 40000
    private let initialGridSize: Float = 64
    private let initialGridHeightMultiplier: Float = 1.0
    
    // UI Elements
    internal var controlPanel: UIStackView!
    internal var modeButton: UIButton!
    internal var stepButton: UIButton!
    internal var resetButton: UIButton!
    internal var renderModeButton: UIButton!
    internal var particleSizeSlider: UISlider!
    internal var particleSizeLabel: UILabel!
    internal var massScaleSlider: UISlider!
    internal var massScaleLabel: UILabel!
    internal var cubeScaleSlider: UISlider!
    internal var cubeScaleLabel: UILabel!
    internal var materialScaleStackView: UIStackView!
    internal var particleCountSlider: UISlider!
    internal var particleCountLabel: UILabel!
    internal var gridSizeSlider: UISlider!
    internal var gridSizeLabel: UILabel!
    internal var gridSpacingSlider: UISlider!
    internal var gridSpacingLabel: UILabel!
    internal var sdfScaleSlider: UISlider!
    internal var sdfScaleLabel: UILabel!
    internal var sdfYOffsetSlider: UISlider!
    internal var sdfYOffsetLabel: UILabel!
    internal var sdfYRotationSlider: UISlider!
    internal var sdfYRotationLabel: UILabel!
    
    // World translate controls
    internal var worldTranslateXSlider: UISlider!
    internal var worldTranslateXLabel: UILabel!
    internal var worldTranslateYSlider: UISlider!
    internal var worldTranslateYLabel: UILabel!
    internal var worldTranslateZSlider: UISlider!
    internal var worldTranslateZLabel: UILabel!
    
    
    // Collision controls panel (right side)
    internal var collisionPanel: UIStackView!
    internal var collisionToggleButton: UIButton!
    internal var meshVisibilityButton: UIButton!
    internal var wireframeButton: UIButton!
    internal var materialModeButton: UIButton!
    
    // AR controls panel (bottom right)
    internal var arPanel: UIView!
    internal var arToggleButton: UIButton!
    internal var arMeshToggleButton: UIButton!
    internal var arCameraToggleButton: UIButton!
    internal var arCollisionRenderCollectionView: UICollectionView!
    internal var arCollisionRenderLayout: UICollectionViewFlowLayout!
    internal var arCollisionTitleLabel: UILabel!
    internal var arBBRadiusSlider: UISlider!
    internal var arBBRadiusLabel: UILabel!
    internal var arPanelHeightConstraint: NSLayoutConstraint!
    
    // ReplayKit recording
    private let screenRecorder = RPScreenRecorder.shared()
    internal var isRecording = false
    internal let meshLoader: MeshLoader = MeshLoader(scaleFactor: 1.0)
    
    public func computeWorldTransform() -> float4x4 {
        // Compute transform from current coefficients: T * R * S order
        let S = float4x4(scale: SIMD3<Float>(worldScale, worldScale, worldScale))
        let R = float4x4(rotationY: worldYaw) * float4x4(rotationX: worldPitch)
        
        // Add domain center offset so fluid domain is centered at worldTranslation + worldTranslationOffset
        let domainCenter = fluidRenderer.getDomainBasePosition()
        let finalTranslation = worldTranslation + worldTranslationOffset
        let adjustedTranslation = finalTranslation - domainCenter
        let T = float4x4(translation: adjustedTranslation)
        
        return T * R * S
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
        setupBunny()
        
        // Sync grid spacing multiplier slider with renderer default
        gridSpacingSlider.value = fluidRenderer.getGridSpacingMultiplier()
        gridSpacingLabel.text = String(format: "Spacing: %.1fx", fluidRenderer.getGridSpacingMultiplier())
        
        // Initialize material mode UI
        updateMaterialModeUI()
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
        
        // Enable triple buffering
        if let layer = metalView.layer as? CAMetalLayer {
            layer.maximumDrawableCount = 3
        }
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
        
        integratedRenderer = IntegratedRenderer(
            device: fluidRenderer.device,
            commandQueue: fluidRenderer.commandQueue,
            fluidRenderer: fluidRenderer
        )
        
        // Install default background renderer
        installDefaultBackgroundRenderer()
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
        controlPanel = UIStackView()
        controlPanel.axis = .vertical
        controlPanel.distribution = .fill
        controlPanel.alignment = .fill
        controlPanel.spacing = 10
        controlPanel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        controlPanel.layer.cornerRadius = 10
        controlPanel.isLayoutMarginsRelativeArrangement = true
        controlPanel.layoutMargins = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
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
        
        // Cube scale slider (for Elastic mode)
        cubeScaleSlider = UISlider()
        cubeScaleSlider.minimumValue = 0.1
        cubeScaleSlider.maximumValue = 1.5
        cubeScaleSlider.value = 0.8
        cubeScaleSlider.addTarget(
            self,
            action: #selector(cubeScaleChanged),
            for: .valueChanged
        )
        
        // Cube scale label
        cubeScaleLabel = UILabel()
        cubeScaleLabel.text = "Cube: 0.8x"
        cubeScaleLabel.textColor = .white
        cubeScaleLabel.font = UIFont.systemFont(ofSize: 14)
        cubeScaleLabel.textAlignment = .center
        
        // Material scale stack view (contains mass and cube controls)
        materialScaleStackView = UIStackView()
        materialScaleStackView.axis = .vertical
        materialScaleStackView.distribution = .fill
        materialScaleStackView.alignment = .fill
        materialScaleStackView.spacing = 10
        
        // Height constraints are set in the main NSLayoutConstraint.activate section
        
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
        
        // Grid spacing multiplier slider
        gridSpacingSlider = UISlider()
        gridSpacingSlider.minimumValue = 0.1
        gridSpacingSlider.maximumValue = 5.0
        gridSpacingSlider.value = 1.0  // Default multiplier
        gridSpacingSlider.addTarget(
            self,
            action: #selector(gridSpacingChanged),
            for: .valueChanged
        )
        
        // Grid spacing multiplier label
        gridSpacingLabel = UILabel()
        gridSpacingLabel.text = String(format: "Spacing: %.1fx", gridSpacingSlider.value)
        gridSpacingLabel.textColor = .white
        gridSpacingLabel.font = UIFont.systemFont(ofSize: 14)
        gridSpacingLabel.textAlignment = .center
        
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
        
        
        // SDF Y rotation label
        sdfYRotationLabel = UILabel()
        sdfYRotationLabel.text = String(format: "SDF Y Rotation: %.0f°", sdfYRotationSlider.value)
        sdfYRotationLabel.textColor = .white
        sdfYRotationLabel.font = UIFont.systemFont(ofSize: 14)
        sdfYRotationLabel.textAlignment = .center
        
        // World translate X slider
        worldTranslateXSlider = UISlider()
        worldTranslateXSlider.minimumValue = -5.0
        worldTranslateXSlider.maximumValue = 5.0
        worldTranslateXSlider.value = 0.0
        worldTranslateXSlider.addTarget(
            self,
            action: #selector(worldTranslateXChanged),
            for: .valueChanged
        )
        
        // World translate X label
        worldTranslateXLabel = UILabel()
        worldTranslateXLabel.text = String(format: "World X: %.1f", worldTranslateXSlider.value)
        worldTranslateXLabel.textColor = .white
        worldTranslateXLabel.font = UIFont.systemFont(ofSize: 14)
        worldTranslateXLabel.textAlignment = .center
        
        // World translate Y slider
        worldTranslateYSlider = UISlider()
        worldTranslateYSlider.minimumValue = -5.0
        worldTranslateYSlider.maximumValue = 5.0
        worldTranslateYSlider.value = 0.0
        worldTranslateYSlider.addTarget(
            self,
            action: #selector(worldTranslateYChanged),
            for: .valueChanged
        )
        
        // World translate Y label
        worldTranslateYLabel = UILabel()
        worldTranslateYLabel.text = String(format: "World Y: %.1f", worldTranslateYSlider.value)
        worldTranslateYLabel.textColor = .white
        worldTranslateYLabel.font = UIFont.systemFont(ofSize: 14)
        worldTranslateYLabel.textAlignment = .center
        
        // World translate Z slider
        worldTranslateZSlider = UISlider()
        worldTranslateZSlider.minimumValue = -5.0
        worldTranslateZSlider.maximumValue = 5.0
        worldTranslateZSlider.value = 0.0
        worldTranslateZSlider.addTarget(
            self,
            action: #selector(worldTranslateZChanged),
            for: .valueChanged
        )
        
        // World translate Z label
        worldTranslateZLabel = UILabel()
        worldTranslateZLabel.text = String(format: "World Z: %.1f", worldTranslateZSlider.value)
        worldTranslateZLabel.textColor = .white
        worldTranslateZLabel.font = UIFont.systemFont(ofSize: 14)
        worldTranslateZLabel.textAlignment = .center
        
        controlPanel.addArrangedSubview(modeButton)
        controlPanel.addArrangedSubview(stepButton)
        controlPanel.addArrangedSubview(resetButton)
        controlPanel.addArrangedSubview(renderModeButton)
        controlPanel.addArrangedSubview(materialModeButton)
        controlPanel.addArrangedSubview(particleSizeLabel)
        controlPanel.addArrangedSubview(particleSizeSlider)
        controlPanel.addArrangedSubview(materialScaleStackView)
        controlPanel.addArrangedSubview(particleCountLabel)
        controlPanel.addArrangedSubview(particleCountSlider)
        controlPanel.addArrangedSubview(gridSizeLabel)
        controlPanel.addArrangedSubview(gridSizeSlider)
        controlPanel.addArrangedSubview(gridSpacingLabel)
        controlPanel.addArrangedSubview(gridSpacingSlider)
        controlPanel.addArrangedSubview(worldTranslateXLabel)
        controlPanel.addArrangedSubview(worldTranslateXSlider)
        controlPanel.addArrangedSubview(worldTranslateYLabel)
        controlPanel.addArrangedSubview(worldTranslateYSlider)
        controlPanel.addArrangedSubview(worldTranslateZLabel)
        controlPanel.addArrangedSubview(worldTranslateZSlider)
        
        controlPanel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Control panel constraints (only position and width needed for StackView)
            controlPanel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            controlPanel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            controlPanel.widthAnchor.constraint(equalToConstant: 200),
            
            // Set heights for buttons and sliders
            modeButton.heightAnchor.constraint(equalToConstant: 40),
            stepButton.heightAnchor.constraint(equalToConstant: 40),
            resetButton.heightAnchor.constraint(equalToConstant: 40),
            renderModeButton.heightAnchor.constraint(equalToConstant: 40),
            materialModeButton.heightAnchor.constraint(equalToConstant: 40),
            particleSizeLabel.heightAnchor.constraint(equalToConstant: 20),
            particleSizeSlider.heightAnchor.constraint(equalToConstant: 30),
            particleCountLabel.heightAnchor.constraint(equalToConstant: 20),
            particleCountSlider.heightAnchor.constraint(equalToConstant: 30),
            gridSizeLabel.heightAnchor.constraint(equalToConstant: 20),
            gridSizeSlider.heightAnchor.constraint(equalToConstant: 30),
            gridSpacingLabel.heightAnchor.constraint(equalToConstant: 20),
            gridSpacingSlider.heightAnchor.constraint(equalToConstant: 30),
            worldTranslateXLabel.heightAnchor.constraint(equalToConstant: 20),
            worldTranslateXSlider.heightAnchor.constraint(equalToConstant: 30),
            worldTranslateYLabel.heightAnchor.constraint(equalToConstant: 20),
            worldTranslateYSlider.heightAnchor.constraint(equalToConstant: 30),
            worldTranslateZLabel.heightAnchor.constraint(equalToConstant: 20),
            worldTranslateZSlider.heightAnchor.constraint(equalToConstant: 30),
            
            // Material scale elements (mass/cube)
            massScaleLabel.heightAnchor.constraint(equalToConstant: 20),
            massScaleSlider.heightAnchor.constraint(equalToConstant: 30),
            cubeScaleLabel.heightAnchor.constraint(equalToConstant: 20),
            cubeScaleSlider.heightAnchor.constraint(equalToConstant: 30)
        ])
    }
    
    private func setupCollisionPanel() {
        collisionPanel = UIStackView()
        collisionPanel.axis = .vertical
        collisionPanel.distribution = .fill
        collisionPanel.alignment = .fill
        collisionPanel.spacing = 10
        collisionPanel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        collisionPanel.layer.cornerRadius = 10
        collisionPanel.isLayoutMarginsRelativeArrangement = true
        collisionPanel.layoutMargins = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
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
        collisionPanel.addArrangedSubview(collisionToggleButton)
        collisionPanel.addArrangedSubview(meshVisibilityButton)
        collisionPanel.addArrangedSubview(wireframeButton)
        collisionPanel.addArrangedSubview(sdfScaleLabel)
        collisionPanel.addArrangedSubview(sdfScaleSlider)
        collisionPanel.addArrangedSubview(sdfYOffsetLabel)
        collisionPanel.addArrangedSubview(sdfYOffsetSlider)
        collisionPanel.addArrangedSubview(sdfYRotationLabel)
        collisionPanel.addArrangedSubview(sdfYRotationSlider)
        
        collisionPanel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Collision panel constraints (only position and width needed for StackView)
            collisionPanel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            collisionPanel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            collisionPanel.widthAnchor.constraint(equalToConstant: 200),
            
            // Set heights for buttons and sliders
            collisionToggleButton.heightAnchor.constraint(equalToConstant: 40),
            meshVisibilityButton.heightAnchor.constraint(equalToConstant: 40),
            wireframeButton.heightAnchor.constraint(equalToConstant: 40),
            sdfScaleLabel.heightAnchor.constraint(equalToConstant: 20),
            sdfScaleSlider.heightAnchor.constraint(equalToConstant: 30),
            sdfYOffsetLabel.heightAnchor.constraint(equalToConstant: 20),
            sdfYOffsetSlider.heightAnchor.constraint(equalToConstant: 30),
            sdfYRotationLabel.heightAnchor.constraint(equalToConstant: 20),
            sdfYRotationSlider.heightAnchor.constraint(equalToConstant: 30)
        ])
    }
    
    private func setupARPanel() {
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
        
        // AR Mesh toggle button
        arMeshToggleButton = UIButton(type: .system)
        arMeshToggleButton.setTitle("Mesh: ON", for: .normal)
        arMeshToggleButton.setTitleColor(.white, for: .normal)
        arMeshToggleButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        arMeshToggleButton.layer.cornerRadius = 8
        arMeshToggleButton.addTarget(
            self,
            action: #selector(toggleARMesh),
            for: .touchUpInside
        )
        
        // AR Camera toggle button
        arCameraToggleButton = UIButton(type: .system)
        arCameraToggleButton.setTitle("Camera: ON", for: .normal)
        arCameraToggleButton.setTitleColor(.white, for: .normal)
        arCameraToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        arCameraToggleButton.layer.cornerRadius = 8
        arCameraToggleButton.addTarget(
            self,
            action: #selector(toggleARCamera),
            for: .touchUpInside
        )
        
        // AR Collision Render collection view
        arCollisionRenderLayout = UICollectionViewFlowLayout()
        arCollisionRenderLayout.scrollDirection = .horizontal
        arCollisionRenderLayout.minimumLineSpacing = 8
        arCollisionRenderLayout.minimumInteritemSpacing = 8
        arCollisionRenderLayout.itemSize = CGSize(width: 60, height: 36)
        
        arCollisionRenderCollectionView = UICollectionView(frame: .zero, collectionViewLayout: arCollisionRenderLayout)
        arCollisionRenderCollectionView.backgroundColor = UIColor.clear
        arCollisionRenderCollectionView.showsHorizontalScrollIndicator = false
        arCollisionRenderCollectionView.allowsMultipleSelection = true
        arCollisionRenderCollectionView.delegate = self
        arCollisionRenderCollectionView.dataSource = self
        arCollisionRenderCollectionView.register(CollisionRenderCell.self, forCellWithReuseIdentifier: "CollisionRenderCell")
        
        // AR Collision Title label
        arCollisionTitleLabel = UILabel()
        arCollisionTitleLabel.text = "Active Collision:"
        arCollisionTitleLabel.textColor = .white
        arCollisionTitleLabel.font = UIFont.systemFont(ofSize: 14, weight: .medium)
        arCollisionTitleLabel.textAlignment = .left
        
        // AR Bounding Box Radius slider
        arBBRadiusSlider = UISlider()
        arBBRadiusSlider.minimumValue = 0.1
        arBBRadiusSlider.maximumValue = 2.0
        arBBRadiusSlider.value = arBoundingBoxRadius
        arBBRadiusSlider.addTarget(
            self,
            action: #selector(arBBRadiusChanged),
            for: .valueChanged
        )
        
        // AR Bounding Box Radius label
        arBBRadiusLabel = UILabel()
        arBBRadiusLabel.text = String(format: "BB: %.1fm", arBoundingBoxRadius)
        arBBRadiusLabel.textColor = .white
        arBBRadiusLabel.font = UIFont.systemFont(ofSize: 14)
        arBBRadiusLabel.textAlignment = .center
        
        // Add controls to AR panel (bottom right)
        arPanel.addSubview(arToggleButton)
        arPanel.addSubview(arMeshToggleButton)
        arPanel.addSubview(arCameraToggleButton)
        arPanel.addSubview(arCollisionTitleLabel)
        arPanel.addSubview(arCollisionRenderCollectionView)
        arPanel.addSubview(arBBRadiusSlider)
        arPanel.addSubview(arBBRadiusLabel)
        
        
        arPanel.translatesAutoresizingMaskIntoConstraints = false
        arToggleButton.translatesAutoresizingMaskIntoConstraints = false
        arMeshToggleButton.translatesAutoresizingMaskIntoConstraints = false
        arCameraToggleButton.translatesAutoresizingMaskIntoConstraints = false
        arCollisionTitleLabel.translatesAutoresizingMaskIntoConstraints = false
        arCollisionRenderCollectionView.translatesAutoresizingMaskIntoConstraints = false
        arBBRadiusSlider.translatesAutoresizingMaskIntoConstraints = false
        arBBRadiusLabel.translatesAutoresizingMaskIntoConstraints = false
        
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
            
            // AR Mesh toggle button constraints (below AR toggle)
            arMeshToggleButton.topAnchor.constraint(
                equalTo: arToggleButton.bottomAnchor,
                constant: 10
            ),
            arMeshToggleButton.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arMeshToggleButton.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arMeshToggleButton.heightAnchor.constraint(equalToConstant: 40),
            
            // AR Camera toggle button constraints (below AR mesh toggle)
            arCameraToggleButton.topAnchor.constraint(
                equalTo: arMeshToggleButton.bottomAnchor,
                constant: 10
            ),
            arCameraToggleButton.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arCameraToggleButton.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arCameraToggleButton.heightAnchor.constraint(equalToConstant: 40),
            
            // AR Collision Title label constraints (below AR camera toggle)
            arCollisionTitleLabel.topAnchor.constraint(
                equalTo: arCameraToggleButton.bottomAnchor,
                constant: 10
            ),
            arCollisionTitleLabel.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arCollisionTitleLabel.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arCollisionTitleLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // AR Collision Render collection view constraints (below title label)
            arCollisionRenderCollectionView.topAnchor.constraint(
                equalTo: arCollisionTitleLabel.bottomAnchor,
                constant: 5
            ),
            arCollisionRenderCollectionView.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arCollisionRenderCollectionView.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arCollisionRenderCollectionView.heightAnchor.constraint(equalToConstant: 40),
            
            // AR Bounding Box Radius label constraints (below collision render collection view)
            arBBRadiusLabel.topAnchor.constraint(
                equalTo: arCollisionRenderCollectionView.bottomAnchor,
                constant: 10
            ),
            arBBRadiusLabel.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arBBRadiusLabel.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arBBRadiusLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // AR Bounding Box Radius slider constraints (below label)
            arBBRadiusSlider.topAnchor.constraint(
                equalTo: arBBRadiusLabel.bottomAnchor,
                constant: 5
            ),
            arBBRadiusSlider.leadingAnchor.constraint(
                equalTo: arPanel.leadingAnchor,
                constant: 10
            ),
            arBBRadiusSlider.trailingAnchor.constraint(
                equalTo: arPanel.trailingAnchor,
                constant: -10
            ),
            arBBRadiusSlider.heightAnchor.constraint(equalToConstant: 30),
        ])
        
        // Create and manage panel height constraint (adjusted dynamically)
        // Height: AR OFF = AR(40) + paddings(20) = 70
        // Height: AR ON = AR(40) + Mesh(40) + Camera(40) + CollisionTitle(20) + spacing(5) + Collision(40) + BBLabel(20) + BBSlider(30) + 7 spacings(65) + paddings(20) = 320
        arPanelHeightConstraint = arPanel.heightAnchor.constraint(equalToConstant: 70)
        arPanelHeightConstraint.isActive = true
        
        // Initialize AR panel visibility
        updateARButtonsVisibility()
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
    
    @objc private func cubeScaleChanged(_ slider: UISlider) {
        let scale = slider.value
        cubeScaleLabel.text = String(format: "Cube: %.1fx", scale)
        fluidRenderer.elasticCubeScale = scale
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
    
    @objc private func gridSpacingChanged(_ slider: UISlider) {
        let multiplier = slider.value
        gridSpacingLabel.text = String(format: "Spacing: %.1fx", multiplier)
        fluidRenderer.setGridSpacingMultiplier(multiplier)
    }
    
    @objc private func sdfScaleChanged(_ slider: UISlider) {
        let scale = slider.value
        sdfScaleLabel.text = String(format: "SDF Scale: %.2fx", scale)
        updateSdfTransform()
    }
    
    @objc private func sdfYOffsetChanged(_ slider: UISlider) {
        let offset = slider.value
        sdfYOffsetLabel.text = String(format: "SDF Y Offset: %.1f", offset)
        updateSdfTransform()
    }
    
    @objc private func sdfYRotationChanged(_ slider: UISlider) {
        let rotation = slider.value
        sdfYRotationLabel.text = String(format: "SDF Y Rotation: %.0f°", rotation)
        updateSdfTransform()
    }
    internal func updateSdfTransform(){
        let scale = sdfScaleSlider.value
        let offset = sdfYOffsetSlider.value
        let rotation = sdfYRotationSlider.value
        fluidRenderer.collisionManager?.representativeItem.updateSdfTransform(scale: SIMD3<Float>(scale, scale, scale), translate:SIMD3<Float>(0.0, offset, 0.0),rotate: SIMD3<Float>(0.0, rotation * Float.pi / 180.0, 0.0))
    }
    
    @objc private func worldTranslateXChanged(_ slider: UISlider) {
        let xOffset = slider.value
        worldTranslateXLabel.text = String(format: "World X: %.1f", xOffset)
        worldTranslationOffset.x = xOffset
    }
    
    @objc private func worldTranslateYChanged(_ slider: UISlider) {
        let yOffset = slider.value
        worldTranslateYLabel.text = String(format: "World Y: %.1f", yOffset)
        worldTranslationOffset.y = yOffset
    }
    
    @objc private func worldTranslateZChanged(_ slider: UISlider) {
        let zOffset = slider.value
        worldTranslateZLabel.text = String(format: "World Z: %.1f", zOffset)
        worldTranslationOffset.z = zOffset
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
        
        // Update UI visibility based on material mode
        updateMaterialModeUI()
        
        // Reset simulation when material mode changes
        renderer.reset()
        print("🔄 Material mode changed to: \(renderer.materialParameters.currentMaterialMode)")
    }
    
    private func updateMaterialModeUI() {
        guard let renderer = fluidRenderer else { return }
        
        let isElasticMode = (renderer.materialParameters.currentMaterialMode == .neoHookeanElastic)
        
        // Clear all arranged subviews
        materialScaleStackView.arrangedSubviews.forEach { subview in
            materialScaleStackView.removeArrangedSubview(subview)
            subview.removeFromSuperview()
        }
        
        if isElasticMode {
            // Show only Cube Scale controls for Elastic mode
            materialScaleStackView.addArrangedSubview(cubeScaleLabel)
            materialScaleStackView.addArrangedSubview(cubeScaleSlider)
        } else {
            // Show only Mass Scale controls for Fluid mode  
            materialScaleStackView.addArrangedSubview(massScaleLabel)
            materialScaleStackView.addArrangedSubview(massScaleSlider)
        }
        
        print("🔧 Material Mode UI - Elastic: \(isElasticMode ? "ON" : "OFF"), Stack contains: \(materialScaleStackView.arrangedSubviews.count) views")
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
        // Reset the UI offset when setting new base translation
        worldTranslationOffset = SIMD3<Float>(0.0, 0.0, 0.0)
        // Update UI sliders to reflect the reset offset
        DispatchQueue.main.async { [weak self] in
            self?.worldTranslateXSlider.value = 0.0
            self?.worldTranslateYSlider.value = 0.0
            self?.worldTranslateZSlider.value = 0.0
            self?.worldTranslateXLabel.text = "World X: 0.0"
            self?.worldTranslateYLabel.text = "World Y: 0.0"
            self?.worldTranslateZLabel.text = "World Z: 0.0"
        }
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
    
    
    internal var isAREnabled: Bool = false
    internal var arRenderer: ARRenderer?
    internal var savedBackgroundRenderer: BackgroundRenderer?
    internal var arBoundingBoxRadius: Float = 0.5  // Default 50cm radius
    
    
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
            let baseViewMatrix = getViewMatrix()
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
            // Perform GPU raycast to get hit position
            if let hitPosition = arRenderer.performGPURaycast(
                at: tapPoint,
                viewportSize: metalView.bounds.size,
                orientation: orientation
            ) {
                print("✅ AR raycast hit at: \(hitPosition)")
                
                // Simplified: use hit position directly (no grid transform compensation needed)
                setFluidWorldTranslation(hitPosition)
                
                // Extract mesh triangles around tap point and create collision
                setupCollisionFromARMesh(arRenderer: arRenderer, tapWorldPosition: hitPosition)
                
                // Set tap highlight for debug visualization (same bounding box as SDF)
                let boxSize = SIMD3<Float>(arBoundingBoxRadius, arBoundingBoxRadius, arBoundingBoxRadius) // Use configurable radius
                arRenderer.setTapHighlight(at: hitPosition, boxSize: boxSize)
                
                // Show visual feedback
                showARTapFeedback(at: tapPoint)
            } else {
                print("❌ AR raycast missed - no mesh hit at tap position")
            }
        }
#endif
    }
    
    private func showARTapFeedback(at tapPoint: CGPoint) {
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
        
        // Also reset translation offset and update UI
        worldTranslationOffset = SIMD3<Float>(0.0, 0.0, 0.0)
        worldTranslateXSlider.value = 0.0
        worldTranslateYSlider.value = 0.0
        worldTranslateZSlider.value = 0.0
        worldTranslateXLabel.text = "World X: 0.0"
        worldTranslateYLabel.text = "World Y: 0.0"
        worldTranslateZLabel.text = "World Z: 0.0"
        
        // Clear tap highlight when double tapping
        if let arRenderer = arRenderer {
            arRenderer.clearTapHighlight()
        }
        
        // Coefficients updated, no direct application to renderer
    }
    
    private func updateARButtonsVisibility() {
        let isAREnabled = self.isAREnabled
        
        print("🔍 AR State: \(isAREnabled), AR buttons hidden: \(!isAREnabled)")
        
        // Show/hide AR mesh, camera, collision render, and BB radius controls based on AR state
        arMeshToggleButton.isHidden = !isAREnabled
        arCameraToggleButton.isHidden = !isAREnabled
        arCollisionTitleLabel.isHidden = !isAREnabled
        arCollisionRenderCollectionView.isHidden = !isAREnabled
        arBBRadiusSlider.isHidden = !isAREnabled
        arBBRadiusLabel.isHidden = !isAREnabled
        
        // Reload collection view data when AR state changes
        if isAREnabled {
            arCollisionRenderCollectionView.reloadData()
        }
        
        // Hide collision controls in AR mode
        collisionPanel.isHidden = isAREnabled
        
        // Update AR panel height based on visibility
        // Elements: AR button (always), Mesh, Camera, CollisionTitle, Collision, BB Label, BB Slider (AR ON only)
        // Height formula: top padding (10) + sum(button heights + inner spacings) + bottom padding (10)
        // - AR OFF: AR(40) + paddings(20) = 70
        // - AR ON:  AR(40) + spacing(10) + Mesh(40) + spacing(10) + Camera(40) + spacing(10) + CollisionTitle(20) + spacing(5) + Collision(40) + spacing(10) + BBLabel(20) + spacing(5) + BBSlider(30) + paddings(20) = 320
        arPanelHeightConstraint.constant = isAREnabled ? 320 : 70
        
        print("📏 AR Panel height set to: \(arPanelHeightConstraint.constant)")
        
        // Animate the height change
        UIView.animate(withDuration: 0.3) {
            self.view.layoutIfNeeded()
        }
    }
    
    
    private func getViewMatrix() -> float4x4 {
        return matrix_identity_float4x4
    }
    
    private func getProjectionMatrix(aspectRatio: Float) -> float4x4 {
        let fov: Float = 45.0 * Float.pi / 180.0
        let nearPlane: Float = 0.1
        let farPlane: Float = 10.0
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
        guard let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
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
        var viewMatrix = getViewMatrix()
        var projectionMatrix = getProjectionMatrix(aspectRatio: aspectRatio)
        // Get AR camera matrices if available
        var cameraProjectionMatrix: float4x4? = nil
        var cameraViewMatrix: float4x4? = nil
        if isAREnabled {
#if canImport(ARKit)
            if #available(iOS 11.0, macOS 10.13, *) {
                let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
                if let (proj, viewM) = arRenderer?.getCameraMatrices(viewportSize: metalView.bounds.size, orientation: orientation) {
                    // Store raw camera matrices for collision rendering
                    cameraProjectionMatrix = proj
                    cameraViewMatrix = viewM
                    
                    // Also use for fluid coordinates (existing behavior)
                    projectionMatrix = proj
                    viewMatrix = viewM
                }
            }
#endif
        }
        viewMatrix = viewMatrix * computeWorldTransform()
        // Update simulation based on mode
        if performStep {
            fluidRenderer.updateComputeUniforms(worldTransform:computeWorldTransform())
        }
        fluidRenderer.updateVertexUniforms(
            deltaTime: deltaTime,
            screenSize: screenSize,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            worldTransform: computeWorldTransform()
        )
        
        // Use IntegratedRenderer as facade for all rendering
        integratedRenderer.render(
            renderPassDescriptor: renderPassDescriptor,
            performCompute: performStep,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            cameraProjectionMatrix: cameraProjectionMatrix,
            cameraViewMatrix: cameraViewMatrix
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
                fluidRenderer.backgroundRenderer = ARBackgroundRendererAdapter(arRenderer: ar, fluidRenderer: fluidRenderer, isTransparent: true)
                integratedRenderer.overlayRenderer = arRenderer
            }
            
            // AR mode alignment now handled by IntegratedRenderer
            
            arToggleButton.setTitle("AR: ON", for: .normal)
            arToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            // Stop AR session and detach background renderer
            integratedRenderer.overlayRenderer = nil
            arRenderer?.stopARSession()
            if let saved = savedBackgroundRenderer {
                fluidRenderer.backgroundRenderer = saved
                savedBackgroundRenderer = nil
            } else {
                installDefaultBackgroundRenderer()
            }

            // AR mode alignment now handled by IntegratedRenderer
            
            arToggleButton.setTitle("AR: OFF", for: .normal)
            arToggleButton.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        }
        onARStateChanged()
    }
    
    // Called when AR state changes
    func onARStateChanged() {
        DispatchQueue.main.async {
            self.updateARButtonsVisibility()
        }
    }
    
    // MARK: - AR Mesh Collision Setup
    
    private func setupCollisionFromARMesh(arRenderer: ARRenderer, tapWorldPosition: SIMD3<Float>) {
#if canImport(ARKit)
        
        // Define bounding box size around tap point (in world coordinates)
        let boundingBoxSize: Float = arBoundingBoxRadius // Use configurable radius
        let boundingBox = SIMD3<Float>(boundingBoxSize, boundingBoxSize, boundingBoxSize)
        
        // Extract triangles from AR mesh around adjusted tap point
        guard let worldTriangles = arRenderer.extractMeshesInBoundingBoxGPU(
            center: tapWorldPosition,
            size: boundingBox
        ) else {
            print("❌ Failed to extract mesh triangles around tap point")
            return
        }
        
        if worldTriangles.isEmpty {
            print("⚠️ No triangles found around tap point")
            return
        }
        
        // Use triangles directly (no transformation needed)
        generateSDFFromTriangles(triangles: worldTriangles)
        
#endif
    }
    
    
    private func generateSDFFromTriangles(triangles: [Triangle]) {
        print("🔧 Generating SDF from \(triangles.count) transformed triangles")
        
        guard let collisionManager = fluidRenderer.collisionManager else {
            print("❌ Collision manager not available")
            return
        }
        
        // Get scene boundary for SDF resolution
        let (boundaryMin, boundaryMax) = fluidRenderer.getBoundaryMinMax()
        let sdfResolution = SIMD3<Int32>(64, 64, 64) // Fixed resolution
        
        print("🔧 SDF Generation Parameters:")
        print("  Boundary Min: \(boundaryMin)")
        print("  Boundary Max: \(boundaryMax)")
        print("  SDF Resolution: \(sdfResolution)")
        
        // Debug: Show some transformed triangle bounds
        if !triangles.isEmpty {
            let firstTriangle = triangles[0]
            let minBounds = min(min(firstTriangle.v0, firstTriangle.v1), firstTriangle.v2)
            let maxBounds = max(max(firstTriangle.v0, firstTriangle.v1), firstTriangle.v2)
            print("🔍 First transformed triangle bounds:")
            print("  Min: \(minBounds)")
            print("  Max: \(maxBounds)")
        }
        collisionManager.setMeshVisible(false)
        collisionManager.representativeItem.setEnabled(false)
        let arItem = CollisionItem(device: fluidRenderer.device,isARMode: true)
        
        // Process and generate SDF from the transformed triangles
        arItem.processAndGenerateSDF(
            sdfGenerator: collisionManager.sdfGenerator,
            triangles: triangles,
            resolution: sdfResolution,
            gridBoundaryMin: boundaryMin,
            gridBoundaryMax: boundaryMax
        )
        collisionManager.representativeItem = arItem
        
        // Enable collision and make mesh visible
        collisionManager.representativeItem.setEnabled(true)
        collisionManager.setMeshVisible(true)
        collisionManager.representativeItem.setMeshColor(SIMD4<Float>(0.0, 1.0, 0.0, 0.8)) // Green semi-transparent
        
        // Reload collection view to show new collision item
        DispatchQueue.main.async { [weak self] in
            self?.arCollisionRenderCollectionView.reloadData()
        }
        
        print("✅ SDF generated and collision enabled from AR mesh")
    }
    
    @objc private func toggleARMesh() {
        guard let arRenderer = arRenderer else { return }
        
        let currentlyVisible = arRenderer.showARMeshWireframe
        let newVisibility = !currentlyVisible
        
        arRenderer.setMeshRenderingEnabled(newVisibility)
        
        if newVisibility {
            arMeshToggleButton.setTitle("Mesh: ON", for: .normal)
            arMeshToggleButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        } else {
            arMeshToggleButton.setTitle("Mesh: OFF", for: .normal)
            arMeshToggleButton.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        }
        
        print("🔧 AR Mesh visibility: \(newVisibility)")
    }
    
    @objc private func arBBRadiusChanged(_ slider: UISlider) {
        arBoundingBoxRadius = slider.value
        arBBRadiusLabel.text = String(format: "BB: %.1fm", arBoundingBoxRadius)
        print("🔧 AR Bounding Box Radius: \(arBoundingBoxRadius)m")
    }
    
    @objc private func toggleARCamera() {
        guard let arRenderer = arRenderer else { return }
        
        // Get current camera rendering state from ARRenderer
        let currentlyActive = arRenderer.isCameraRenderingEnabled
        let newActive = !currentlyActive
        
        arRenderer.setCameraRenderingEnabled(newActive)
        
        // Switch background renderer based on camera state
        if newActive {
            // Camera ON: Use AR background
            fluidRenderer.backgroundRenderer = ARBackgroundRendererAdapter(arRenderer: arRenderer, fluidRenderer: fluidRenderer, isTransparent: true)
            arCameraToggleButton.setTitle("Camera: ON", for: .normal)
            arCameraToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            // Camera OFF: Use original fluid background
            if let saved = savedBackgroundRenderer {
                fluidRenderer.backgroundRenderer = saved
            } else {
                installDefaultBackgroundRenderer()
            }
            arCameraToggleButton.setTitle("Camera: OFF", for: .normal)
            arCameraToggleButton.backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        }
        
        print("🔧 AR Camera active: \(newActive)")
    }
    
}

// MARK: - CollisionRenderCell
class CollisionRenderCell: UICollectionViewCell {
    private let label: UILabel = UILabel()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupCell()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupCell()
    }
    
    private func setupCell() {
        backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
        layer.cornerRadius = 8
        
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 12, weight: .medium)
        label.textColor = .white
        addSubview(label)
        
        label.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: centerXAnchor),
            label.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }
    
    func configure(with index: Int, isSelected: Bool) {
        label.text = "\(index + 1)"
        
        if isSelected {
            backgroundColor = UIColor.systemOrange.withAlphaComponent(0.9)
            layer.borderWidth = 2
            layer.borderColor = UIColor.white.cgColor
        } else {
            backgroundColor = UIColor.systemGray.withAlphaComponent(0.8)
            layer.borderWidth = 0
            layer.borderColor = UIColor.clear.cgColor
        }
    }
}

// MARK: - UICollectionView DataSource & Delegate
extension ViewController: UICollectionViewDataSource, UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        // Return the number of collision items
        guard collectionView == arCollisionRenderCollectionView else { return 0 }
        return fluidRenderer?.collisionManager?.items.count ?? 0
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        guard collectionView == arCollisionRenderCollectionView else { return UICollectionViewCell() }
        
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "CollisionRenderCell", for: indexPath) as! CollisionRenderCell
        
        // Check if this collision item is currently enabled (not skipped)
        let isEnabled = !(fluidRenderer?.collisionManager?.items[indexPath.item].skipRenderInAR ?? true)
        cell.configure(with: indexPath.item, isSelected: isEnabled)
        
        return cell
    }
    
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        guard collectionView == arCollisionRenderCollectionView,
              let collisionManager = fluidRenderer?.collisionManager,
              indexPath.item < collisionManager.items.count else { return }
        
        // Toggle skipRenderInAR for the selected collision item
        let item = collisionManager.items[indexPath.item]
        item.skipRenderInAR = !item.skipRenderInAR
        
        // Update the cell appearance
        collectionView.reloadItems(at: [indexPath])
        
        print("🔧 Collision Item \(indexPath.item + 1) Render: \(item.skipRenderInAR ? "DISABLED" : "ENABLED")")
    }
}
