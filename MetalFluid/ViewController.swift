import MetalKit
import UIKit
import simd
import ReplayKit

class ViewController: UIViewController {
    private let recordKey = "r"
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
    
    // Wireframe state tracking
    private var isWireframeMode: Bool = false

    // Initial values (shared between sliders and FluidRenderer)
    private let initialParticleCount: Float = 40000
    private let initialGridSize: Float = 48
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
    
    // Collision controls panel (right side)
    private var collisionPanel: UIView!
    private var collisionToggleButton: UIButton!
    private var fillModeButton: UIButton!
    private var meshVisibilityButton: UIButton!
    private var wireframeButton: UIButton!
    
    // ReplayKit recording
    private let screenRecorder = RPScreenRecorder.shared()
    private var isRecording = false

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
        setupKeyboardHandling()
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
        
        // Load Stanford Bunny for collision detection
        setupCollisionMesh()
    }
    
    private func setupCollisionMesh() {
        guard let bunnyURL = Bundle.main.url(forResource: "bunny", withExtension: "obj") else {
            print("⚠️ bunny.obj not found in bundle. Collision detection disabled.")
            return
        }
        // Get grid boundary to position bunny correctly
        let (boundaryMin, _) = fluidRenderer.getBoundaryMinMax()
        
        fluidRenderer.collisionManager?.loadMesh(
            objURL: bunnyURL,
            resolution: fluidRenderer.getGridRes(),
            fillMode: true,
            gridBoundaryMin: boundaryMin
        )
        
        // Configure collision visualization
        fluidRenderer.collisionManager?.setMeshVisible(true)
        fluidRenderer.collisionManager?.setMeshColor(SIMD4<Float>(0.8, 0.3, 0.3, 0.4)) // Semi-transparent red
        
        print("🐰 Stanford Bunny collision mesh loaded successfully!")
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
        
        // Particle count slider
        particleCountSlider = UISlider()
        particleCountSlider.minimumValue = 1000
        particleCountSlider.maximumValue = 300000
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

        // Add buttons to control panel
        controlPanel.addSubview(modeButton)
        controlPanel.addSubview(stepButton)
        controlPanel.addSubview(resetButton)
        controlPanel.addSubview(renderModeButton)
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
        particleSizeSlider.translatesAutoresizingMaskIntoConstraints = false
        particleSizeLabel.translatesAutoresizingMaskIntoConstraints = false
        massScaleSlider.translatesAutoresizingMaskIntoConstraints = false
        massScaleLabel.translatesAutoresizingMaskIntoConstraints = false
        particleCountSlider.translatesAutoresizingMaskIntoConstraints = false
        particleCountLabel.translatesAutoresizingMaskIntoConstraints = false
        gridSizeSlider.translatesAutoresizingMaskIntoConstraints = false
        gridSizeLabel.translatesAutoresizingMaskIntoConstraints = false

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
        
        // Fill mode button
        fillModeButton = UIButton(type: .system)
        fillModeButton.setTitle("Surface", for: .normal)
        fillModeButton.setTitleColor(.white, for: .normal)
        fillModeButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
        fillModeButton.layer.cornerRadius = 8
        fillModeButton.addTarget(
            self,
            action: #selector(toggleFillMode),
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
        collisionPanel.addSubview(fillModeButton)
        collisionPanel.addSubview(meshVisibilityButton)
        collisionPanel.addSubview(wireframeButton)
        
        // Setup constraints
        collisionPanel.translatesAutoresizingMaskIntoConstraints = false
        collisionToggleButton.translatesAutoresizingMaskIntoConstraints = false
        fillModeButton.translatesAutoresizingMaskIntoConstraints = false
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
            
            // Fill mode button constraints
            fillModeButton.topAnchor.constraint(
                equalTo: collisionToggleButton.bottomAnchor,
                constant: 10
            ),
            fillModeButton.leadingAnchor.constraint(
                equalTo: collisionPanel.leadingAnchor,
                constant: 10
            ),
            fillModeButton.trailingAnchor.constraint(
                equalTo: collisionPanel.trailingAnchor,
                constant: -10
            ),
            fillModeButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Mesh visibility button constraints
            meshVisibilityButton.topAnchor.constraint(
                equalTo: fillModeButton.bottomAnchor,
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
            
            // Bottom constraint to define collision panel height
            collisionPanel.bottomAnchor.constraint(
                equalTo: wireframeButton.bottomAnchor,
                constant: 10
            ),
        ])
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
        // Reload collision mesh after reset
        setupCollisionMesh()
    }
    
    // MARK: - Collision Control Actions
    
    @objc private func toggleCollision() {
        guard let isEnabled = fluidRenderer.collisionManager?.isEnabled() else {
            return
        }
        fluidRenderer.collisionManager?.setEnabled(!isEnabled)
        
        if !isEnabled{
            collisionToggleButton.setTitle("Collision: ON", for: .normal)
            collisionToggleButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.8)
        } else {
            collisionToggleButton.setTitle("Collision: OFF", for: .normal)
            collisionToggleButton.backgroundColor = UIColor.systemRed.withAlphaComponent(0.8)
        }
    }
    
    @objc private func toggleFillMode() {
        guard let isFillMode = fluidRenderer.collisionManager?.getFillMode() else {
            return
        }
        fluidRenderer.collisionManager?.setFillMode(!isFillMode)
        if !isFillMode {
            fillModeButton.setTitle("Fill Inside", for: .normal)
            fillModeButton.backgroundColor = UIColor.systemTeal.withAlphaComponent(0.8)
        } else {
            fillModeButton.setTitle("Surface", for: .normal)
            fillModeButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
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
        
        fluidRenderer.collisionManager?.setMeshWireframe(isWireframeMode)
        
        if isWireframeMode {
            wireframeButton.setTitle("Solid", for: .normal)
            wireframeButton.backgroundColor = UIColor.systemIndigo.withAlphaComponent(0.8)
        } else {
            wireframeButton.setTitle("Wireframe", for: .normal)
            wireframeButton.backgroundColor = UIColor.systemPurple.withAlphaComponent(0.8)
        }
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
