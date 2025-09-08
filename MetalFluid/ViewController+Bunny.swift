//
//  ViewController+Extension.swift
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/09/08.
//

import MetalKit
import UIKit
import simd
import ReplayKit
#if canImport(ARKit)
import ARKit
#endif

extension ViewController {
    // MARK: - Transform Management
    private func setInitialWorldTransform() {
        // Just set the initial coefficients, don't apply to renderer yet
        worldTranslation = SIMD3<Float>(0.0, 0.0, -1.5)
        worldTranslationOffset = SIMD3<Float>(0.0, 0.0, 0.0)
        worldYaw = 0.0
        worldPitch = 0.0
        worldScale = 1.0
    }
    
    internal func setupBunny(){
        setupCollisionMesh()
        setInitialWorldTransform()
    }
    // MARK: - Mesh Loading
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
                // Apply initial transform
                self?.updateSdfTransform()
                self?.fluidRenderer.collisionManager?.representativeItem.setMeshColor(SIMD4<Float>(1.0, 1.0, 1.0, 0.8)) // Semi-transparent white
                print("🐰 Stanford Bunny collision mesh configured!")
            } else {
                print("❌ Failed to load Stanford Bunny. Collision detection disabled.")
            }
        }
    }
}
