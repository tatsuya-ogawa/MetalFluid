//
//  FluidRenderer+Sort.swift
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/08/17.
//

import Metal
import MetalKit
import simd

// MARK: - Sort Extension - Delegates to SortManager
extension MPMFluidRenderer {
    
    // MARK: - Public API for particle sorting configuration (delegates to SortManager)
    
    public func setParticleSorting(enabled: Bool, frequency: Int = 4) {
        sortManager.setParticleSorting(enabled: enabled, frequency: frequency)
    }
    
    public func forceSortParticles() {
        var computeBuffer = scene.getComputeParticleBuffer()
        sortManager.forceSortParticles(
            computeParticleBuffer: &computeBuffer,
            uniformBuffer: scene.getComputeUniformBuffer()
        )
    }
    
    // MARK: - Sorting Algorithm Selection (delegates to SortManager)
    
    public func setSortingAlgorithm(_ algorithm: SortingAlgorithm) {
        sortManager.setSortingAlgorithm(algorithm)
    }
    
    public func toggleSortingAlgorithm() {
        sortManager.toggleSortingAlgorithm()
    }
    
    public func getSortingAlgorithmName() -> String {
        return sortManager.getSortingAlgorithmName()
    }
    
    // MARK: - Computed properties for compatibility
    
    public var enableParticleSorting: Bool {
        get { sortManager.enableParticleSorting }
        set { sortManager.enableParticleSorting = newValue }
    }
    
    public var sortingFrequency: Int {
        get { sortManager.sortingFrequency }
        set { sortManager.sortingFrequency = newValue }
    }
    
    public var currentSortingAlgorithm: SortingAlgorithm {
        get { sortManager.currentSortingAlgorithm }
        set { sortManager.currentSortingAlgorithm = newValue }
    }
}
