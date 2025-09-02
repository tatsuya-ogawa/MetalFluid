# MetalFluid

A high-performance fluid simulation engine for iOS/macOS using the MLS-MPM (Moving Least Squares Material Point Method) algorithm, implemented with Apple's Metal compute shaders.

## Overview

MetalFluid is a real-time fluid simulation system that leverages the GPU compute capabilities of Metal to achieve high-performance particle-based fluid dynamics. The project implements the MLS-MPM algorithm, which combines the advantages of both grid-based and particle-based methods for accurate and stable fluid simulation.

## Features

- **Real-time MLS-MPM Simulation**: Advanced fluid dynamics using Moving Least Squares Material Point Method
- **Metal Compute Shaders**: GPU-accelerated computation for optimal performance on Apple devices
- **Dual Rendering Modes**: 
  - Particle rendering with pressure heatmap visualization
  - Realistic water surface rendering with depth-based filtering and thickness mapping
- **Interactive Controls**: Touch-based camera manipulation and fluid interaction
- **Optimized Performance**: Configurable simulation substeps and particle sorting for performance tuning

## Technical Implementation

### Algorithm Details

The simulation follows the standard MLS-MPM pipeline:

1. **Particle to Grid (P2G)**: Transfer particle data (mass, momentum, stress) to grid nodes
2. **Grid Update**: Apply forces, boundary conditions, and velocity updates on the grid
3. **Grid to Particle (G2P)**: Interpolate updated velocities back to particles
4. **Particle Advection**: Update particle positions and deformation gradients

### Key Components

- **Compute Kernels** (`ComputeShaders.metal`): Core MLS-MPM algorithm implementation
- **Rendering Pipeline** (`RenderShaders.metal`): Particle and fluid surface visualization
- **Sorting System** (`BitonicSort.metal`): Spatial sorting for cache-efficient computation
- **Fluid Renderer** (`FluidRenderer.swift`): Main simulation coordinator and Metal setup

### Metal-Specific Optimizations

- **Atomic Float Operations**: Unlike WebGPU limitations, Metal allows atomic operations on float types, enabling more efficient grid updates
- **Compute Pipeline Optimization**: Leverages Metal's threadgroup memory and SIMD operations
- **Buffer Management**: Efficient memory layout for optimal GPU cache utilization

## Project Origins

This implementation is primarily based on [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) with significant adaptations for Metal:

- **Atomic Operations**: WebGPU's integer-only atomic limitations vs. Metal's native float atomic support
- **Shader Language**: WGSL to Metal Shading Language translation
- **Platform Integration**: iOS/macOS-specific UI and rendering pipeline

## Configuration

Key simulation parameters can be adjusted in `FluidRenderer.swift`:

```swift
public let particleCount: Int = 40000        // Number of simulation particles
public let gridSize: Int = 64                // Grid resolution (64³)
public var simulationSubsteps: Int = 2       // Substeps per frame
public let gravity: Float = -2.5             // Gravitational acceleration
public let gridSpacing: Float = 1.0          // Grid cell size
```

## Rendering Modes

### Particle Mode
- Visualizes individual particles with pressure-based color mapping
- Real-time pressure heatmap display
- Adjustable particle size scaling

### Water Mode
- Realistic fluid surface rendering
- Screen-space depth filtering for smooth surfaces
- Thickness-based transparency and refraction
- Environment mapping for realistic reflections

## Performance Considerations

- **Particle Count**: Higher particle counts provide better detail but impact performance
- **Grid Resolution**: Larger grids increase memory usage and computation time
- **Substeps**: More substeps improve stability but reduce frame rate
- **Sorting**: Spatial sorting can improve cache coherency but adds overhead

## Requirements

- iOS 13.0+ or macOS 10.15+
- Metal-capable device
- Xcode 12.0+

## Building and Running

1. Open `MetalFluid.xcodeproj` in Xcode
2. Select your target device (iOS Simulator/Device or Mac)
3. Build and run the project

## Controls

- **Pan**: Rotate camera around the fluid
- **Pinch**: Zoom in/out
- **Tap**: Add forces to the fluid at touch location

## License

This project is open source. Please refer to the original [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) project for licensing terms of the base algorithm implementation.