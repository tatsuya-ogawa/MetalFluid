//
//  FluidRenderer+Debug.swift
//  MetalFluid
//
//  Created by Tatsuya Ogawa on 2025/08/17.
//
import Metal
import MetalKit
import simd
extension MPMFluidRenderer{
    // Debug functionality for development
    // Debug output particle positions
    public func dumpParticlePositions() {
        let pointer = particleBuffer.contents().bindMemory(to: MPMParticle.self, capacity: particleCount)
        print("--- Dumping particle positions (", particleCount, ") ---")
        let n = min(20, particleCount)
        for i in 0..<n {
            let pos = pointer[i].position
            print("Particle[\(i)]: (x=\(pos.x), y=\(pos.y), z=\(pos.z))")
            let vel = pointer[i].velocity
            print("Velocity[\(i)]: (x=\(vel.x), y=\(vel.y), z=\(vel.z))")
            let C = pointer[i].C
            print("C[\(i)]: (x0=\(C.columns.0.x), y0=\(C.columns.0.y), z0=\(C.columns.0.z))")
            print("C[\(i)]: (x1=\(C.columns.1.x), y1=\(C.columns.1.y), z1=\(C.columns.1.z))")
            print("C[\(i)]: (x2=\(C.columns.2.x), y2=\(C.columns.2.y), z2=\(C.columns.2.z))")
        }
        if particleCount > 20 {
            print("... (Summary) ...")
            // Calculate mean and variance
            var sum = SIMD3<Float>(repeating: 0)
            var sumSq = SIMD3<Float>(repeating: 0)
            for i in 0..<particleCount {
                let p = pointer[i].position
                sum += p
                sumSq += p * p
            }
            let count = Float(particleCount)
            let mean = sum / count
            let meanSq = sumSq / count
            let variance = meanSq - mean * mean
            print("Mean: (x=\(mean.x), y=\(mean.y), z=\(mean.z))")
            print("Variance: (x=\(variance.x), y=\(variance.y), z=\(variance.z))")
        }
    }


    // Copy contents of gridBuffer to debugGridBuffer
    public func copyGridForDebug() {
        memcpy(debugGridBuffer.contents(), gridBuffer.contents(), gridBuffer.length)
    }

    // Dump contents of debugGridBuffer
    public func dumpGridNodes() {
        let pointer = debugGridBuffer.contents().bindMemory(to: MPMGridNode.self, capacity: gridNodes)
        print("--- Dumping grid nodes (", gridNodes, ") ---")
        let n = min(20, gridNodes)
        for i in 0..<n {
            let node = pointer[i]
            print("Grid[\(i)]: mass=\(node.mass), vx=\(node.velocity_x), vy=\(node.velocity_y), vz=\(node.velocity_z)")
        }
        if gridNodes > 20 {
            print("... (Summary) ...")
            var sumMass: Float = 0
            var sumV = SIMD3<Float>(repeating: 0)
            for i in 0..<gridNodes {
                let node = pointer[i]
                sumMass += node.mass
                sumV += SIMD3<Float>(node.velocity_x, node.velocity_y, node.velocity_z)
                if node.mass != 0{
                    print("effective mass id:\(i),mass:\(node.mass)")
                    print("effective velocity id:\(i),vx:\(node.velocity_x),vy:\(node.velocity_y),vz:\(node.velocity_z)")
                }
            }
            let count = Float(gridNodes)
            let meanMass = sumMass / count
            let meanV = sumV / count
            print("Mean mass: \(meanMass)")
            print("Mean velocity: (x=\(meanV.x), y=\(meanV.y), z=\(meanV.z))")
        }
    }

}
