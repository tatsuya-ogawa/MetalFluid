#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
using namespace metal;

// ===== BITONIC SORT IMPLEMENTATION =====

// Sort key structure for particle sorting
struct SortKey {
    uint key;      // Grid index as sort key
    uint value;    // Original particle index
};

// Utility function for grid index calculation
inline uint gridIndex(uint x, uint y, uint z, int3 res) {
    return x * (res.z * res.y) + y * res.z + z;
}

// Extract grid index as sort key for each particle
kernel void extractSortKeys(
    device const MPMParticle* particles [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    device SortKey* sortKeys [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    
    // Convert particle position to grid coordinates
    float3 position = (particles[id].position - uniforms.domainOrigin) / uniforms.gridSpacing;
    int3 cell_ids = int3(floor(position));
    
    // Clamp to grid bounds
    cell_ids = clamp(cell_ids, int3(0), uniforms.gridResolution - 1);
    
    // Calculate grid index
    uint gridIdx = gridIndex(cell_ids.x, cell_ids.y, cell_ids.z, uniforms.gridResolution);
    
    sortKeys[id].key = gridIdx;
    sortKeys[id].value = id;
}

// Bitonic sort merge step
kernel void bitonicSort(
    device SortKey* keys [[buffer(0)]],
    constant uint& numKeys [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    constant uint& step [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numKeys) return;
    
    uint partner = id ^ step;
    
    if (partner > id && partner < numKeys) {
        bool ascending = ((id & stage) == 0);
        
        if ((keys[id].key > keys[partner].key) == ascending) {
            // Swap elements
            SortKey temp = keys[id];
            keys[id] = keys[partner];
            keys[partner] = temp;
        }
    }
}

// Initialize sort keys with sequential indices (for empty grid cells)
kernel void initializeSortKeys(
    device SortKey* sortKeys [[buffer(0)]],
    constant uint& numKeys [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numKeys) return;
    
    sortKeys[id].key = UINT_MAX;  // Empty cells go to the end
    sortKeys[id].value = id;
}

// Reorder particles based on sorted keys
kernel void reorderParticles(
    device const MPMParticle* inputParticles [[buffer(0)]],
    device MPMParticle* outputParticles [[buffer(1)]],
    device const SortKey* sortedKeys [[buffer(2)]],
    constant uint& numParticles [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numParticles) return;
    
    uint originalIndex = sortedKeys[id].value;
    
    // Bounds check for safety
    if (originalIndex < numParticles) {
        outputParticles[id] = inputParticles[originalIndex];
        // Note: do not reorder rigid info. Particles store originalIndex and rigid info
        // should be fetched via that index. This keeps rigid info in place (unsorted).
    }
}

// ===== UTILITY KERNELS =====

// Verify sort correctness (debugging utility)
kernel void verifySortOrder(
    device const SortKey* sortedKeys [[buffer(0)]],
    device atomic_uint* errorCount [[buffer(1)]],
    constant uint& numKeys [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numKeys - 1) return;
    
    // Check if current key <= next key
    if (sortedKeys[id].key > sortedKeys[id + 1].key) {
        atomic_fetch_add_explicit(errorCount, 1u, memory_order_relaxed);
    }
}

// Count particles per grid cell (for debugging and analysis)
kernel void countParticlesPerCell(
    device const MPMParticle* particles [[buffer(0)]],
    constant ComputeShaderUniforms& uniforms [[buffer(1)]],
    device atomic_uint* cellCounts [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms.particleCount) return;
    
    // Convert particle position to grid coordinates
    float3 position = (particles[id].position - uniforms.domainOrigin) / uniforms.gridSpacing;
    int3 cell_ids = int3(floor(position));
    
    // Clamp to grid bounds
    cell_ids = clamp(cell_ids, int3(0), uniforms.gridResolution - 1);
    
    // Calculate grid index
    uint gridIdx = gridIndex(cell_ids.x, cell_ids.y, cell_ids.z, uniforms.gridResolution);
    
    // Bounds check
    uint totalCells = uniforms.gridResolution.x * uniforms.gridResolution.y * uniforms.gridResolution.z;
    if (gridIdx < totalCells) {
        atomic_fetch_add_explicit(&cellCounts[gridIdx], 1u, memory_order_relaxed);
    }
}
