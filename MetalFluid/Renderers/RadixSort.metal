#include <metal_stdlib>
#include "../MPMTypes.h"
using namespace metal;

// ===== ONE-SWEEP RADIX SORT IMPLEMENTATION =====

// Sort key structure (shared with BitonicSort.metal)
struct SortKey {
    uint key;      // Grid index as sort key
    uint value;    // Original particle index
};

// Constants for radix sort
constant uint RADIX_BITS = 4;
constant uint RADIX_SIZE = 16; // 2^RADIX_BITS
constant uint NUM_PASSES = 8;  // ceil(32 / RADIX_BITS) for 32-bit keys

// Extract sort keys for radix sort (same as bitonic sort)
kernel void extractSortKeysRadix(
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
    uint gridIdx = cell_ids.x * (uniforms.gridResolution.z * uniforms.gridResolution.y) + 
                   cell_ids.y * uniforms.gridResolution.z + cell_ids.z;
    
    sortKeys[id].key = gridIdx;
    sortKeys[id].value = id;
}

// One-sweep radix sort with local histogram
kernel void onesweepRadixSortPass(
    device const SortKey* inputKeys [[buffer(0)]],
    device SortKey* outputKeys [[buffer(1)]],
    device atomic_uint* globalHistogram [[buffer(2)]],
    constant uint& numKeys [[buffer(3)]],
    constant uint& pass [[buffer(4)]],
    uint threadgroupId [[threadgroup_position_in_grid]],
    uint threadId [[thread_position_in_threadgroup]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]],
    threadgroup atomic_uint* localHistogram [[threadgroup(0)]]
) {
    uint globalId = threadgroupId * threadsPerThreadgroup + threadId;
    uint shift = pass * RADIX_BITS;
    uint mask = RADIX_SIZE - 1;
    
    // Initialize local histogram
    if (threadId < RADIX_SIZE) {
        atomic_store_explicit(&localHistogram[threadId], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Count phase: build local histogram
    uint key = UINT_MAX;
    uint digit = 0;
    if (globalId < numKeys) {
        key = inputKeys[globalId].key;
        digit = (key >> shift) & mask;
        atomic_fetch_add_explicit(&localHistogram[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Upload local histogram to global histogram
    if (threadId < RADIX_SIZE) {
        uint localCount = atomic_load_explicit(&localHistogram[threadId], memory_order_relaxed);
        if (localCount > 0) {
            atomic_fetch_add_explicit(&globalHistogram[threadgroupId * RADIX_SIZE + threadId], 
                                    localCount, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    // Scatter phase: write to output based on prefix sum
    if (globalId < numKeys) {
        // Calculate global position for this key
        uint position = 0;
        
        // Add counts from all previous digits across all threadgroups
        for (uint d = 0; d < digit; d++) {
            for (uint tg = 0; tg <= threadgroupId; tg++) {
                position += atomic_load_explicit(&globalHistogram[tg * RADIX_SIZE + d], memory_order_relaxed);
            }
        }
        
        // Add counts from previous threadgroups for current digit
        for (uint tg = 0; tg < threadgroupId; tg++) {
            position += atomic_load_explicit(&globalHistogram[tg * RADIX_SIZE + digit], memory_order_relaxed);
        }
        
        // Add local offset within current threadgroup
        uint localOffset = 0;
        for (uint tid = 0; tid < threadId; tid++) {
            uint otherGlobalId = threadgroupId * threadsPerThreadgroup + tid;
            if (otherGlobalId < numKeys) {
                uint otherKey = inputKeys[otherGlobalId].key;
                uint otherDigit = (otherKey >> shift) & mask;
                if (otherDigit == digit) {
                    localOffset++;
                }
            }
        }
        
        uint finalPosition = position + localOffset;
        if (finalPosition < numKeys) {
            outputKeys[finalPosition].key = key;
            outputKeys[finalPosition].value = inputKeys[globalId].value;
        }
    }
}

// Simplified radix sort using local sorting within threadgroups
kernel void radixSortLocal(
    device SortKey* keys [[buffer(0)]],
    device SortKey* tempKeys [[buffer(1)]],
    constant uint& numKeys [[buffer(2)]],
    constant uint& pass [[buffer(3)]],
    uint threadgroupId [[threadgroup_position_in_grid]],
    uint threadId [[thread_position_in_threadgroup]],
    uint threadsPerThreadgroup [[threads_per_threadgroup]],
    threadgroup SortKey* sharedData [[threadgroup(0)]]
) {
    uint globalId = threadgroupId * threadsPerThreadgroup + threadId;
    uint shift = pass * RADIX_BITS;
    uint mask = RADIX_SIZE - 1;
    uint elementsPerThreadgroup = min(threadsPerThreadgroup, numKeys - threadgroupId * threadsPerThreadgroup);
    
    // Load data into shared memory
    if (threadId < elementsPerThreadgroup && globalId < numKeys) {
        sharedData[threadId] = keys[globalId];
    } else if (threadId < threadsPerThreadgroup) {
        sharedData[threadId].key = UINT_MAX;
        sharedData[threadId].value = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform local radix sort
    for (uint digit = 0; digit < RADIX_SIZE; digit++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Count prefix for current digit
        uint hasDigit = ((sharedData[threadId].key >> shift) & mask) == digit ? 1u : 0u;
        uint position = 0;
        
        // Calculate position within digit group
        for (uint i = 0; i < threadId; i++) {
            if (((sharedData[i].key >> shift) & mask) == digit) {
                position++;
            }
        }
        
        // Calculate base position for this digit
        uint basePos = 0;
        for (uint d = 0; d < digit; d++) {
            for (uint i = 0; i < elementsPerThreadgroup; i++) {
                if (((sharedData[i].key >> shift) & mask) == d) {
                    basePos++;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Write to temporary position if this thread has current digit
        if (hasDigit && (basePos + position) < elementsPerThreadgroup) {
            tempKeys[basePos + position] = sharedData[threadId];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Copy sorted data back to shared memory
        if (threadId < elementsPerThreadgroup) {
            sharedData[threadId] = tempKeys[threadId];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write sorted data back to global memory
    if (threadId < elementsPerThreadgroup && globalId < numKeys) {
        keys[globalId] = sharedData[threadId];
    }
}

// Initialize global histogram for radix sort
kernel void initializeRadixHistogram(
    device atomic_uint* histogram [[buffer(0)]],
    constant uint& numThreadgroups [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint totalSize = numThreadgroups * RADIX_SIZE;
    if (id < totalSize) {
        atomic_store_explicit(&histogram[id], 0u, memory_order_relaxed);
    }
}

// Compute global prefix sums for histogram
kernel void computePrefixSums(
    device atomic_uint* histogram [[buffer(0)]],
    constant uint& numThreadgroups [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= RADIX_SIZE) return;
    
    // Compute prefix sum for each digit across all threadgroups
    uint sum = 0;
    for (uint tg = 0; tg < numThreadgroups; tg++) {
        uint currentCount = atomic_load_explicit(&histogram[tg * RADIX_SIZE + id], memory_order_relaxed);
        atomic_store_explicit(&histogram[tg * RADIX_SIZE + id], sum, memory_order_relaxed);
        sum += currentCount;
    }
}

// Reorder particles after radix sort (same as bitonic sort)
kernel void reorderParticlesRadix(
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
    // Do not reorder rigid info; particles contain originalIndex to fetch rigid info
    }
}

// Verify radix sort correctness
kernel void verifyRadixSortOrder(
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
