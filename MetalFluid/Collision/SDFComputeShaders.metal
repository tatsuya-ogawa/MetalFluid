#include <metal_stdlib>
#include "../MPMTypes.h"
using namespace metal;

// Triangle structure for SDF generation
struct SDFTriangle {
    float3 v0;
    float3 v1;
    float3 v2;
};

// Distance from point to triangle
inline float distanceToTriangle(float3 point, SDFTriangle triangle) {
    float3 v0 = triangle.v0;
    float3 v1 = triangle.v1;
    float3 v2 = triangle.v2;
    
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 v0p = point - v0;
    
    float d00 = dot(v0v1, v0v1);
    float d01 = dot(v0v1, v0v2);
    float d11 = dot(v0v2, v0v2);
    float d20 = dot(v0p, v0v1);
    float d21 = dot(v0p, v0v2);
    
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0 - v - w;
    
    if (u >= 0 && v >= 0 && w >= 0) {
        // Point is inside triangle
        float3 closestPoint = u * v0 + v * v1 + w * v2;
        return distance(point, closestPoint);
    } else {
        // Point is outside triangle, find closest point on edges
        float3 ab = v1 - v0;
        float3 ap = point - v0;
        float t1 = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
        float3 closest1 = v0 + t1 * ab;
        float dist1 = distance(point, closest1);
        
        float3 bc = v2 - v1;
        float3 bp = point - v1;
        float t2 = clamp(dot(bp, bc) / dot(bc, bc), 0.0, 1.0);
        float3 closest2 = v1 + t2 * bc;
        float dist2 = distance(point, closest2);
        
        float3 ca = v0 - v2;
        float3 cp = point - v2;
        float t3 = clamp(dot(cp, ca) / dot(ca, ca), 0.0, 1.0);
        float3 closest3 = v2 + t3 * ca;
        float dist3 = distance(point, closest3);
        
        return min(dist1, min(dist2, dist3));
    }
}

// SDF generation compute shader
kernel void generateSDF(
    device const SDFTriangle* triangles [[buffer(0)]],
    device float* sdfData [[buffer(1)]],
    constant uint& triangleCount [[buffer(2)]],
    constant float3& sdfOrigin [[buffer(3)]],
    constant float3& voxelSize [[buffer(4)]],
    constant int3& resolution [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= uint(resolution.x) || 
        gid.y >= uint(resolution.y) || 
        gid.z >= uint(resolution.z)) {
        return;
    }
    
    // Calculate world position for this voxel
    float3 worldPos = sdfOrigin + float3(gid) * voxelSize;
    
    // Find minimum distance to all triangles
    float minDistance = INFINITY;
    
    for (uint i = 0; i < triangleCount; i++) {
        float distance = distanceToTriangle(worldPos, triangles[i]);
        minDistance = min(minDistance, distance);
    }
    
    // Store result
    uint index = gid.x + gid.y * uint(resolution.x) + gid.z * uint(resolution.x) * uint(resolution.y);
    sdfData[index] = minDistance;
}

// Optimized SDF generation with shared memory (for better performance)
kernel void generateSDFOptimized(
    device const SDFTriangle* triangles [[buffer(0)]],
    device float* sdfData [[buffer(1)]],
    constant uint& triangleCount [[buffer(2)]],
    constant float3& sdfOrigin [[buffer(3)]],
    constant float3& voxelSize [[buffer(4)]],
    constant int3& resolution [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Shared memory for triangle cache (adjust size based on available memory)
    threadgroup SDFTriangle sharedTriangles[64];
    
    // Check bounds
    if (gid.x >= uint(resolution.x) || 
        gid.y >= uint(resolution.y) || 
        gid.z >= uint(resolution.z)) {
        return;
    }
    
    // Calculate world position for this voxel
    float3 worldPos = sdfOrigin + float3(gid) * voxelSize;
    float minDistance = INFINITY;
    
    // Process triangles in batches to fit in shared memory
    uint batchSize = 64;
    uint numBatches = (triangleCount + batchSize - 1) / batchSize;
    
    for (uint batch = 0; batch < numBatches; batch++) {
        uint batchStart = batch * batchSize;
        uint batchEnd = min(batchStart + batchSize, triangleCount);
        uint batchCount = batchEnd - batchStart;
        
        // Load triangles into shared memory
        uint threadId = lid.x + lid.y * 8 + lid.z * 64; // Assuming 8x8x1 threadgroup
        if (threadId < batchCount) {
            uint triangleIndex = batchStart + threadId;
            if (triangleIndex < triangleCount) {
                sharedTriangles[threadId] = triangles[triangleIndex];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Calculate distances to triangles in this batch
        for (uint i = 0; i < batchCount; i++) {
            float distance = distanceToTriangle(worldPos, sharedTriangles[i]);
            minDistance = min(minDistance, distance);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    uint index = gid.x + gid.y * uint(resolution.x) + gid.z * uint(resolution.x) * uint(resolution.y);
    sdfData[index] = minDistance;
}

// Generate signed distance field (inside/outside detection)
kernel void generateSignedSDF(
    device const SDFTriangle* triangles [[buffer(0)]],
    device float* sdfData [[buffer(1)]],
    constant uint& triangleCount [[buffer(2)]],
    constant float3& sdfOrigin [[buffer(3)]],
    constant float3& voxelSize [[buffer(4)]],
    constant int3& resolution [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= uint(resolution.x) || 
        gid.y >= uint(resolution.y) || 
        gid.z >= uint(resolution.z)) {
        return;
    }
    
    float3 worldPos = sdfOrigin + float3(gid) * voxelSize;
    float minDistance = INFINITY;
    
    // Find minimum distance to all triangles
    for (uint i = 0; i < triangleCount; i++) {
        float distance = distanceToTriangle(worldPos, triangles[i]);
        minDistance = min(minDistance, distance);
    }
    
    // Simple inside/outside test using ray casting
    // Cast ray in +X direction and count intersections
    int intersectionCount = 0;
    float3 rayOrigin = worldPos;
    float3 rayDir = float3(1, 0, 0);
    
    for (uint i = 0; i < triangleCount; i++) {
        SDFTriangle tri = triangles[i];
        
        // Ray-triangle intersection test
        float3 edge1 = tri.v1 - tri.v0;
        float3 edge2 = tri.v2 - tri.v0;
        float3 h = cross(rayDir, edge2);
        float det = dot(edge1, h);
        
        if (abs(det) > 0.0001) {
            float invDet = 1.0 / det;
            float3 s = rayOrigin - tri.v0;
            float u = invDet * dot(s, h);
            
            if (u >= 0.0 && u <= 1.0) {
                float3 q = cross(s, edge1);
                float v = invDet * dot(rayDir, q);
                
                if (v >= 0.0 && u + v <= 1.0) {
                    float t = invDet * dot(edge2, q);
                    if (t > 0.0001) { // Ray intersection
                        intersectionCount++;
                    }
                }
            }
        }
    }
    
    // If odd number of intersections, point is inside
    if (intersectionCount % 2 == 1) {
        minDistance = -minDistance;
    }
    
    // Store result
    uint index = gid.x + gid.y * uint(resolution.x) + gid.z * uint(resolution.x) * uint(resolution.y);
    sdfData[index] = minDistance;
}
