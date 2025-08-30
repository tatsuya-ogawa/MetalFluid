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
inline float unsignedDistanceToTriangle(float3 point, SDFTriangle triangle) {
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
inline float signedDistanceToTriangle(float3 point, SDFTriangle triangle) {
    float d = unsignedDistanceToTriangle(point, triangle);

    // 法線で符号付け（単一三角形ならこれでOK）
    float3 v0 = triangle.v0;
    float3 v1 = triangle.v1;
    float3 v2 = triangle.v2;

    float3 n = normalize(cross(v1 - v0, v2 - v0));

    // 最近傍点を平面に投影
    float3 projected = point - dot(point - v0, n) * n;

    // 法線方向で内外判定（注意: メッシュ全体だと不十分）
    float signVal = dot(point - projected, n);

    return signVal >= 0.0 ? d : -d;
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
        float distance = signedDistanceToTriangle(worldPos, triangles[i]);
        if(abs(distance) < abs(minDistance)){
            minDistance = distance;
        }
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
            float distance = signedDistanceToTriangle(worldPos, sharedTriangles[i]);
            if(abs(distance) < abs(minDistance)){
                minDistance = distance;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    uint index = gid.x + gid.y * uint(resolution.x) + gid.z * uint(resolution.x) * uint(resolution.y);
    sdfData[index] = minDistance;
}
