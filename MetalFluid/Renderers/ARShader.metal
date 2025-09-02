//
//  ARShader.metal
//  MetalFluid
//
//  ARKit background and mesh rendering shaders
//

#include <metal_stdlib>
using namespace metal;

// Vertex input structure for AR camera background
struct CameraVertex {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

// Vertex output structure for AR camera background
struct CameraVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// AR Camera Background Vertex Shader
vertex CameraVertexOut cameraBackgroundVertex(CameraVertex in [[stage_in]]) {
    CameraVertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

// AR Camera Background Fragment Shader
fragment float4 cameraBackgroundFragment(CameraVertexOut in [[stage_in]],
                                        texture2d<float> cameraImageTextureY [[texture(0)]],
                                        texture2d<float> cameraImageTextureCbCr [[texture(1)]]) {
    constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    
    // Sample Y and CbCr textures
    float y = cameraImageTextureY.sample(colorSampler, in.texCoord).r;
    float2 cbcr = cameraImageTextureCbCr.sample(colorSampler, in.texCoord).rg;
    
    // Convert YCbCr to RGB
    float cb = cbcr.r;
    float cr = cbcr.g;
    
    float r = y + 1.402 * (cr - 0.5);
    float g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5);
    float b = y + 1.772 * (cb - 0.5);
    
    return float4(r, g, b, 1.0);
}

// MARK: - AR Mesh Wireframe Shaders

// AR Mesh Vertex Shader
vertex float4 arMeshWireVertex(const device float3* position [[buffer(0)]],
                              constant float4x4& projectionMatrix [[buffer(1)]],
                              constant float4x4& viewMatrix [[buffer(2)]],
                              uint vid [[vertex_id]]) {
    float4 worldPosition = float4(position[vid], 1.0);
    return projectionMatrix * viewMatrix * worldPosition;
}

// AR Mesh Fragment Shader
fragment float4 arMeshWireFragment(constant float4& color [[buffer(0)]]) {
    return color;
}

// MARK: - AR Mesh Solid Shaders

struct MeshVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
};

// AR Mesh Solid Vertex Shader
vertex MeshVertexOut arMeshSolidVertex(const device float3* position [[buffer(0)]],
                                      const device float3* normal [[buffer(1)]],
                                      constant float4x4& projectionMatrix [[buffer(2)]],
                                      constant float4x4& viewMatrix [[buffer(3)]],
                                      uint vid [[vertex_id]]) {
    MeshVertexOut out;
    float4 worldPosition = float4(position[vid], 1.0);
    out.position = projectionMatrix * viewMatrix * worldPosition;
    out.worldPos = position[vid];
    out.normal = normal[vid];
    return out;
}

// AR Mesh Solid Fragment Shader with basic lighting
fragment float4 arMeshSolidFragment(MeshVertexOut in [[stage_in]],
                                   constant float4& baseColor [[buffer(0)]],
                                   constant float3& lightDirection [[buffer(1)]]) {
    float3 normal = normalize(in.normal);
    float3 lightDir = normalize(-lightDirection);
    
    // Simple diffuse lighting
    float diffuse = max(dot(normal, lightDir), 0.0);
    float3 color = baseColor.rgb * (0.3 + 0.7 * diffuse); // ambient + diffuse
    
    return float4(color, baseColor.a * 0.7); // Semi-transparent
}

// MARK: - GPU Raycast Compute Kernels

struct RaycastUniforms {
    float3 rayOrigin;
    float3 rayDirection;
    uint triangleCount;
    float3 boundingBoxCenter;
    float3 boundingBoxSize;
};

struct Triangle {
    float3 v0, v1, v2;
};

struct RaycastResult {
    float3 hitPoint;
    float distance;
    uint triangleIndex;
    uint hit;  // Use uint instead of bool for compatibility
    uint3 _padding;  // Ensure proper alignment
};

// Ray-triangle intersection using Möller-Trumbore algorithm
bool rayTriangleIntersection(float3 rayOrigin, float3 rayDirection,
                           float3 v0, float3 v1, float3 v2,
                           thread float& t, thread float& u, thread float& v) {
    const float epsilon = 1e-6f;
    
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(rayDirection, edge2);
    float a = dot(edge1, h);
    
    if (abs(a) < epsilon) return false; // Ray is parallel to triangle
    
    float f = 1.0f / a;
    float3 s = rayOrigin - v0;
    u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return false;
    
    float3 q = cross(s, edge1);
    v = f * dot(rayDirection, q);
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    t = f * dot(edge2, q);
    
    return (t > epsilon);
}

// Check if triangle overlaps with bounding box
bool triangleOverlapsBoundingBox(float3 v0, float3 v1, float3 v2,
                               float3 minBounds, float3 maxBounds) {
    // Check if any vertex is inside the bounding box
    if ((all(v0 >= minBounds) && all(v0 <= maxBounds)) ||
        (all(v1 >= minBounds) && all(v1 <= maxBounds)) ||
        (all(v2 >= minBounds) && all(v2 <= maxBounds))) {
        return true;
    }
    
    // Check if triangle's bounding box overlaps with our bounding box
    float3 triMin = min(min(v0, v1), v2);
    float3 triMax = max(max(v0, v1), v2);
    
    return all(triMax >= minBounds) && all(triMin <= maxBounds);
}

// GPU Raycast Compute Kernel - finds closest hit point
kernel void performGPURaycast(constant RaycastUniforms& uniforms [[buffer(0)]],
                             const device Triangle* triangles [[buffer(1)]],
                             device RaycastResult& result [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    
    // Initialize result for this thread
    if (tid == 0) {
        result.hit = 0u;
        result.distance = INFINITY;
        result.triangleIndex = 0;
        result.hitPoint = float3(0.0f);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    if (tid >= uniforms.triangleCount) return;
    
    Triangle tri = triangles[tid];
    
    float t, u, v;
    if (rayTriangleIntersection(uniforms.rayOrigin, uniforms.rayDirection,
                               tri.v0, tri.v1, tri.v2, t, u, v)) {
        
        // Use atomic operations to find the closest hit
        device atomic<float>* atomicDistance = (device atomic<float>*)&result.distance;
        float currentDistance = atomic_load_explicit(atomicDistance, memory_order_relaxed);
        
        if (t < currentDistance) {
            // Try to update with closer hit
            float expectedDistance = currentDistance;
            while (t < expectedDistance && 
                   !atomic_compare_exchange_weak_explicit(atomicDistance, &expectedDistance, t,
                                                        memory_order_relaxed, memory_order_relaxed)) {
                // Loop until we successfully update or find a closer hit
            }
            
            // If we successfully updated the distance, update other fields
            if (t == atomic_load_explicit(atomicDistance, memory_order_relaxed)) {
                result.hitPoint = uniforms.rayOrigin + uniforms.rayDirection * t;
                result.triangleIndex = tid;
                result.hit = 1u;
            }
        }
    }
}

// GPU Mesh Extraction Compute Kernel - extracts triangles within bounding box
kernel void extractMeshesInBoundingBox(constant RaycastUniforms& uniforms [[buffer(0)]],
                                      const device Triangle* inputTriangles [[buffer(1)]],
                                      device Triangle* outputTriangles [[buffer(2)]],
                                      device atomic<uint>* outputCount [[buffer(3)]],
                                      uint tid [[thread_position_in_grid]]) {
    
    if (tid >= uniforms.triangleCount) return;
    
    Triangle tri = inputTriangles[tid];
    
    float3 halfSize = uniforms.boundingBoxSize * 0.5f;
    float3 minBounds = uniforms.boundingBoxCenter - halfSize;
    float3 maxBounds = uniforms.boundingBoxCenter + halfSize;
    
    if (triangleOverlapsBoundingBox(tri.v0, tri.v1, tri.v2, minBounds, maxBounds)) {
        // Atomically get next output index
        uint outputIndex = atomic_fetch_add_explicit(outputCount, 1u, memory_order_relaxed);
        
        // Store triangle in output buffer
        outputTriangles[outputIndex] = tri;
    }
}
