#include <metal_stdlib>
#include "MPMTypes.h"
using namespace metal;

// Collision mesh vertex structure
struct CollisionVertex {
    float3 position;
    float3 normal;
};

struct CollisionVertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float4 color;
};

// Collision mesh vertex shader
vertex CollisionVertexOut collisionMeshVertexShader(
    const device CollisionVertex* vertices [[buffer(0)]],
    constant VertexShaderUniforms& uniforms [[buffer(1)]],
    uint vertexID [[vertex_id]]
) {
    CollisionVertexOut out;
    
    float3 worldPos = vertices[vertexID].position;
    out.worldPosition = worldPos;
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * float4(worldPos, 1.0);
    out.normal = vertices[vertexID].normal;
    
    // Semi-transparent collision mesh color
    out.color = float4(0.8, 0.3, 0.3, 0.4); // Red with transparency
    
    return out;
}

// Collision mesh fragment shader
fragment float4 collisionMeshFragmentShader(
    CollisionVertexOut in [[stage_in]]
) {
    // Simple shading based on normal
    float3 lightDir = normalize(float3(0.0, 1.0, 1.0));
    float ndotl = max(0.2, dot(normalize(in.normal), lightDir));
    
    float4 finalColor = in.color;
    finalColor.rgb *= ndotl;
    
    return finalColor;
}

// Wireframe vertex shader (for debugging)
vertex CollisionVertexOut collisionMeshWireframeVertexShader(
    const device CollisionVertex* vertices [[buffer(0)]],
    constant VertexShaderUniforms& uniforms [[buffer(1)]],
    uint vertexID [[vertex_id]]
) {
    CollisionVertexOut out;
    
    float3 worldPos = vertices[vertexID].position;
    out.worldPosition = worldPos;
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * float4(worldPos, 1.0);
    out.normal = vertices[vertexID].normal;
    
    // Wireframe color - more opaque
    out.color = float4(0.9, 0.1, 0.1, 0.8); // Bright red wireframe
    
    return out;
}

// Wireframe fragment shader
fragment float4 collisionMeshWireframeFragmentShader(
    CollisionVertexOut in [[stage_in]]
) {
    return in.color; // Flat color for wireframe
}