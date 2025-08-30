#include <metal_stdlib>
#include "../MPMTypes.h"
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
    constant VertexShaderUniforms& vertexUniforms [[buffer(1)]],
    constant CollisionMeshUniforms& meshUniforms [[buffer(2)]],
    constant CollisionUniforms& collisionUniforms [[buffer(3)]],
    uint vertexID [[vertex_id]]
) {
    CollisionVertexOut out;
    
    // Apply collision transform to vertex position
    float4 meshPos4 = float4(vertices[vertexID].position, 1.0);
    float4 worldPos4 = collisionUniforms.collisionTransform * meshPos4;
    float3 worldPos = worldPos4.xyz;
    out.worldPosition = worldPos;
    out.position = vertexUniforms.projectionMatrix * vertexUniforms.viewMatrix * float4(worldPos, 1.0);
    out.normal = vertices[vertexID].normal;
    
    // Use mesh color from uniforms
    out.color = meshUniforms.meshColor;
    
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
    constant VertexShaderUniforms& vertexUniforms [[buffer(1)]],
    constant CollisionMeshUniforms& meshUniforms [[buffer(2)]],
    constant CollisionUniforms& collisionUniforms [[buffer(3)]],
    uint vertexID [[vertex_id]]
) {
    CollisionVertexOut out;
    
    // Apply collision transform to vertex position
    float4 meshPos4 = float4(vertices[vertexID].position, 1.0);
    float4 worldPos4 = collisionUniforms.collisionTransform * meshPos4;
    float3 worldPos = worldPos4.xyz;
    out.worldPosition = worldPos;
    out.position = vertexUniforms.projectionMatrix * vertexUniforms.viewMatrix * float4(worldPos, 1.0);
    out.normal = vertices[vertexID].normal;
    
    // Use mesh color from uniforms with higher opacity for wireframe
    out.color = meshUniforms.meshColor;//float4(meshUniforms.meshColor.rgb, max(meshUniforms.meshColor.a, 0.8));
    
    return out;
}

// Wireframe fragment shader
fragment float4 collisionMeshWireframeFragmentShader(
    CollisionVertexOut in [[stage_in]]
) {
    return in.color; // Flat color for wireframe
}
