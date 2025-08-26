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

// Vertex input structure for AR mesh
struct ARMeshVertex {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
};

// Vertex output structure for AR mesh
struct ARMeshVertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float depth;
};

// Uniforms for AR mesh rendering
struct ARMeshUniforms {
    float4x4 modelViewProjectionMatrix;
    float4x4 modelMatrix;
    float4x4 normalMatrix;
    float4 meshColor;
    float opacity;
};

// AR Mesh Vertex Shader
vertex ARMeshVertexOut arMeshVertex(ARMeshVertex in [[stage_in]],
                                   constant ARMeshUniforms& uniforms [[buffer(0)]]) {
    ARMeshVertexOut out;
    
    float4 worldPosition = uniforms.modelMatrix * float4(in.position, 1.0);
    out.worldPosition = worldPosition.xyz;
    out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
    out.normal = (uniforms.normalMatrix * float4(in.normal, 0.0)).xyz;
    out.depth = out.position.z / out.position.w;
    
    return out;
}

// AR Mesh Fragment Shader (wireframe style)
fragment float4 arMeshFragment(ARMeshVertexOut in [[stage_in]],
                              constant ARMeshUniforms& uniforms [[buffer(0)]]) {
    // Simple lighting calculation
    float3 lightDirection = normalize(float3(0.0, 1.0, 1.0));
    float3 normal = normalize(in.normal);
    float lightIntensity = max(dot(normal, lightDirection), 0.3);
    
    float3 baseColor = uniforms.meshColor.rgb;
    float3 litColor = baseColor * lightIntensity;
    
    // Add depth-based fade
    float depthFade = 1.0 - smoothstep(0.0, 1.0, in.depth);
    float alpha = uniforms.opacity * depthFade;
    
    return float4(litColor, alpha);
}

// AR Mesh Fragment Shader (solid style with transparency)
fragment float4 arMeshSolidFragment(ARMeshVertexOut in [[stage_in]],
                                   constant ARMeshUniforms& uniforms [[buffer(0)]]) {
    // Simple lighting
    float3 lightDirection = normalize(float3(0.0, 1.0, 1.0));
    float3 normal = normalize(in.normal);
    float lightIntensity = max(dot(normal, lightDirection), 0.4);
    
    float3 baseColor = uniforms.meshColor.rgb;
    float3 litColor = baseColor * lightIntensity;
    
    return float4(litColor, uniforms.opacity);
}

// AR Mesh Fragment Shader (wireframe style)
fragment float4 arMeshWireframeFragment(ARMeshVertexOut in [[stage_in]],
                                       constant ARMeshUniforms& uniforms [[buffer(0)]]) {
    return float4(uniforms.meshColor.rgb, uniforms.opacity);
}