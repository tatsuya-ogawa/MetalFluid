#include <metal_stdlib>
#include "../MPMTypes.h"  // Get struct definitions from shared header file
using namespace metal;

// Grid index calculation function (same as in ComputeShaders.metal)
inline uint gridIndex(uint x, uint y, uint z, int3 res) {
    return x * (res.z * res.y) + y * res.z + z;
}
// MLS-MPM rendering shaders
struct VertexOut {
    float4 position [[position]];
    float4 color;
    float pointSize [[point_size]];
};

inline float getJp(const MPMParticle particle){
    return 1.0;
}
inline float3 getColor(const MPMParticle particle){
    return float3(0.2,0.2,0.8);
}
inline float getBaseSize(const float distance){
    return max(10.0, 10.0 / (distance * 0.1 + 1.0));
}


// Pressure heatmap vertex shader
vertex VertexOut pressureHeatmapVertexShader(
                              const device MPMParticle* particles [[buffer(0)]],
                              constant VertexShaderUniforms& uniforms [[buffer(1)]],
                              const device NonAtomicMPMGridNode* grid [[buffer(2)]],
                              uint id [[vertex_id]]
                              ) {
    VertexOut out;
    
    // Convert particle position to 4D homogeneous coordinates
    float4 worldPos = float4(particles[id].position, 1.0);
    
    // Apply MVP transformation
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    
    // Sample grid data at particle position for pressure-based coloring
    // Use physicalDomainOrigin for grid calculations (same coordinate system as compute shaders)
    float3 gridPos = (particles[id].position - uniforms.physicalDomainOrigin) / uniforms.gridSpacing;
    int3 baseCell = int3(floor(gridPos - 0.5));
    
    // Sample grid mass (density/pressure) using B-spline interpolation
    float3 cellDiff = gridPos - (float3(baseCell) + 0.5);
    float3 weights[3];
    // B-spline weights calculation (inline)
    weights[0] = 0.5 * (0.5 - cellDiff) * (0.5 - cellDiff);
    weights[1] = 0.75 - cellDiff * cellDiff;
    weights[2] = 0.5 * (0.5 + cellDiff) * (0.5 + cellDiff);
    
    float gridMass = 0.0;
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                int3 cellIdx = baseCell + int3(gx-1, gy-1, gz-1);
                
                if (cellIdx.x >= 0 && cellIdx.x < uniforms.gridResolution.x &&
                    cellIdx.y >= 0 && cellIdx.y < uniforms.gridResolution.y &&
                    cellIdx.z >= 0 && cellIdx.z < uniforms.gridResolution.z) {
                    
                    uint gridIdx = gridIndex(cellIdx.x, cellIdx.y, cellIdx.z, uniforms.gridResolution);
                    
                    float weight = weights[gx].x * weights[gy].y * weights[gz].z;
                    gridMass += grid[gridIdx].mass * weight;
                }
            }
        }
    }
    
    // Convert mass density to pressure ratio (normalize by rest density)
    float density = max(0.0, gridMass);
    float pressure_ratio = min(1.0, density / uniforms.rest_density);
    
    // Pressure-based heatmap colors
    float3 low_pressure = float3(0.0, 0.2, 0.8);    // Blue (low pressure)
    float3 mid_pressure = float3(0.0, 0.8, 0.2);    // Green (medium pressure)
    float3 high_pressure = float3(0.8, 0.8, 0.0);   // Yellow (high pressure)
    float3 extreme_pressure = float3(0.8, 0.0, 0.0); // Red (extreme pressure)
    
    float3 dynamic_color;
    if (pressure_ratio < 0.33) {
        dynamic_color = mix(low_pressure, mid_pressure, pressure_ratio * 3.0);
    } else if (pressure_ratio < 0.66) {
        dynamic_color = mix(mid_pressure, high_pressure, (pressure_ratio - 0.33) * 3.0);
    } else {
        dynamic_color = mix(high_pressure, extreme_pressure, (pressure_ratio - 0.66) * 3.0);
    }
    
    // Add some velocity-based intensity variation
    float speed = length(particles[id].velocity);
    float intensity = 1.0 + min(0.5, speed * 0.1);
    dynamic_color *= intensity;
    
    out.color = float4(dynamic_color, 0.85);
    
    // Distance-based particle size adjustment - make larger and smoother
    float distance = length(out.position.xyz);
    float baseSize = getBaseSize(distance);
    out.pointSize = baseSize * uniforms.particleSizeMultiplier;
    
    return out;
}

fragment float4 pressureHeatmapFragmentShader(VertexOut in [[stage_in]],
                               float2 pointCoord [[point_coord]]) {
    // Circular particle
    float2 coord = pointCoord * 2.0 - 1.0;
    float dist = length(coord);
    if (dist > 1.0) discard_fragment();
    
    // Smooth alpha blending
    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    
    // Brighten center
    float intensity = 1.0 - dist * 0.3;
    
    return float4(in.color.rgb * intensity, in.color.a * alpha);
}


// Billboard quad vertex data for particle rendering
constant float2 cornerPositions[6] = {
    float2( 0.5,  0.5),
    float2( 0.5, -0.5),
    float2(-0.5, -0.5),
    float2( 0.5,  0.5),
    float2(-0.5, -0.5),
    float2(-0.5,  0.5)
};

// Depth generation shaders
struct DepthVertexOut {
    float4 position [[position]];
    float2 uv;
    float3 view_position;
    float sphere_size;
};

vertex DepthVertexOut vs_depth(
    const device MPMParticle* particles [[buffer(0)]],
    constant VertexShaderUniforms& uniforms [[buffer(1)]],
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]]
) {
    DepthVertexOut out;
    
    // Get corner position for billboard quad
    float2 corner_pos = cornerPositions[vertex_id];
    float3 corner = float3(corner_pos * uniforms.sphere_size, 0.0);
    out.uv = corner_pos + 0.5;
    
    // Get particle data
    float3 real_position = particles[instance_id].position;
    float3 view_position = (uniforms.viewMatrix * float4(real_position, 1.0)).xyz;
    out.view_position = view_position;
    out.sphere_size = uniforms.sphere_size;
    
    // Billboard positioning: add corner in view space, then project
    float4 view_pos_with_corner = float4(view_position + corner, 1.0);
    out.position = uniforms.projectionMatrix * view_pos_with_corner;
    
    return out;
}

struct FragmentOutput {
    float4 frag_color [[color(0)]];
    float frag_depth [[depth(any)]];
};

fragment FragmentOutput fs_depth(
    DepthVertexOut in [[stage_in]],
    constant FluidRenderUniforms& uniforms [[buffer(1)]]
) {
    FragmentOutput out;
    
    // Calculate surface normal and depth
    float2 normalxy = in.uv * 2.0 - 1.0;
    float r2 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard_fragment();
    }
    float normalz = sqrt(1.0 - r2);
    float3 normal = float3(normalxy, normalz);
    
    // Calculate sphere radius and real view position
    float radius = in.sphere_size / 2.0;
    float4 real_view_pos = float4(in.view_position + normal * radius, 1.0);
    float4 clip_space_pos = uniforms.projectionMatrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;
    
    out.frag_color = float4(real_view_pos.z, 0.0, 0.0, 1.0);
    return out;
}

// Bilateral depth filter shaders
struct FilterUniforms {
    float2 direction;     // Filter direction (1,0) for horizontal, (0,1) for vertical
    float2 screenSize;    // Screen dimensions
    float depthThreshold; // Depth difference threshold
    int filterRadius;     // Filter kernel radius
    float projectedParticleConstant; // Adaptive filter size calculation
    float maxFilterSize;  // Maximum filter size
};

struct QuadVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Full-screen quad vertex shader for depth filtering
vertex QuadVertexOut vs_bilateral(uint vid [[vertex_id]]) {
    QuadVertexOut out;
    
    // Generate full-screen quad
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    
    float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    
    out.position = float4(positions[vid], 0.0, 1.0);
    out.texCoord = texCoords[vid];
    
    return out;
}

// Bilateral depth filter fragment shader
fragment float fs_bilateral(
    QuadVertexOut in [[stage_in]],
    texture2d<float> depthTexture [[texture(0)]],
    constant FilterUniforms& filterUniforms [[buffer(0)]]
) {
    float2 iuv = in.texCoord * filterUniforms.screenSize;
    float centerDepth = abs(depthTexture.read(uint2(iuv)).r);
    
    // Skip filtering for background pixels
    if (centerDepth >= 1e4 || centerDepth <= 0.0) {
        return centerDepth;
    }
    
    // Adaptive filter size calculation
    int filterSize = min(int(filterUniforms.maxFilterSize), 
                        int(ceil(filterUniforms.projectedParticleConstant / centerDepth)));
    
    // Filter parameters
    float sigma = float(filterSize) / 3.0;
    float twoSigma = 2.0 * sigma * sigma;
    float sigmaDepth = filterUniforms.depthThreshold / 3.0;
    float twoSigmaDepth = 2.0 * sigmaDepth * sigmaDepth;
    
    float sum = 0.0;
    float wsum = 0.0;
    
    // Bilateral filter kernel
    for (int x = -filterSize; x <= filterSize; x++) {
        float2 coords = float2(float(x));
        float2 samplePos = iuv + coords * filterUniforms.direction;
        
        // Boundary check to prevent crashes
        if (samplePos.x < 0.0 || samplePos.x >= filterUniforms.screenSize.x ||
            samplePos.y < 0.0 || samplePos.y >= filterUniforms.screenSize.y) {
            continue;
        }
        
        float sampledDepth = abs(depthTexture.read(uint2(samplePos)).r);
        
        // Spatial weight
        float rr = dot(coords, coords);
        float w = exp(-rr / twoSigma);
        
        // Depth weight
        float rDepth = sampledDepth - centerDepth;
        float wd = exp(-rDepth * rDepth / twoSigmaDepth);
        
        sum += sampledDepth * w * wd;
        wsum += w * wd;
    }
    
    return sum / wsum;
}

struct FluidFragmentInput {
    float4 position [[position]];
    float2 uv;
    float2 iuv;
};

// Helper function to compute view position from UV and depth
float3 computeViewPosFromUVDepth(float2 texCoord, float depth, constant FluidRenderUniforms& uniforms) {
    // Convert screen coordinates to normalized device coordinates
    float4 ndc = float4(texCoord.x * 2.0 - 1.0, 1.0 - 2.0 * texCoord.y, 0.0, 1.0);
    
    // Calculate the z component based on projection matrix
    ndc.z = -uniforms.projectionMatrix[2].z + uniforms.projectionMatrix[3].z / depth;
    ndc.w = 1.0;
    
    // Transform to view space
    float4 viewPos = uniforms.invProjectionMatrix * ndc;
    return viewPos.xyz / viewPos.w;
}

// Helper function to get view position from texture coordinate
float3 getViewPosFromTexCoord(float2 texCoord, float2 iuv, 
                             texture2d<float> depthTexture,
                             constant FluidRenderUniforms& uniforms) {
    float depth = abs(depthTexture.read(uint2(iuv)).r);
    return computeViewPosFromUVDepth(texCoord, depth, uniforms);
}

// Fluid surface rendering vertex shader
vertex FluidFragmentInput fluidVertexShader(
    uint vid [[vertex_id]],
    constant FluidRenderUniforms& uniforms [[buffer(0)]]
) {
    FluidFragmentInput out;
    
    // Generate full-screen quad
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    
    float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = texCoords[vid];
    // Calculate screen coordinates using texel size
    float2 screenSize = 1.0 / uniforms.texelSize;
    out.iuv = texCoords[vid] * screenSize;
    
    return out;
}

// Output structure for fluid fragment shader with depth
struct FluidFragmentOutput {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

// Fluid surface rendering fragment shader
fragment FluidFragmentOutput fluidFragmentShader(
    FluidFragmentInput input [[stage_in]],
    texture2d<float> depthTexture [[texture(0)]],
    texture2d<float> thicknessTexture [[texture(1)]],
    texturecube<float> envmapTexture [[texture(2)]],
    constant FluidRenderUniforms& uniforms [[buffer(0)]],
    float4 currentColor [[color(0)]]
) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    float depth = abs(depthTexture.read(uint2(input.iuv)).r);
    float3 bgColor = currentColor.rgb;
    
    if (depth >= 1e4 || depth <= 0.0) {
        FluidFragmentOutput output;
        output.color = currentColor;
        output.depth = 1.0;  // Far depth for background
        return output;
    }
    
    // Convert view space depth to NDC depth for depth buffer
    float4 clipPos = uniforms.projectionMatrix * float4(0, 0, -depth, 1.0);
    float ndcDepth = clipPos.z / clipPos.w;
    
    float3 viewPos = computeViewPosFromUVDepth(input.uv, depth, uniforms);
    
    // Calculate surface normal using finite differences
    float3 ddx = getViewPosFromTexCoord(input.uv + float2(uniforms.texelSize.x, 0.0), 
                                       input.iuv + float2(1.0, 0.0), 
                                       depthTexture, uniforms) - viewPos;
    float3 ddy = getViewPosFromTexCoord(input.uv + float2(0.0, uniforms.texelSize.y), 
                                       input.iuv + float2(0.0, 1.0), 
                                       depthTexture, uniforms) - viewPos;
    float3 ddx2 = viewPos - getViewPosFromTexCoord(input.uv + float2(-uniforms.texelSize.x, 0.0), 
                                                  input.iuv + float2(-1.0, 0.0), 
                                                  depthTexture, uniforms);
    float3 ddy2 = viewPos - getViewPosFromTexCoord(input.uv + float2(0.0, -uniforms.texelSize.y), 
                                                  input.iuv + float2(0.0, -1.0), 
                                                  depthTexture, uniforms);
    
    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2;
    }
    if (abs(ddy.z) > abs(ddy2.z)) {
        ddy = ddy2;
    }
    
    float3 normal = -normalize(cross(ddx, ddy));
    float3 rayDir = normalize(viewPos);
    float3 lightDir = normalize((uniforms.viewMatrix * float4(0, 0, -1, 0)).xyz);
    float3 H = normalize(lightDir - rayDir);
    float specular = pow(max(0.0, dot(H, normal)), 250.0);
    
    float density = 1.5;
    
    float thickness = thicknessTexture.read(uint2(input.iuv)).r;
    float3 diffuseColor = float3(0.085, 0.6375, 0.9);
    float3 transmittance = exp(-density * thickness * (1.0 - diffuseColor));
    float3 refractionColor = bgColor * transmittance;
    
    float F0 = 0.02;
    float fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0), 0.0, 1.0);
    
    float3 reflectionDir = reflect(rayDir, normal);
    float3 reflectionDirWorld = (uniforms.invViewMatrix * float4(reflectionDir, 0.0)).xyz;
    float3 reflectionColor = envmapTexture.sample(textureSampler, reflectionDirWorld).rgb;
    float3 finalColor = 1.0 * specular + mix(refractionColor, reflectionColor, fresnel);
    
    FluidFragmentOutput output;
    output.color = float4(finalColor, 1.0);
    output.depth = ndcDepth;  // Output the fluid surface depth
    return output;
}

// Thickness rendering shaders
struct ThicknessVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex ThicknessVertexOut vs_thickness(
    const device MPMParticle* particles [[buffer(0)]],
    constant VertexShaderUniforms& uniforms [[buffer(1)]],
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]]
) {
    ThicknessVertexOut out;
    
    // Get corner position for billboard quad
    float2 corner_pos = cornerPositions[vertex_id];
    float3 corner = float3(corner_pos * uniforms.sphere_size, 0.0);
    out.uv = corner_pos + 0.5;
    
    // Get particle data
    float3 real_position = particles[instance_id].position;
    float3 view_position = (uniforms.viewMatrix * float4(real_position, 1.0)).xyz;
    
    // Billboard positioning: add corner in view space, then project
    float4 view_pos_with_corner = float4(view_position + corner, 1.0);
    out.position = uniforms.projectionMatrix * view_pos_with_corner;
    
    return out;
}

fragment float fs_thickness(
    ThicknessVertexOut in [[stage_in]]
) {
    // Calculate thickness contribution
    float2 normalxy = in.uv * 2.0 - 1.0;
    float r2 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard_fragment();
    }
    float thickness = sqrt(1.0 - r2);
    float particle_alpha = 0.05;
    
    return particle_alpha * thickness;
}

// Gaussian filter shaders for thickness texture
struct GaussianUniforms {
    float2 direction;     // Filter direction (1,0) for horizontal, (0,1) for vertical
    float2 screenSize;    // Screen dimensions
    int filterRadius;     // Filter kernel radius
};

// Gaussian filter fragment shader
fragment float fs_gaussian(
    QuadVertexOut in [[stage_in]],
    texture2d<float> inputTexture [[texture(0)]],
    constant GaussianUniforms& gaussianUniforms [[buffer(0)]]
) {
    float2 iuv = in.texCoord * gaussianUniforms.screenSize;
    float thickness = inputTexture.read(uint2(iuv)).r;
    
    if (thickness == 0.0) {
        return 0.0;
    }
    
    // Fixed filter size
    int filterSize = 30;
    float sigma = float(filterSize) / 3.0;
    float twoSigma = 2.0 * sigma * sigma;
    
    float sum = 0.0;
    float wsum = 0.0;
    
    // Apply 1D Gaussian blur
    for (int x = -filterSize; x <= filterSize; x++) {
        float2 coords = float2(float(x));
        float2 samplePos = iuv + gaussianUniforms.direction * coords;
        
        // Boundary check to prevent crashes
        if (samplePos.x < 0.0 || samplePos.x >= gaussianUniforms.screenSize.x ||
            samplePos.y < 0.0 || samplePos.y >= gaussianUniforms.screenSize.y) {
            continue;
        }
        
        float sampledThickness = inputTexture.read(uint2(samplePos)).r;
        
        float w = exp(-coords.x * coords.x / twoSigma);
        
        sum += sampledThickness * w;
        wsum += w;
    }
    
    sum /= wsum;
    
    return sum;
}
