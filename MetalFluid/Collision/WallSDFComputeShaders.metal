#include <metal_stdlib>
using namespace metal;

// Generate an SDF representing thin walls (6 faces) around a box domain
// phi = d - thickness, where d is min distance to any of the 6 planes of the box
kernel void generateWallSDF(
    texture3d<float, access::write> sdfTexture [[texture(0)]],
    constant float3& sdfOrigin [[buffer(0)]],
    constant float3& sdfSize [[buffer(1)]],
    constant int3& sdfResolution [[buffer(2)]],
    constant float& wallThickness [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= sdfTexture.get_width() || gid.y >= sdfTexture.get_height() || gid.z >= sdfTexture.get_depth()) {
        return;
    }

    // Compute world position at voxel center
    float3 uv = (float3(gid) + 0.5) / float3(sdfResolution);
    float3 worldPos = sdfOrigin + uv * sdfSize;

    // Distances to each wall plane within the box [origin, origin+size]
    float3 toMin = worldPos - sdfOrigin;
    float3 toMax = (sdfOrigin + sdfSize) - worldPos;
    float d = min(min(toMin.x, toMax.x), min(min(toMin.y, toMax.y), min(toMin.z, toMax.z)));

    // SDF: negative inside the thin wall band, positive elsewhere
    float phi = d - wallThickness;
    
    // Write SDF value
    sdfTexture.write(phi, gid);
}

