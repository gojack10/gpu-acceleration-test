// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

struct Uniforms {
    model_view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = uniforms.model_view_proj * vec4<f32>(model.position, 1.0);
    out.normal = model.normal;
    return out;
}

// Fragment shader

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}

// Specialized fragment shader for debug axes that uses vertex normal as color
@fragment
fn fs_debug_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use normal as color (red = x, green = y, blue = z)
    // Convert from [-1, 1] to [0, 1] range for color
    let color = abs(in.normal);
    return vec4<f32>(color, 1.0);
} 