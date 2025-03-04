#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// Uniform data
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub model_view_proj: [[f32; 4]; 4],
}

pub fn create_cube_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertices = vec![
        // Front face
        Vertex { position: [-0.5, -0.5, 0.5], tex_coords: [0.0, 1.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [0.5, -0.5, 0.5], tex_coords: [1.0, 1.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [0.5, 0.5, 0.5], tex_coords: [1.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [-0.5, 0.5, 0.5], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Back face
        Vertex { position: [-0.5, -0.5, -0.5], tex_coords: [1.0, 1.0], normal: [0.0, 0.0, -1.0] },
        Vertex { position: [-0.5, 0.5, -0.5], tex_coords: [1.0, 0.0], normal: [0.0, 0.0, -1.0] },
        Vertex { position: [0.5, 0.5, -0.5], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, -1.0] },
        Vertex { position: [0.5, -0.5, -0.5], tex_coords: [0.0, 1.0], normal: [0.0, 0.0, -1.0] },
        
        // Top face
        Vertex { position: [-0.5, 0.5, -0.5], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [-0.5, 0.5, 0.5], tex_coords: [0.0, 0.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.5, 0.5, 0.5], tex_coords: [1.0, 0.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.5, 0.5, -0.5], tex_coords: [1.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Bottom face
        Vertex { position: [-0.5, -0.5, -0.5], tex_coords: [1.0, 1.0], normal: [0.0, -1.0, 0.0] },
        Vertex { position: [0.5, -0.5, -0.5], tex_coords: [0.0, 1.0], normal: [0.0, -1.0, 0.0] },
        Vertex { position: [0.5, -0.5, 0.5], tex_coords: [0.0, 0.0], normal: [0.0, -1.0, 0.0] },
        Vertex { position: [-0.5, -0.5, 0.5], tex_coords: [1.0, 0.0], normal: [0.0, -1.0, 0.0] },
        
        // Right face
        Vertex { position: [0.5, -0.5, -0.5], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [0.5, 0.5, -0.5], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [0.5, 0.5, 0.5], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [0.5, -0.5, 0.5], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
        
        // Left face
        Vertex { position: [-0.5, -0.5, -0.5], tex_coords: [0.0, 1.0], normal: [-1.0, 0.0, 0.0] },
        Vertex { position: [-0.5, -0.5, 0.5], tex_coords: [1.0, 1.0], normal: [-1.0, 0.0, 0.0] },
        Vertex { position: [-0.5, 0.5, 0.5], tex_coords: [1.0, 0.0], normal: [-1.0, 0.0, 0.0] },
        Vertex { position: [-0.5, 0.5, -0.5], tex_coords: [0.0, 0.0], normal: [-1.0, 0.0, 0.0] },
    ];

    let indices = vec![
        0, 1, 2, 2, 3, 0, // front
        4, 5, 6, 6, 7, 4, // back
        8, 9, 10, 10, 11, 8, // top
        12, 13, 14, 14, 15, 12, // bottom
        16, 17, 18, 18, 19, 16, // right
        20, 21, 22, 22, 23, 20, // left
    ];

    (vertices, indices)
} 