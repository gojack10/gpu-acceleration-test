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
        // Front face (Face Index 0 in Blender)
        // Positions are scaled by 0.5 to match our -0.5 to 0.5 cube size (Blender uses -1.0 to 1.0)
        Vertex { position: [0.5, 0.5, 0.5], tex_coords: [0.5000, 0.6667], normal: [0.0, 0.0, 1.0] },       // Vertex 0
        Vertex { position: [-0.5, 0.5, 0.5], tex_coords: [0.5000, 1.0000], normal: [0.0, 0.0, 1.0] },      // Vertex 4
        Vertex { position: [-0.5, -0.5, 0.5], tex_coords: [0.2502, 1.0000], normal: [0.0, 0.0, 1.0] },     // Vertex 6
        Vertex { position: [0.5, -0.5, 0.5], tex_coords: [0.2502, 0.6667], normal: [0.0, 0.0, 1.0] },      // Vertex 2
        
        // Bottom face (Face Index 1 in Blender)
        Vertex { position: [0.5, -0.5, -0.5], tex_coords: [0.2502, 0.3333], normal: [0.0, -1.0, 0.0] },    // Vertex 3
        Vertex { position: [0.5, -0.5, 0.5], tex_coords: [0.2502, 0.6667], normal: [0.0, -1.0, 0.0] },     // Vertex 2
        Vertex { position: [-0.5, -0.5, 0.5], tex_coords: [0.0004, 0.6667], normal: [0.0, -1.0, 0.0] },    // Vertex 6
        Vertex { position: [-0.5, -0.5, -0.5], tex_coords: [0.0004, 0.3333], normal: [0.0, -1.0, 0.0] },   // Vertex 7
        
        // Left face (Face Index 2 in Blender)
        Vertex { position: [-0.5, -0.5, -0.5], tex_coords: [0.9996, 0.3333], normal: [-1.0, 0.0, 0.0] },   // Vertex 7
        Vertex { position: [-0.5, -0.5, 0.5], tex_coords: [0.9996, 0.6667], normal: [-1.0, 0.0, 0.0] },    // Vertex 6
        Vertex { position: [-0.5, 0.5, 0.5], tex_coords: [0.7498, 0.6667], normal: [-1.0, 0.0, 0.0] },     // Vertex 4
        Vertex { position: [-0.5, 0.5, -0.5], tex_coords: [0.7498, 0.3333], normal: [-1.0, 0.0, 0.0] },    // Vertex 5
        
        // Back face (Face Index 3 in Blender)
        Vertex { position: [-0.5, 0.5, -0.5], tex_coords: [0.5000, 0.0000], normal: [0.0, 0.0, -1.0] },    // Vertex 5
        Vertex { position: [0.5, 0.5, -0.5], tex_coords: [0.5000, 0.3333], normal: [0.0, 0.0, -1.0] },     // Vertex 1
        Vertex { position: [0.5, -0.5, -0.5], tex_coords: [0.2502, 0.3333], normal: [0.0, 0.0, -1.0] },    // Vertex 3
        Vertex { position: [-0.5, -0.5, -0.5], tex_coords: [0.2502, 0.0000], normal: [0.0, 0.0, -1.0] },   // Vertex 7
        
        // Right face (Face Index 4 in Blender)
        Vertex { position: [0.5, 0.5, -0.5], tex_coords: [0.5000, 0.3333], normal: [1.0, 0.0, 0.0] },      // Vertex 1
        Vertex { position: [0.5, 0.5, 0.5], tex_coords: [0.5000, 0.6667], normal: [1.0, 0.0, 0.0] },       // Vertex 0
        Vertex { position: [0.5, -0.5, 0.5], tex_coords: [0.2502, 0.6667], normal: [1.0, 0.0, 0.0] },      // Vertex 2
        Vertex { position: [0.5, -0.5, -0.5], tex_coords: [0.2502, 0.3333], normal: [1.0, 0.0, 0.0] },     // Vertex 3
        
        // Top face (Face Index 5 in Blender)
        Vertex { position: [-0.5, 0.5, -0.5], tex_coords: [0.7498, 0.3333], normal: [0.0, 1.0, 0.0] },     // Vertex 5
        Vertex { position: [-0.5, 0.5, 0.5], tex_coords: [0.7498, 0.6667], normal: [0.0, 1.0, 0.0] },      // Vertex 4
        Vertex { position: [0.5, 0.5, 0.5], tex_coords: [0.5000, 0.6667], normal: [0.0, 1.0, 0.0] },       // Vertex 0
        Vertex { position: [0.5, 0.5, -0.5], tex_coords: [0.5000, 0.3333], normal: [0.0, 1.0, 0.0] },      // Vertex 1
    ];

    let indices = vec![
        0, 1, 2, 2, 3, 0, // front
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // left
        12, 13, 14, 14, 15, 12, // back
        16, 17, 18, 18, 19, 16, // right
        20, 21, 22, 22, 23, 20, // top
    ];

    (vertices, indices)
} 