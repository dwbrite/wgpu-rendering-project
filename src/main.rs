use cgmath::prelude::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use wgpu::{RequestAdapterOptions, SwapChainDescriptor, CommandEncoderDescriptor, RenderPass, VertexBufferDescriptor, BindGroupLayoutDescriptor, TextureViewDimension};
use itertools::Itertools;
use gltf::image::Source;
use futures::FutureExt;

struct GraphicsContext {
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,

    size: winit::dpi::PhysicalSize<u32>,
}

impl GraphicsContext {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(
            &RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY,
        ).await.unwrap();
        
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false
            },
            limits: Default::default()
        }).await;

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        GraphicsContext {
            surface,
            adapter,
            device,
            queue,
            sc_desc,
            swap_chain,
            size
        }
    }
}

struct Foo {
    ctx: GraphicsContext,
    render_pipeline: wgpu::RenderPipeline,
    test_model: Model
}

// main.rs
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GltfVertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coords: [f32; 2],
}

unsafe impl bytemuck::Pod for GltfVertex {}
unsafe impl bytemuck::Zeroable for GltfVertex {}


// impl Vertex for GltfVertex
impl GltfVertex {
    fn desc<'a>() -> VertexBufferDescriptor<'a> {
        // there ought to be a macro for this...
        VertexBufferDescriptor {
            stride: std::mem::size_of::<GltfVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor { // position
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor { // color
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor { // tex_coords
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float2,
                }
            ]
        }
    }
}

struct PrimitiveIntermediate<'a> {
    vertices: Vec<GltfVertex>,
    indices: Vec<u32>,
    tex_diffuse: TexInfo<'a>,
    // textures?
}

struct GltfPrimitive {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

struct ModelIntermediate<'a> {
    primitives: Vec<PrimitiveIntermediate<'a>>,
}

struct Model {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    diffuse_texture: WgpuTexture,
    num_indices: u32,
}

struct TexInfo<'a> {
    uri: &'a str,
    idx: Option<usize>,
}

struct WgpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl Model {
    fn from_gltf(uri: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (gltf, buffers, images) = gltf::import(uri).unwrap();

        let mut submeshes: Vec<ModelIntermediate> = Vec::new();

        //let doc, _ = gltf::open()
        //let buf = gltf::import::import_buffer_data();

        // TODO: load all applicable attributes
        for mesh in gltf.meshes() {
            println!("Mesh #{}", mesh.index());

            let mut submesh = ModelIntermediate {
                primitives: Vec::new()
            };

            // a primitive is a section of mesh with one material
            // not to be confused with primitive topology
            for primitive in mesh.primitives() {
                let mut vertices = Vec::<GltfVertex>::new();
                let mut indices = Vec::<u32>::new();

                // for each primitive, we need to create a list of vertices
                // each vertex has a position, normal, tangent, and optionally color and tex_coords

                println!("- Primitive #{}", primitive.index());
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions: Vec<[f32;3]> = reader.read_positions().map_or(
                    Vec::new(),
                    |iter| iter.collect_vec()
                );
                let num_vertices = positions.len();

                // TODO: maybe rgba vert colors? (nah, just use texture transparency)
                let set = 0; // assume each vertex only has one color
                let colors : Vec<[f32;3]> = reader.read_colors(set).map_or(
                    vec![[1.0, 1.0, 1.0]; num_vertices],
                    |color_iter| color_iter.into_rgb_f32().collect_vec()
                );
                assert_eq!(colors.len(), num_vertices, "at least one vertex is missing a color!");

                let set = 0; // TODO: _maybe_ don't assume each vertex only has one tex coord
                let tex_coords: Vec<[f32;2]> = reader.read_tex_coords(set).map_or(
                    vec![[0.5, 0.5]; num_vertices],
                    |iter| iter.into_f32().collect_vec()
                );
                assert_eq!(tex_coords.len(), num_vertices, "at least one vertex is missing a tex coord!");

                // push vertices.push
                for i in 0..num_vertices {
                    vertices.push(
                        GltfVertex {
                            position: positions[i],
                            color: colors[i],
                            tex_coords: tex_coords[i],
                        }
                    )
                }

                indices = reader.read_indices().map_or(
                    Vec::new(),
                    |iter| iter.into_u32().collect_vec()
                );
                // is this right? 🤔
                assert!(indices.len() >= num_vertices,
                        "I don't think you can have more vertices than indices. ");

                // TODO: sometimes don't load an image (
                let mut diffuse_info = TexInfo {
                    uri: "../tex/uv_grid.png",
                    idx: None
                };

                let texture = primitive.material().pbr_metallic_roughness().base_color_texture();
                if let Some(info) = texture {
                    let src = info.texture().source().source();
                    if let Source::Uri { uri, mime_type } = src {
                        diffuse_info = TexInfo {
                            uri,
                            idx: Some(info.texture().source().index())
                        };
                        assert!(diffuse_info.idx.unwrap() < images.len());
                    }
                }

                submesh.primitives.push(PrimitiveIntermediate {
                    vertices,
                    indices,
                    tex_diffuse: diffuse_info
                });
            }
            submeshes.push(submesh);
        }

        let mut model: Option<Model> = None;

        for submesh in submeshes {
            for primitive in submesh.primitives {
                let vertex_buffer = device.create_buffer_with_data(
                    bytemuck::cast_slice(&primitive.vertices),
                    wgpu::BufferUsage::VERTEX
                );

                let index_buffer = device.create_buffer_with_data(
                    bytemuck::cast_slice(&primitive.indices),
                    wgpu::BufferUsage::INDEX,
                );

                // TODO: don't hardcode
                let size = wgpu::Extent3d {
                    width: 1024,
                    height: 1024,
                    depth: 1
                };

                let texture = {
                    let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor{
                        label: Some("big boi diffuse texture"),
                        size,
                        array_layer_count: 1,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                    });

                    let buffer = device.create_buffer_with_data(
                        &images.get(primitive.tex_diffuse.idx.unwrap()).unwrap().pixels,
                        wgpu::BufferUsage::COPY_SRC
                    );

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("texture loader command")
                    });

                    encoder.copy_buffer_to_texture(
                        wgpu::BufferCopyView {
                            buffer: &buffer,
                            offset: 0,
                            bytes_per_row: 4 * size.width,
                            rows_per_image: size.height
                        },
                        wgpu::TextureCopyView {
                            texture: &diffuse_texture,
                            mip_level: 0,
                            array_layer: 0,
                            origin: wgpu::Origin3d::ZERO
                        },
                        size
                    );

                    queue.submit(&[encoder.finish()]);

                    let diffuse_texture_view = diffuse_texture.create_default_view();

                    let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        address_mode_u: wgpu::AddressMode::ClampToEdge,
                        address_mode_v: wgpu::AddressMode::ClampToEdge,
                        address_mode_w: wgpu::AddressMode::ClampToEdge,
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Nearest,
                        mipmap_filter: wgpu::FilterMode::Nearest,
                        lod_min_clamp: -100.0,
                        lod_max_clamp: 100.0,
                        compare: wgpu::CompareFunction::Always,
                    });

                    let texture_bind_group_layout = device.create_bind_group_layout(
                        &BindGroupLayoutDescriptor {
                            bindings: &[
                                wgpu::BindGroupLayoutEntry{
                                    binding: 0,
                                    visibility: wgpu::ShaderStage::FRAGMENT,
                                    ty: wgpu::BindingType::SampledTexture {
                                        dimension: TextureViewDimension::D2,
                                        component_type: wgpu::TextureComponentType::Uint,
                                        multisampled: false
                                    }
                                },
                                wgpu::BindGroupLayoutEntry{
                                    binding: 1,
                                    visibility: wgpu::ShaderStage::FRAGMENT,
                                    ty: wgpu::BindingType::Sampler {
                                        comparison: false
                                    }
                                }
                            ],
                            label: Some("texture_bind_layout")
                        }
                    );

                    let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &texture_bind_group_layout,
                        bindings: &[
                            wgpu::Binding {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                            },
                            wgpu::Binding {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                            }
                        ],
                        label: Some("diffuse_bind_group"),
                    });

                    WgpuTexture {
                        texture: diffuse_texture,
                        view: diffuse_texture_view,
                        sampler: diffuse_sampler,
                        bind_layout: texture_bind_group_layout,
                        bind_group: diffuse_bind_group,
                    }
                };

                model = Some(Self {
                    vertex_buffer,
                    index_buffer,
                    diffuse_texture: texture,
                    num_indices: primitive.indices.len() as u32
                });
            }
        }

        // TODO: currently only loading one (the last) submesh. load the other submeshes fam.
        model.unwrap()
    }
}

impl Foo {
    async fn new(window: &Window) -> Self {
        let ctx = GraphicsContext::new(window).await;
        let test_model = Model::from_gltf("my_res/suzanne.gltf", &ctx.device, &ctx.queue);
        //let test_model = Model::test_pentagon(&ctx.device);

        let (vs_module, fs_module) = {
            let vs_src = include_str!("gltf.vert");
            let fs_src = include_str!("gltf.frag");
            let mut compiler = shaderc::Compiler::new().unwrap();
            let vs_spirv = compiler.compile_into_spirv(vs_src, shaderc::ShaderKind::Vertex, "gltf.vert", "main", None).unwrap();
            let fs_spirv = compiler.compile_into_spirv(fs_src, shaderc::ShaderKind::Fragment, "gltf.frag", "main", None).unwrap();
            let vs_data = wgpu::read_spirv(std::io::Cursor::new(vs_spirv.as_binary_u8())).unwrap();
            let fs_data = wgpu::read_spirv(std::io::Cursor::new(fs_spirv.as_binary_u8())).unwrap();
            let vs_module = ctx.device.create_shader_module(&vs_data);
            let fs_module = ctx.device.create_shader_module(&fs_data);
            (vs_module, fs_module)
        };

        let render_pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&test_model.diffuse_texture.bind_layout],
        });

        let render_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &render_pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: ctx.sc_desc.format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    GltfVertex::desc(),
                ],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });



        Foo {
            ctx,
            render_pipeline,
            test_model,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let mut ctx = &mut self.ctx;
        ctx.size = new_size;
        ctx.sc_desc.width = new_size.width;
        ctx.sc_desc.height = new_size.height;
        ctx.swap_chain = ctx.device.create_swap_chain(&ctx.surface, &ctx.sc_desc);
    }

    // input() won't deal with GPU code, so it can be synchronous
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        //unimplemented!()
    }

    fn render(&mut self) {
        let mut encoder = self.ctx.device.create_command_encoder(
            &CommandEncoderDescriptor { label: None }
        );

        let frame = self.ctx.swap_chain.get_next_texture()
            .expect("Timeout getting texture");
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }
                    }
                ],
                depth_stencil_attachment: None
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.test_model.diffuse_texture.bind_group, &[]);
            render_pass.set_vertex_buffer(0, &self.test_model.vertex_buffer, 0, 0);
            render_pass.set_index_buffer(&self.test_model.index_buffer, 0, 0);
            render_pass.draw_indexed(0..self.test_model.num_indices, 0, 0..1);
        }
        self.ctx.queue.submit(&[encoder.finish()])
    }
}



fn main() {
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    use futures::executor::block_on;
    //let mut state = block_on(state::State::new(&window));
    let mut state = block_on(Foo::new(&window));
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => {
                                *control_flow = ControlFlow::Exit;
                            }
                            _ => {}
                        },
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                state.render();
            }
            _ => {}
        }
    });
}

