mod utils;

use std::cmp::{max, min};
use std::ffi::CString;
use std::sync::mpsc::Receiver;
use std::time::{Duration, SystemTime};
use glfw::Context;
use glm::{ceil, cos, IVec2, Mat4, Mat4x3, radians, sin, UVec2, Vec3};
use num_traits::identities::Zero;

struct State {
    width: i32,
    height: i32,
    bounce_count: u16,
    frame_number: u32,
    reset_framebuffer: bool,
    mouse_x: f64,
    mouse_down_x: f64,
    mouse_down: bool,
    current_y_rot: f64,
    rotation_about_y: f64,
    first_time: SystemTime,
    framebuffer_image_binding: i32,
    tex: u32,
    work_group_size: IVec2,
    vao: u32,
    sampler: u32,
    last_time: SystemTime,
}

fn init() -> (glfw::Glfw, glfw::Window, Receiver<(f64, glfw::WindowEvent)>, u32, u32, Uniforms, State) {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 768;

    let mut glfw = glfw::init(glfw::LOG_ERRORS).expect("Unable to initialize GLFW");

    glfw.default_window_hints();
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 3));
    glfw.window_hint(glfw::WindowHint::Visible(false));
    glfw.window_hint(glfw::WindowHint::Resizable(true));

    let (mut window, events) = glfw.create_window(WIDTH, HEIGHT, "Raytracing Demo (compute shader)", glfw::WindowMode::Windowed)
        .expect("Failed to create the GLFW window");

    glfw.with_primary_monitor(|_, monitor| {
        match monitor {
            Some(monitor) => {
                let vid_mode = monitor.get_video_mode();
                match vid_mode {
                    Some(vid_mode) => { window.set_pos(((vid_mode.width - WIDTH) / 2) as i32, ((vid_mode.height - HEIGHT) / 2) as i32); }
                    _ => {}
                }
            }
            _ => {}
        }
    });

    window.make_current();
    window.set_all_polling(true);
    glfw.set_swap_interval(glfw::SwapInterval::None);

    let (width, height) = window.get_framebuffer_size();

    gl::load_with(|s| glfw.get_proc_address_raw(s));
    gl::Viewport::load_with(|s| glfw.get_proc_address_raw(s));

    let tex = create_framebuffer_texture(width, height);
    let sampler = create_sampler();
    let vao = {
        let mut vao: u32 = u32::MAX;
        unsafe { gl::GenVertexArrays(1, &mut vao); }
        vao
    };
    let compute_program = create_compute_program();
    let (framebuffer_image_binding, work_group_size, uniforms) = init_compute_program(compute_program);
    let quad_program = create_quad_program();
    init_quad_program(quad_program);

    window.show();

    (glfw, window, events, quad_program, compute_program, uniforms, State {
        width, height,
        bounce_count: 5,
        frame_number: 0,
        reset_framebuffer: true,
        mouse_x: 0.0,
        mouse_down_x: 0.0,
        mouse_down: false,
        current_y_rot: 0.0,
        rotation_about_y: 0.8,
        first_time: SystemTime::now(),
        framebuffer_image_binding,
        tex,
        work_group_size,
        vao,
        sampler,
        last_time: SystemTime::now(),
    })
}

fn create_quad_program() -> u32 {
    let program: u32 = unsafe { gl::CreateProgram() };

    let vshader = utils::create_shader("./quad.vs", gl::VERTEX_SHADER, Some("330"));
    let fshader = utils::create_shader("./quad.fs", gl::FRAGMENT_SHADER, Some("330"));

    unsafe {
        gl::AttachShader(program, vshader);
        gl::AttachShader(program, fshader);

        gl::BindFragDataLocation(program, 0, CString::new("color").unwrap().into_raw());

        gl::LinkProgram(program);
        let mut linked: i32 = -1;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut linked);

        let mut program_log_length: i32 = -1;
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut program_log_length);
        let mut program_log: Vec<u8> = Vec::with_capacity(program_log_length as usize + 1);
        program_log.extend([b' '].iter().cycle().take(program_log_length as usize));
        let program_log = CString::from_vec_unchecked(program_log);
        gl::GetProgramInfoLog(program, program_log_length, std::ptr::null_mut(), program_log.as_ptr() as *mut gl::types::GLchar);
        let program_log = program_log.to_string_lossy().into_owned().trim().to_string();

        if program_log.len() > 0 {
            eprintln!("{}", program_log);
        }
        if linked == 0 {
            panic!("Could not link program");
        }
    }
    program
}

fn create_compute_program() -> u32 {
    let program: u32 = unsafe { gl::CreateProgram() };

    let cshader = utils::create_shader("./raytracing.glsl", gl::COMPUTE_SHADER, None);
    let random = utils::create_shader("./random.glsl", gl::COMPUTE_SHADER, None);
    let random_common =  utils::create_shader("./randomCommon.glsl", gl::COMPUTE_SHADER, Some("330"));

    unsafe {
        gl::AttachShader(program, cshader);
        gl::AttachShader(program, random);
        gl::AttachShader(program, random_common);

        gl::LinkProgram(program);
        let mut linked: i32 = -1;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut linked);

        let mut program_log_length: i32 = -1;
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut program_log_length);
        let mut program_log: Vec<u8> = Vec::with_capacity(program_log_length as usize + 1);
        program_log.extend([b' '].iter().cycle().take(program_log_length as usize));
        let program_log = CString::from_vec_unchecked(program_log);
        gl::GetProgramInfoLog(program, program_log_length, std::ptr::null_mut(), program_log.as_ptr() as *mut gl::types::GLchar);
        let program_log = program_log.to_string_lossy().into_owned().trim().to_string();

        if program_log.len() > 0 {
            eprintln!("{}", program_log);
        }
        if linked == 0 {
            panic!("Could not link program");
        }
    }

    program
}

fn init_quad_program(program: u32) {
    unsafe {
        gl::UseProgram(program);
        let tex_uniform = gl::GetUniformLocation(program, CString::new("tex").unwrap().into_raw());
        gl::Uniform1i(tex_uniform, 0);
        gl::UseProgram(0);
    }
}

struct Uniforms {
    eye_uniform: i32,
    ray00_uniform: i32,
    ray01_uniform: i32,
    ray10_uniform: i32,
    ray11_uniform: i32,
    time_uniform: i32,
    blend_factor_uniform: i32,
    bounce_count_uniform: i32,
}

fn init_compute_program(program: u32) -> (i32, IVec2, Uniforms) {
    unsafe {
        gl::UseProgram(program);
        let mut work_group_size: [i32; 3] = [0; 3];
        gl::GetProgramiv(program, gl::COMPUTE_WORK_GROUP_SIZE, work_group_size.as_mut_ptr());
        //println!("{:?}", work_group_size);

        let eye_uniform = gl::GetUniformLocation(program, CString::new("eye").unwrap().into_raw());
        let ray00_uniform = gl::GetUniformLocation(program, CString::new("ray00").unwrap().into_raw());
        let ray01_uniform = gl::GetUniformLocation(program, CString::new("ray01").unwrap().into_raw());
        let ray10_uniform = gl::GetUniformLocation(program, CString::new("ray10").unwrap().into_raw());
        let ray11_uniform = gl::GetUniformLocation(program, CString::new("ray11").unwrap().into_raw());
        let time_uniform = gl::GetUniformLocation(program, CString::new("time").unwrap().into_raw());
        let blend_factor_uniform = gl::GetUniformLocation(program, CString::new("blendFactor").unwrap().into_raw());
        let bounce_count_uniform = gl::GetUniformLocation(program, CString::new("bounceCount").unwrap().into_raw());
        //println!("{} {} {} {} {} {} {} {}", eye_uniform, ray00_uniform, ray01_uniform, ray10_uniform, ray11_uniform, time_uniform, blend_factor_uniform, bounce_count_uniform);

        let mut framebuffer_image_binding: i32 = -1;
        let loc = gl::GetUniformLocation(program, CString::new("framebufferImage").unwrap().into_raw());
        gl::GetUniformiv(program, loc, &mut framebuffer_image_binding);
        //println!("{}", framebuffer_image_binding);

        gl::UseProgram(0);

        print_info();

        (framebuffer_image_binding, IVec2::new(work_group_size[0], work_group_size[1]), Uniforms {
            eye_uniform,
            ray00_uniform,
            ray01_uniform,
            ray10_uniform,
            ray11_uniform,
            time_uniform,
            blend_factor_uniform,
            bounce_count_uniform,
        })
    }
    /*debugProc = GLUtil.setupDebugMessageCallback();*/
}

fn create_framebuffer_texture(width: i32, height: i32) -> u32 {
    let mut tex: u32 = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA32F, width, height);
        gl::BindTexture(gl::TEXTURE_2D, 0);
    }
    tex
}

fn create_sampler() -> u32 {
    let mut sampler: u32 = 0;
    unsafe {
        gl::GenSamplers(1, &mut sampler);
        gl::SamplerParameteri(sampler, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::SamplerParameteri(sampler, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
    }
    sampler
}

/**
 * Recreate the framebuffer when the window size changes.
 */
fn resize_framebuffer_texture(tex: u32, width: i32, height: i32) {
    unsafe { gl::DeleteTextures(1, &tex); }
    create_framebuffer_texture(width, height);
}

#[derive(Debug)]
struct Matrices {
    proj: Mat4,
    view: Mat4x3,
    inv_view_proj: Mat4
}

#[derive(Debug, Clone, Copy)]
struct Camera {
    position: Vec3,
    look_at: Vec3,
    up: Vec3,
}

fn trace(compute_program: u32, camera: &mut Camera, matrices: &mut Matrices, uniforms: &Uniforms, state: &mut State) {
    unsafe { gl::UseProgram(compute_program); };

    if state.mouse_down {
        /*
         * If mouse is down, compute the camera rotation based on mouse
         * cursor location.
         */
        state.current_y_rot = state.rotation_about_y + (state.mouse_x - state.mouse_down_x) * 0.01;
    } else {
        state.current_y_rot = state.rotation_about_y;
    }

    /* Rotate camera about Y axis. */
    camera.position.x = (sin(-state.current_y_rot) * 3.0) as f32;
    camera.position.z = (cos(-state.current_y_rot) * 3.0) as f32;
    utils::mat4x3_set_look_at(&mut matrices.view, camera.position, camera.look_at, camera.up);

    if state.reset_framebuffer {
        utils::mat4_set_perspective(&mut matrices.proj, radians(60.0), state.width as f32 / state.height as f32, 1.0, 2.0);
        resize_framebuffer_texture(state.tex, state.width, state.height);
        state.reset_framebuffer = false;
    }
    utils::mat4_invert_perspective_view(matrices.proj, matrices.view, &mut matrices.inv_view_proj);

    let this_time = SystemTime::now();
    let elapsed_time = this_time.duration_since(state.first_time).unwrap();
    unsafe { gl::Uniform1f(uniforms.time_uniform, elapsed_time.as_secs_f32()); }

    let frame_time = this_time.duration_since(state.last_time).unwrap();
    state.last_time = this_time;
    let fps = (1 as f64 / (frame_time.as_secs_f64())) as u32;
    print!("\r\x1b[1;32mFPS: {}\x1b[0m \x08", fps);

    /*
     * We are going to average multiple successive frames, so here we
     * compute the blend factor between old frame and new frame. 0.0 - use
     * only the new frame > 0.0 - blend between old frame and new frame
     */
    let blend_factor = state.frame_number as f32 / (state.frame_number + 1) as f32;
    unsafe {
        gl::Uniform1f(uniforms.blend_factor_uniform, blend_factor);
        gl::Uniform1i(uniforms.bounce_count_uniform, state.bounce_count as i32);
    }

    /* Set viewing frustum corner rays in shader */
    let mut temp_vec = Vec3::zero();
    unsafe {
        gl::Uniform3f(uniforms.eye_uniform, camera.position.x, camera.position.y, camera.position.z);

        temp_vec = Vec3::new(-1.0, -1.0, 0.0);
        utils::vec3_mul_project(&mut temp_vec, matrices.inv_view_proj);
        temp_vec = temp_vec - camera.position;
        gl::Uniform3f(uniforms.ray00_uniform, temp_vec.x, temp_vec.y, temp_vec.z);

        temp_vec = Vec3::new(-1.0, 1.0, 0.0);
        utils::vec3_mul_project(&mut temp_vec, matrices.inv_view_proj);
        temp_vec = temp_vec - camera.position;
        gl::Uniform3f(uniforms.ray01_uniform, temp_vec.x, temp_vec.y, temp_vec.z);

        temp_vec = Vec3::new(1.0, -1.0, 0.0);
        utils::vec3_mul_project(&mut temp_vec, matrices.inv_view_proj);
        temp_vec = temp_vec - camera.position;
        gl::Uniform3f(uniforms.ray10_uniform, temp_vec.x, temp_vec.y, temp_vec.z);

        temp_vec = Vec3::new(1.0, 1.0, 0.0);
        utils::vec3_mul_project(&mut temp_vec, matrices.inv_view_proj);
        temp_vec = temp_vec - camera.position;
        gl::Uniform3f(uniforms.ray11_uniform, temp_vec.x, temp_vec.y, temp_vec.z);
    }

    /* Bind level 0 of framebuffer texture as writable image in the shader. */
    unsafe { gl::BindImageTexture(state.framebuffer_image_binding as u32, state.tex, 0, gl::FALSE, 0, gl::READ_WRITE, gl::RGBA32F); }

    /*
     * Compute appropriate global work size dimensions.
     */
    let num_groups = UVec2::new(
        ceil(state.width as f64 / state.work_group_size.x as f64) as u32,
        ceil(state.height as f64 / state.work_group_size.y as f64) as u32,
    );

    /* Invoke the compute shader. */
    unsafe { gl::DispatchCompute(num_groups.x, num_groups.y, 1); }
    /*
     * Synchronize all writes to the framebuffer image before we let OpenGL
     * source texels from it afterwards when rendering the final image with
     * the full-screen quad.
     */
    unsafe { gl::MemoryBarrier(gl::SHADER_IMAGE_ACCESS_BARRIER_BIT); }

    /* Reset bindings. */
    unsafe {
        gl::BindImageTexture(state.framebuffer_image_binding as u32, 0, 0, gl::FALSE, 0, gl::READ_WRITE, gl::RGBA32F);
        gl::UseProgram(0);
    }

    state.frame_number += 1;
}

/**
 * Present the final image on the screen/viewport.
 */
fn present(quad_program: u32, state: &State) {
    /*
     * Draw the rendered image on the screen using textured full-screen
     * quad.
     */
    unsafe {
        gl::UseProgram(quad_program);
        gl::BindVertexArray(state.vao);
        gl::BindTexture(gl::TEXTURE_2D, state.tex);
        gl::BindSampler(0, state.sampler);
        gl::DrawArrays(gl::TRIANGLES, 0, 3);
        gl::BindSampler(0, 0);
        gl::BindTexture(gl::TEXTURE_2D, 0);
        gl::BindVertexArray(0);
        gl::UseProgram(0);
    }
}

fn run_loop(mut glfw: glfw::Glfw, mut window: glfw::Window, events: Receiver<(f64, glfw::WindowEvent)>, quad_program: u32, compute_program: u32, uniforms: Uniforms, mut state: State) {
    let mut camera = Camera {
        position: Vec3::new(0.0, 2.0, 0.0),
        look_at: Vec3::new(0.0, 0.5, 0.0),
        up: Vec3::new(0.0, 1.0, 0.0),
    };
    let mut matrices = Matrices {
        proj: Mat4::zero(),
        view: Mat4x3::zero(),
        inv_view_proj: Mat4::zero(),
    };

    while !window.should_close() {
        glfw.poll_events();
        unsafe { gl::Viewport(0, 0, state.width, state.height); }

        trace(compute_program, &mut camera, &mut matrices, &uniforms, &mut state);
        present(quad_program, &state);

        window.swap_buffers();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(key, _, action, _) => {
                    if action == glfw::Action::Release { break; }

                    match key {
                        glfw::Key::Escape => {
                            window.set_should_close(true);
                        }
                        glfw::Key::KpAdd | glfw::Key::PageUp => {
                            let new_bounce_count = min(100, state.bounce_count + 1);
                            if new_bounce_count != state.bounce_count {
                                state.bounce_count = new_bounce_count;
                                println!("\rRay bounce count is now: {}", new_bounce_count);
                                state.frame_number = 0;
                            }
                        }
                        glfw::Key::KpSubtract | glfw::Key::PageDown => {
                            let new_bounce_count = max(1, state.bounce_count - 1);
                            if new_bounce_count != state.bounce_count {
                                state.bounce_count = new_bounce_count;
                                println!("\rRay bounce count is now: {}", new_bounce_count);
                                state.frame_number = 0;
                            }
                        }
                        _ => {}
                    }
                }
                glfw::WindowEvent::FramebufferSize(width, height) => {
                    if width > 0 && height > 0 && (state.width != width || state.height != height) {
                        state.width = width;
                        state.height = height;
                        state.reset_framebuffer = true;
                        state.frame_number = 0;
                    }
                }
                glfw::WindowEvent::CursorPos(x, _y) => {
                    state.mouse_x = x;
                    if state.mouse_down {
                        state.frame_number = 0;
                    }
                }
                glfw::WindowEvent::MouseButton(_, action, _) => {
                    if action == glfw::Action::Press {
                        state.mouse_down_x = state.mouse_x;
                        state.mouse_down = true;
                    } else if action == glfw::Action::Release {
                        state.mouse_down = false;
                        state.rotation_about_y = state.current_y_rot;
                    }
                }
                _ => {}
            }
        }
    }
}

fn print_info() {
    println!("{}", unsafe { glfw::string_from_c_str(gl::GetString(gl::VENDOR) as *const i8) });
    println!("{}", unsafe { glfw::string_from_c_str(gl::GetString(gl::RENDERER) as *const i8) });
    println!("{}", unsafe { glfw::string_from_c_str(gl::GetString(gl::VERSION) as *const i8) });
    //println!("{}", unsafe { glfw::string_from_c_str(gl::GetString(gl::SHADING_LANGUAGE_VERSION) as *const i8) });
    println!();

    println!("Press keypad '+' or 'page up' to increase the number of bounces.");
    println!("Press keypad '-' or 'page down' to decrease the number of bounces.");

    println!();
    print!("FPS: -");
}

fn main() {
    let (glfw, window, events, quad_program, compute_program, uniforms, state) = init();
    run_loop(glfw, window, events,  quad_program, compute_program, uniforms, state);
}
