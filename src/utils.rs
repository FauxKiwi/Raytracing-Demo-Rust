use std::ffi::CString;
use glm::{fma, GenMat, Mat4, Mat4x3, tan, Vec3, Vec4};
use num_traits::Zero;

pub fn create_shader(resource: &str, shader_type: gl::types::GLenum, shader_version: Option<&str>) -> u32 {
    let shader: u32 = unsafe { gl::CreateShader(shader_type) };

    let source = std::fs::read_to_string(resource).unwrap();

    let source = match shader_version {
        None => source.to_owned(),
        Some(shader_version) => format!("#version {}\n{}", shader_version, source)
    };

    unsafe {
        let source_length = source.len() as i32;
        let source = CString::new(source).unwrap();
        gl::ShaderSource(shader, 1, &source.as_ptr(), &source_length);

        gl::CompileShader(shader);
        let mut compiled: i32 = -1;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut compiled);

        let mut shader_log_length: i32 = -1;
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut shader_log_length);
        let mut shader_log: Vec<u8> = Vec::with_capacity(shader_log_length as usize + 1);
        shader_log.extend([b' '].iter().cycle().take(shader_log_length as usize));
        let shader_log = CString::from_vec_unchecked(shader_log);
        gl::GetShaderInfoLog(shader, shader_log_length, std::ptr::null_mut(), shader_log.as_ptr() as *mut gl::types::GLchar);
        let shader_log = shader_log.to_string_lossy().into_owned().trim().to_string();

        if shader_log.len() > 0 {
            eprintln!("{}", shader_log);
        }
        if compiled == 0 {
            panic!("Could not compile shader {}", resource);
        }
    }

    shader
}

pub fn mat4x3_set_look_at(mat: &mut Mat4x3, eye: Vec3, center: Vec3, up: Vec3) {
    // Compute direction from position to lookAt
    let dir = eye - center;
    // Normalize direction
    let dir = glm::normalize(dir);
    // left = up x direction
    let left = Vec3::new(
        up.y * dir.z - up.z * dir.y,
        up.z * dir.x - up.x * dir.z,
        up.x * dir.y - up.y * dir.x,
    );
    // normalize left
    let left = glm::normalize(left);
    // up = direction x left
    let upn = Vec3::new(
        dir.y * left.z - dir.z * left.y,
        dir.z * left.x - dir.x * left.z,
        dir.x * left.y - dir.y * left.x,
    );

    mat.c0 = Vec3::new(left.x, upn.x, dir.x);
    mat.c1 = Vec3::new(left.y, upn.y, dir.y);
    mat.c2 = Vec3::new(left.z, upn.z, dir.z);
    mat.c3 = Vec3::new(
        -glm::dot(left, eye),
        -glm::dot(upn, eye),
        -glm::dot(dir, eye),
    );
}

pub fn mat4_set_perspective(mat: &mut Mat4, fov_y: f32, aspect: f32, z_near: f32, z_far: f32) {
    let z_zero_to_one = false;
    mat.set_zero();

    let h = tan(fov_y * 0.5);
    mat.c0.x = 1.0 / (h * aspect);
    mat.c1.y = 1.0 / h;

    let far_inf = z_far > 0.0 && z_far.is_infinite();
    let near_inf = z_near > 0.0 && z_near.is_infinite();

    if far_inf {
        // See: "Infinite Projection Matrix" (http://www.terathon.com/gdc07_lengyel.pdf)
        let e: f32 = 1E-6;
        mat.c2.z = e - 1.0;
        mat.c3.z = (e - (if z_zero_to_one { 1.0 } else { 2.0 }) * z_near);
    } else if near_inf {
        let e: f32 = 1E-6;
        mat.c2.z = (if z_zero_to_one { 0.0 } else { 1.0 }) - e;
        mat.c3.z = ((if z_zero_to_one { 1.0 } else { 2.0 } - e) * z_far);
    } else {
        mat.c2.z = (if z_zero_to_one { z_far } else { z_far + z_near }) / (z_near - z_far);
        mat.c3.z = (if z_zero_to_one { z_far } else { z_far + z_far }) * z_near / (z_near - z_far);
    }

    mat.c2.w = -1.0;
}

pub fn mat4_invert_perspective_view(mat: Mat4, mut view: Mat4x3, dest: &mut Mat4) {
    let a = 1.0 / (mat.c0.x * mat.c1.y);
    let l = -1.0 / (mat.c2.w * mat.c3.z);

    let pm00 = mat.c1.y * a;
    let pm11 = mat.c0.x * a;
    let pm23 = -mat.c2.w * l;
    let pm32 = -mat.c3.z * l;
    let pm33 = mat.c2.z * l;

    let vm = Vec4::new(
        -glm::dot(view.c0, view.c3),
        -glm::dot(view.c1, view.c3),
        -glm::dot(view.c2, view.c3),
        1.0
    );

    view.c3 = Vec3::zero();
    let vt = view.transpose();

    dest.c0 = vt.c0 * pm00;
    dest.c1 = vt.c1 * pm11;
    dest.c2 = vm * pm23;
    dest.c3 = vt.c2 * pm32 + vm * pm33;
}

pub fn vec3_mul_project(vec: &mut Vec3, mat: Mat4) {
    let x = vec.x;
    let y = vec.y;
    let z = vec.z;

    let inv_w = 1.0 / fma(mat.c0.w, x, fma(mat.c1.w, y, fma(mat.c2.w, z, mat.c3.w)));
    let vec0 = fma(
        mat.c0.truncate(3),
        Vec3::new(x, x, x),
        fma(
            mat.c1.truncate(3),
            Vec3::new(y, y, y),
            fma(
                mat.c2.truncate(3),
                Vec3::new(z, z, z),
                mat.c3.truncate(3),
            )
        )
    ) * inv_w;

    vec.x = vec0.x;
    vec.y = vec0.y;
    vec.z = vec0.z;
}