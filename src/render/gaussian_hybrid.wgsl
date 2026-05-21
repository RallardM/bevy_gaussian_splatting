// vendor/bevy_gaussian_splatting/src/render/gaussian_hybrid.wgsl
//
// Per-splat hybrid gaussian renderer.
// Combines 2D surfel and 3D Gaussian rendering in a single draw call.
// No separate entities, no z-fighting, onion-peel layers preserved.
//
// Visibility channel encoding (set in build_planar):
//   floor(visibility) bits:
//     bit 0 (& 1u) : is_square  — 0 = circle / disc, 1 = square
//     bit 1 (& 2u) : is_2d      — 0 = 3D Gaussian,   1 = 2D surfel
//   fract(visibility) = edge fuzziness  (0 = hard edge, 1 = full Gaussian falloff)
//
//   Encoding table:
//     0.f  →  3D Gaussian, circle, fuzz = f
//     1.f  →  3D Gaussian, square, fuzz = f
//     2.f  →  2D Surfel,   circle, fuzz = f
//     3.f  →  2D Surfel,   square, fuzz = f
//
// Register in mod.rs:
//   - Add GaussianMode::GaussianHybrid
//   - push "GAUSSIAN_HYBRID" and "GAUSSIAN_3D_STRUCTURE" shader defs for that variant
//   - load_internal_asset! this file with a new GAUSSIAN_HYBRID_SHADER_HANDLE

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
    Entry,
}
#import bevy_gaussian_splatting::classification::class_to_rgb
#import bevy_gaussian_splatting::depth::depth_to_rgb
#import bevy_gaussian_splatting::optical_flow::{
    calculate_motion_vector,
    optical_flow_to_rgb,
}
#import bevy_gaussian_splatting::helpers::{
    get_rotation_matrix,
    get_scale_matrix,
    cov2d,
    get_bounding_box_clip,
    intrinsic_matrix,
}
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}

// ── Storage backend imports ───────────────────────────────────────────────────
// Always import rotation + scale (needed by 2D surfel path regardless of PRECOMPUTE).
// Additionally import get_cov3d when PRECOMPUTE_COVARIANCE_3D is active.
#ifdef PACKED
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::packed::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::packed::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_STORAGE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::planar::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_TEXTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::texture::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
            get_cov3d,
            location,
        }
    #else
        #import bevy_gaussian_splatting::texture::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
            location,
        }
    #endif
#endif

// ── Sorted entry buffer ───────────────────────────────────────────────────────
#ifdef BUFFER_STORAGE
    @group(3) @binding(0) var<storage, read> sorted_entries: array<Entry>;
    fn get_entry(index: u32) -> Entry {
        return sorted_entries[index];
    }
#else ifdef BUFFER_TEXTURE
    @group(3) @binding(0) var sorted_entries: texture_2d<u32>;
    fn get_entry(index: u32) -> Entry {
        let sample = textureLoad(sorted_entries, location(index), 0);
        return Entry(sample.r, sample.g);
    }
#endif

// ── Per-splat data helpers ────────────────────────────────────────────────────

fn splat_is_square(enc: f32) -> bool {
    return (u32(floor(enc)) & 1u) != 0u;
}

fn splat_is_2d(enc: f32) -> bool {
    return (u32(floor(enc)) & 2u) != 0u;
}

fn splat_fuzziness(enc: f32) -> f32 {
    return fract(enc);
}

// ── 3D Gaussian covariance ────────────────────────────────────────────────────

fn compute_cov3d_hybrid(scale: vec3<f32>, rotation: vec4<f32>) -> array<f32, 6> {
    let S  = get_scale_matrix(scale);
    let T  = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );
    let R  = get_rotation_matrix(rotation);
    let M  = S * R;
    let Sg = transpose(M) * M;
    let TS = T * Sg * transpose(T);
    return array<f32, 6>(
        TS[0][0], TS[0][1], TS[0][2],
        TS[1][1], TS[1][2], TS[2][2],
    );
}

fn compute_cov2d_3dgs_hybrid(position: vec3<f32>, index: u32) -> vec3<f32> {
#ifdef PRECOMPUTE_COVARIANCE_3D
    let cov3d = get_cov3d(index);
#else
    let cov3d = compute_cov3d_hybrid(get_scale(index), get_rotation(index));
#endif
    return cov2d(position, cov3d);
}

// ── 2D Surfel covariance ──────────────────────────────────────────────────────

struct HybridSurfel {
    local_to_pixel: mat3x3<f32>,
    mean_2d:        vec2<f32>,
    extent:         vec2<f32>,
}

fn compute_cov2d_surfel_hybrid(
    gaussian_position: vec3<f32>,
    index:             u32,
    cutoff:            f32,
) -> HybridSurfel {
    var out: HybridSurfel;

    let T_r = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );

    let S = get_scale_matrix(get_scale(index));
    let R = get_rotation_matrix(get_rotation(index));
    let L = T_r * transpose(R) * S;

    let world_from_local = mat3x4<f32>(
        vec4<f32>(L[0], 0.0),
        vec4<f32>(L[1], 0.0),
        vec4<f32>(gaussian_position, 1.0),
    );

    let ndc_from_world  = transpose(view.clip_from_world);
    let pixels_from_ndc = intrinsic_matrix();
    let T               = transpose(world_from_local) * ndc_from_world * pixels_from_ndc;

    let test = vec3<f32>(cutoff * cutoff, cutoff * cutoff, -1.0);
    let d    = dot(test * T[2], T[2]);
    if abs(d) < 1.0e-4 {
        out.extent = vec2<f32>(0.0);
        return out;
    }

    let f      = (1.0 / d) * test;
    let mean2d = vec2<f32>(
        dot(f, T[0] * T[2]),
        dot(f, T[1] * T[2]),
    );
    let t = vec2<f32>(
        dot(f * T[0], T[0]),
        dot(f * T[1], T[1]),
    );

    out.local_to_pixel = T;
    out.mean_2d        = mean2d;
    out.extent         = mean2d * mean2d - t;
    return out;
}

fn get_bounding_box_surfel_hybrid(
    extent:    vec2<f32>,
    direction: vec2<f32>,
    cutoff:    f32,
) -> vec4<f32> {
    let filter_size = 0.707106;
    if extent.x < 1.e-4 || extent.y < 1.e-4 {
        return vec4<f32>(0.0);
    }
    let radius     = sqrt(extent);
    let max_radius = vec2<f32>(max(max(radius.x, radius.y), cutoff * filter_size));
    let radius_ndc = max_radius / view.viewport.zw;
    return vec4<f32>(radius_ndc * direction, max_radius);
}

fn surfel_fragment_power_hybrid(
    local_to_pixel: mat3x3<f32>,
    pixel_coord:    vec2<f32>,
    mean_2d:        vec2<f32>,
) -> f32 {
    let deltas = mean_2d - pixel_coord;
    let hu     = pixel_coord.x * local_to_pixel[2] - local_to_pixel[0];
    let hv     = pixel_coord.y * local_to_pixel[2] - local_to_pixel[1];
    let p      = cross(hu, hv);
    let us     = p.x / p.z;
    let vs     = p.y / p.z;
    let s3d    = us * us + vs * vs;
    let s2d    = 2.0 * (deltas.x * deltas.x + deltas.y * deltas.y);
    return -0.5 * min(s3d, s2d);
}

// ── View-space utilities ──────────────────────────────────────────────────────

fn world_to_local_direction(ray_dir_world: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    let basis   = mat3x3<f32>(
        transform[0].xyz,
        transform[1].xyz,
        transform[2].xyz,
    );
    return normalize(vec3<f32>(
        dot(normalize(basis[0]), ray_dir_world),
        dot(normalize(basis[1]), ray_dir_world),
        dot(normalize(basis[2]), ray_dir_world),
    ));
}

// ── Vertex output struct ──────────────────────────────────────────────────────
// Carries fields for both 3D and 2D paths.
// The fragment shader reads only the relevant subset based on per_splat_data.

#ifdef WEBGL2
struct GaussianVertexOutput {
    @builtin(position) position:         vec4<f32>,
    @location(0)       color:            vec4<f32>,
    @location(1)       uv:               vec2<f32>,
    // Encoded: floor = {is_square bit, is_2d bit}, frac = fuzziness
    @location(2)       per_splat_data:   f32,
    // 3D Gaussian fields (populated when is_2d == 0)
    @location(3)       conic:            vec3<f32>,
    @location(4)       major_minor:      vec2<f32>,
    // 2D Surfel fields (populated when is_2d == 1)
    @location(5)       local_to_pixel_u: vec3<f32>,
    @location(6)       local_to_pixel_v: vec3<f32>,
    @location(7)       local_to_pixel_w: vec3<f32>,
    @location(8)       mean_2d:          vec2<f32>,
    @location(9)       radius:           vec2<f32>,
};
#else
struct GaussianVertexOutput {
    @builtin(position)                   position:         vec4<f32>,
    @location(0) @interpolate(flat)      color:            vec4<f32>,
    @location(1) @interpolate(linear)    uv:               vec2<f32>,
    @location(2) @interpolate(flat)      per_splat_data:   f32,
    // 3D Gaussian fields
    @location(3) @interpolate(flat)      conic:            vec3<f32>,
    @location(4) @interpolate(linear)    major_minor:      vec2<f32>,
    // 2D Surfel fields
    @location(5) @interpolate(flat)      local_to_pixel_u: vec3<f32>,
    @location(6) @interpolate(flat)      local_to_pixel_v: vec3<f32>,
    @location(7) @interpolate(flat)      local_to_pixel_w: vec3<f32>,
    @location(8) @interpolate(flat)      mean_2d:          vec2<f32>,
    @location(9) @interpolate(flat)      radius:           vec2<f32>,
};
#endif

// ── Vertex shader ─────────────────────────────────────────────────────────────

@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index)   vertex_index:   u32,
) -> GaussianVertexOutput {
    var output: GaussianVertexOutput;

    let entry       = get_entry(instance_index);
    let splat_index = entry.value;

    var discard_quad = false;
    discard_quad |= entry.key == 0xFFFFFFFFu;

    let position             = vec4<f32>(get_position(splat_index), 1.0);
    var transformed_position = (gaussian_uniforms.transform * position).xyz;
    let prev_transformed     = transformed_position; // optical flow baseline

    // Decode per-splat data from the visibility channel.
    let enc  = get_visibility(splat_index);
    let is2d = splat_is_2d(enc);
    output.per_splat_data = enc;

    let projected_position = world_to_clip(transformed_position);
    discard_quad |= !in_frustum(projected_position.xyz);

    if discard_quad {
        output.color    = vec4<f32>(0.0);
        output.position = vec4<f32>(0.0);
        return output;
    }

    let quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let quad_offset = quad_vertices[vertex_index % 4u];

    var opacity = get_opacity(splat_index);

#ifdef OPACITY_ADAPTIVE_RADIUS
    let cutoff = sqrt(max(9.0 + 2.0 * log(opacity), 0.000001));
#else
    let cutoff = 3.0;
#endif

    var bb: vec4<f32>;

    if is2d {
        // ── 2D Surfel path ────────────────────────────────────────────────────
        let surfel = compute_cov2d_surfel_hybrid(transformed_position, splat_index, cutoff);

        output.local_to_pixel_u = surfel.local_to_pixel[0];
        output.local_to_pixel_v = surfel.local_to_pixel[1];
        output.local_to_pixel_w = surfel.local_to_pixel[2];
        output.mean_2d          = surfel.mean_2d;

        bb             = get_bounding_box_surfel_hybrid(surfel.extent, quad_offset, cutoff);
        output.radius  = bb.zw;

    } else {
        // ── 3D Gaussian path ──────────────────────────────────────────────────
        let gaussian_cov2d = compute_cov2d_3dgs_hybrid(transformed_position, splat_index);
        bb = get_bounding_box_clip(gaussian_cov2d, quad_offset, cutoff);

#ifdef USE_AABB
        let det     = gaussian_cov2d.x * gaussian_cov2d.z
                    - gaussian_cov2d.y * gaussian_cov2d.y;
        let det_inv = 1.0 / det;
        output.conic = vec3<f32>(
             gaussian_cov2d.z * det_inv,
            -gaussian_cov2d.y * det_inv,
             gaussian_cov2d.x * det_inv,
        );
        output.major_minor = bb.zw;
#endif
    }

    // ── Color ─────────────────────────────────────────────────────────────────
    var rgb = vec3<f32>(0.0);

#ifdef RASTERIZE_CLASSIFICATION
    // Note: visibility channel is repurposed for per-splat encoding in hybrid mode.
    // class_to_rgb receives the encoded value, not a class index.
    // For correct classification rendering, a separate class channel would be needed.
    let ray_dir_world = normalize(transformed_position - view.world_position);
    let ray_dir_local = world_to_local_direction(ray_dir_world, gaussian_uniforms.transform);
    #ifdef GAUSSIAN_3D_STRUCTURE
        rgb = get_color(splat_index, ray_dir_local);
    #endif
    rgb = class_to_rgb(enc, rgb);

#else ifdef RASTERIZE_DEPTH
    let first_pos = vec4<f32>(get_position(get_entry(1u).value), 1.0);
    let last_pos  = vec4<f32>(get_position(get_entry(gaussian_uniforms.count - 1u).value), 1.0);
    let min_pos   = (gaussian_uniforms.transform * last_pos).xyz;
    let max_pos   = (gaussian_uniforms.transform * first_pos).xyz;
    let cam_pos   = view.world_position;
    rgb = depth_to_rgb(
        length(transformed_position - cam_pos),
        length(min_pos - cam_pos),
        length(max_pos - cam_pos),
    );

#else ifdef RASTERIZE_NORMAL
    let R_mat   = get_rotation_matrix(get_rotation(splat_index));
    let S_mat   = get_scale_matrix(get_scale(splat_index));
    let T_mat   = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );
    let L            = T_mat * S_mat * R_mat;
    let local_normal = vec4<f32>(L[2], 0.0);
    let world_normal = view.view_from_world * local_normal;
    let t            = normalize(world_normal);
    rgb = vec3<f32>(0.5 * (t.x + 1.0), 0.5 * (t.y + 1.0), 0.5 * (t.z + 1.0));

#else ifdef RASTERIZE_OPTICAL_FLOW
    rgb = optical_flow_to_rgb(calculate_motion_vector(transformed_position, prev_transformed));

#else ifdef RASTERIZE_POSITION
    rgb = (transformed_position - gaussian_uniforms.min.xyz)
        / (gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz);

#else ifdef RASTERIZE_COLOR
    let ray_dir_world = normalize(transformed_position - view.world_position);
    let ray_dir_local = world_to_local_direction(ray_dir_world, gaussian_uniforms.transform);
    #ifdef GAUSSIAN_3D_STRUCTURE
        rgb = get_color(splat_index, ray_dir_local);
    #endif
#endif

    output.color    = vec4<f32>(rgb, opacity * gaussian_uniforms.global_opacity);
    output.uv       = quad_offset;
    output.position = vec4<f32>(projected_position.xy + bb.xy, projected_position.zw);

    return output;
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(input: GaussianVertexOutput) -> @location(0) vec4<f32> {

    // Decode per-splat mode, shape and fuzziness.
    let is_square = splat_is_square(input.per_splat_data);
    let is_2d     = splat_is_2d(input.per_splat_data);
    let fuzz      = splat_fuzziness(input.per_splat_data);

    var power: f32 = 0.0;

    // ── Gaussian contribution (power) ─────────────────────────────────────────
#ifdef USE_AABB
    if is_2d {
        // 2D surfel: project pixel into surfel local frame.
        let aspect      = vec2<f32>(1.0, view.viewport.z / view.viewport.w);
        let pixel_coord = input.uv * input.radius * aspect + input.mean_2d;
        power = surfel_fragment_power_hybrid(
            mat3x3<f32>(input.local_to_pixel_u, input.local_to_pixel_v, input.local_to_pixel_w),
            pixel_coord,
            input.mean_2d,
        );
    } else {
        // 3D Gaussian: conic quadratic form.
        let d     = -input.major_minor;
        let conic = input.conic;
        power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
    }
    if power > 0.0 { discard; }
#endif

#ifdef USE_OBB
    let sigma_sq   = 2.0 * (1.0 / 3.0) * (1.0 / 3.0);
    let dist_sq    = dot(input.uv, input.uv);
    power = -dist_sq / sigma_sq;
    // Circle clip: skip for square splats.
    if !is_square && dist_sq > 9.0 { discard; }
#endif

#ifdef VISUALIZE_BOUNDING_BOX
    let uv_n   = input.uv * 0.5 + 0.5;
    let edge_w = 0.08;
    if uv_n.x < edge_w || uv_n.x > 1.0 - edge_w ||
       uv_n.y < edge_w || uv_n.y > 1.0 - edge_w {
        return vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

    // ── Alpha — per-splat shape and fuzziness ─────────────────────────────────
    // No 0.999 cap: allows fully opaque splats so black colours reach true black.
    // fuzz = 0 → hard edge (step function for circle, flat fill for square).
    // fuzz = 1 → soft edge (Gaussian falloff for circle, edge_dist fade for square).
    var alpha: f32;
    if is_square {
        let edge_dist = min(1.0 - abs(input.uv.x), 1.0 - abs(input.uv.y));
        let fill      = mix(1.0, saturate(edge_dist), fuzz);
        alpha = mix(input.color.a, fill * input.color.a, fuzz);
    } else {
        let r_sq      = dot(input.uv, input.uv);
        let hard_disc = select(0.0, 1.0, r_sq < 1.0);
        let fill      = mix(hard_disc, exp(power), fuzz);
        alpha = fill * input.color.a;
    }

    return vec4<f32>(input.color.rgb * alpha, alpha);
}