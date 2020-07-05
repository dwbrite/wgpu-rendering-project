#version 450

layout(location=0) in vec3 v_color;
layout(location=1) in vec2 v_tex_coords;

layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

void main() {
     vec4 tex_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
     float alpha = tex_color.a;
     vec4 vert_color = vec4(v_color, 1.0);
     f_color = mix(vert_color, tex_color, clamp(alpha/2, 0.0, 0.5));
}