import moderngl as mgl
import numpy as np
from pyrr import Matrix44
from PIL import Image
import os
import time

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
FRAMES = 1800

context = mgl.create_standalone_context()
context.line_width = 2.0

particle_program = context.program(
    vertex_shader='''
        #version 330

        uniform mat4 projection;
        uniform mat4 view;
        uniform vec3 light;
        uniform vec3 camera;

        in vec3 in_position;
        in vec3 in_velocity;
        in vec3 in_color;

        out vec3 v_position;
        out vec3 v_velocity;
        out vec3 v_color;
        out vec3 v_light;
        out vec3 v_camera;

        vec3 forward_substitution(mat3 A, vec3 b) {
            vec3 y = vec3(0.0);

            y[0] = b[0];
            y[1] = b[1] - A[0][1]*y[0];
            y[2] = b[2] - A[0][2]*y[0] - A[1][2]*y[1];

            return y;
        }

        vec3 backward_substitution(mat3 A, vec3 b) {
            vec3 y = vec3(0.0);

            y[2] = b[2] / A[2][2];
            y[1] = (b[1] - A[2][1]*y[2]) / A[1][1];
            y[0] = (b[0] - A[1][0]*y[1] - A[2][0]*y[2]) / A[0][0];

            return y;
        }

        vec3 lu_solver(mat3 A, vec3 b) {
            mat3 L = mat3(1.0);
            mat3 U = mat3(0.0);

            U[0][0] = A[0][0];
            U[1][0] = A[1][0];
            U[2][0] = A[2][0];

            L[0][1] = A[0][1]/U[0][0];
            U[1][1] = A[1][1] - L[0][1]*U[1][0];
            U[2][1] = A[2][1] - L[0][1]*U[2][0];

            L[0][2] = A[0][2]/U[0][0];
            L[1][2] = (A[1][2] - L[0][2]*U[1][0])/U[1][1];
            U[2][2] = A[2][2] - L[1][2]*U[2][1]-L[0][2]*U[2][0];

            vec3 y = forward_substitution(L, b);
            vec3 x = backward_substitution(U, y);
            
            return x;
        }

        void simulation_step(out vec3 result[2], vec3 angle, vec3 velocity) {
            float g = 9.81;
            float l = 1;

            vec3 cosine = vec3(cos(angle[0] - angle[1]),
                cos(angle[0] - angle[2]),
                cos(angle[1] - angle[2]));

            vec3 sine = vec3(sin(angle[0] - angle[1]),
                sin(angle[0] - angle[2]),
                sin(angle[1] - angle[2]));

            mat3 A = mat3((7.0/3.0)*l, (3.0/2.0)*l*cosine[0], (1.0/2.0)*l*cosine[1],
                (3.0/2.0)*l*cosine[0], (4.0/3.0)*l, (1.0/2.0)*l*cosine[2],
                (1.0/2.0)*l*cosine[1], (1.0/2.0)*l*cosine[2], (1.0/3.0)*l);

            vec3 b = vec3((-3.0/2.0)*l*velocity[1]*velocity[1]*sine[0] - (1.0/2.0)*l*velocity[2]*velocity[2]*sine[1] - (5.0/2.0)*g*sin(angle[0]),
                (3.0/2.0)*l*velocity[0]*velocity[0]*sine[0] - (1.0/2.0)*l*velocity[2]*velocity[2]*sine[2] - (3.0/2.0)*g*sin(angle[1]),
                (1.0/2.0)*l*velocity[0]*velocity[0]*sine[1] + (1.0/2.0)*l*velocity[1]*velocity[1]*sine[2] - (1.0/2.0)*g*sin(angle[2]));

            vec3 acceleration = lu_solver(A, b);

            result[0] = velocity;
            result[1] = acceleration;
        }

        void runge_kutta(out vec3 result[2], vec3 angle, vec3 velocity, float dt) {
            vec3 k1[2];
            vec3 k2[2];
            vec3 k3[2];
            vec3 k4[2];

            simulation_step(k1, angle, velocity);
            simulation_step(k2, angle + dt*(k1[0]/2), velocity + dt*(k1[1]/2));
            simulation_step(k3, angle + dt*(k2[0]/2), velocity + dt*(k2[1]/2));
            simulation_step(k4, angle + dt*k3[0], velocity + dt*k3[1]);

            result[0] = angle + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
            result[1] = velocity + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
        }

        void main() {
            vec4 world_position = view * vec4(in_position, 1.0);
            gl_Position = projection * world_position;

            float distance_from_camera = abs(length(vec4(camera, 1.0) - world_position));
            gl_PointSize = 400.0 / distance_from_camera;

            float slowdown = 5.0;
            float dt = 1.0 / (30.0 * slowdown);
            vec3 data[2];

            runge_kutta(data, in_position/3, in_velocity, dt);


            v_position = data[0]*3;
            v_velocity = data[1];
            v_color = in_color;
            v_light = light;
            v_camera = camera;
        }
        ''',
        fragment_shader='''
        #version 330

        in vec3 v_position;
        in vec3 v_velocity;
        in vec3 v_color;
        in vec3 v_light;
        in vec3 v_camera;

        out vec4 f_color;

        void main() {
            float distance_from_center = length(gl_PointCoord - vec2(0.5, 0.5));
            if (distance_from_center > 0.5) discard;

            vec2 position_inside_point = gl_PointCoord * 2.0 - 1.0;
            position_inside_point.y *= -1;
            
            float z = sqrt(max(0.0, 1.0 - position_inside_point.x * position_inside_point.x - position_inside_point.y * position_inside_point.y));
            vec3 position_on_sphere = vec3(position_inside_point, z);

            vec3 center_of_sphere = normalize(v_camera - v_position);

            vec3 axis_of_rotation = normalize(cross(vec3(0.0, 0.0, 1.0), center_of_sphere));
            float theta = acos(dot(vec3(0.0, 0.0, 1.0), center_of_sphere));
            vec3 adjusted_position_on_sphere = position_on_sphere*cos(theta)
                + cross(axis_of_rotation, position_on_sphere)*sin(theta)
                + axis_of_rotation*(dot(axis_of_rotation, position_on_sphere))*(1-cos(theta));

            vec3 direction_of_light = v_light - (v_position + adjusted_position_on_sphere);
            vec3 normalized_direction_of_light = normalize(direction_of_light);

            float intensity = min(1.0, max(0.2, 1-(length(direction_of_light)/50)));
            float diffusivity = max(dot(normalized_direction_of_light, adjusted_position_on_sphere), 0.0) * intensity;
            float antialiasing = smoothstep(0.5, 0.48, distance_from_center);

            f_color = vec4((0.1 +0.9*diffusivity) * v_color, 1.0 * antialiasing);
        }
        ''',
        varyings=['v_position', 'v_velocity', 'v_color'],
)

axis_program = context.program(
    vertex_shader = '''
    #version 330

    uniform mat4 projection;
    uniform mat4 view;

    in vec3 in_position;
    void main() {
        gl_Position = projection * view * vec4(in_position, 1.0);
    }
    ''',
    fragment_shader='''
        #version 330

        uniform vec4 color;

        out vec4 f_color;

        void main() {
            f_color = color;
        }
    '''
)

point_program = context.program(
    vertex_shader = '''
    #version 330

    uniform mat4 projection;
    uniform mat4 view;

    in vec3 in_position;
    in vec2 in_coordinate;
    in float in_id;

    out vec2 v_coordinate;
    flat out int v_id;

    void main() {
        gl_Position = projection * view * vec4(in_position, 1.0);

        v_coordinate = in_coordinate;
        v_id = int(in_id);
    }
    ''',
    fragment_shader='''
        #version 330

        uniform vec3 color;
        uniform sampler2D center_point_mask;
        uniform sampler2D end_point_mask;

        in vec2 v_coordinate;
        flat in int v_id;

        out vec4 f_color;

        void main() {
            float end_point = texture(end_point_mask, v_coordinate).a;
            float center_point = texture(center_point_mask, v_coordinate).a;

            if(v_id == 0) {
                f_color = vec4(color, center_point);
            } else {
                f_color = vec4(color, end_point);
            }
        }
    '''
)

os.makedirs("frames", exist_ok=True)

points_per_axis = 20
total_points = int(np.power(points_per_axis, 3))

x = np.linspace(-1.5, 1.5, points_per_axis)
y = np.linspace(-1.5, 1.5, points_per_axis)
z = np.linspace(-1.5, 1.5, points_per_axis)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
particle_positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

#particle_positions = 0.8 * np.random.randn(total_points,3)
particle_velocities = np.zeros((total_points, 3))
particle_colors = np.random.uniform(0.75, 1.00, (total_points, 3))

particle_vertices = np.hstack([particle_positions, particle_velocities, particle_colors])

line_positions = np.array([
    [-2.0, -2.0, -2.0],
    [ 3.1, -2.0, -2.0],
    [-2.0,  3.1, -2.0],
    [-2.0, -2.0,  3.1],

    [-1.0, -1.9, -2.0], [-1.0, -2.1, -2.0],
    [ 0.0, -1.9, -2.0], [ 0.0, -2.1, -2.0],
    [ 1.0, -1.9, -2.0], [ 1.0, -2.1, -2.0],
    [ 2.0, -1.9, -2.0], [ 2.0, -2.1, -2.0],
    [ 3.0, -1.9, -2.0], [ 3.0, -2.1, -2.0],

    [-1.9, -1.0, -2.0], [-2.1, -1.0, -2.0],
    [-1.9,  0.0, -2.0], [-2.1,  0.0, -2.0],
    [-1.9,  1.0, -2.0], [-2.1,  1.0, -2.0],
    [-1.9,  2.0, -2.0], [-2.1,  2.0, -2.0],
    [-1.9,  3.0, -2.0], [-2.1,  3.0, -2.0],

    [-2.0, -1.9, -1.0], [-2.0, -2.1, -1.0],
    [-2.0, -1.9,  0.0], [-2.0, -2.1,  0.0],
    [-2.0, -1.9,  1.0], [-2.0, -2.1,  1.0],
    [-2.0, -1.9,  2.0], [-2.0, -2.1,  2.0],
    [-2.0, -1.9,  3.0], [-2.0, -2.1,  3.0],
], dtype='f4')

line_indices = np.array([
     0,  1,
     0,  2,
     0,  3,

     4,  5,
     6,  7,
     8,  9,
    10, 11,
    12, 13,

    14, 15,
    16, 17,
    18, 19,
    20, 21,
    22, 23,

    24, 25,
    26, 27,
    28, 29,
    30, 31,
    32, 33
], dtype='i4')

line_color = np.array([0.96, 0.93, 0.83, 1.0])

center_point_image = Image.open("./texture-data/center_point.png").convert('RGBA')
end_point_image = Image.open("./texture-data/end_point.png").convert('RGBA')

center_point_image = center_point_image.transpose(Image.FLIP_TOP_BOTTOM)
end_point_image = end_point_image.transpose(Image.FLIP_TOP_BOTTOM)

center_point_image_width, center_point_image_height = center_point_image.size
end_point_image_width, end_point_image_height = end_point_image.size

center_point_data = center_point_image.tobytes()
end_point_data = end_point_image.tobytes()

center_point_texture = context.texture((center_point_image_width, center_point_image_height), components=4, data=center_point_data)
end_point_texture = context.texture((end_point_image_width, end_point_image_height), components=4, data=end_point_data)

center_point_texture.filter = (mgl.LINEAR, mgl.LINEAR)
end_point_texture.filter = (mgl.LINEAR, mgl.LINEAR)

center_point_texture.use(location=0)
end_point_texture.use(location=1)

point_positions = np.array([
    [-2.63, -2.33, -2.0],
    [-2.2,  -2.33, -2.0],
    [-2.63, -1.68,  -2.0],

    [-2.63, -1.68,  -2.0],
    [-2.2,  -2.33, -2.0],
    [-2.2,  -1.68,  -2.0],

    [2.92,  -1.8, -2.0],
    [3.08,  -1.8, -2.0],
    [2.92,  -1.54, -2.0],

    [2.92,  -1.54, -2.0],
    [3.08,  -1.8, -2.0],
    [3.08,  -1.54, -2.0],

    [-2.08, -1.8, 3.0],
    [-1.92, -1.8, 3.0],
    [-2.08, -1.54, 3.0],

    [-2.08, -1.54, 3.0],
    [-1.92, -1.8, 3.0],
    [-1.92, -1.54, 3.0],

    [-1.78, 2.87, -2.0],
    [-1.62, 2.87, -2.0],
    [-1.78, 3.13, -2.0],

    [-1.78, 3.13, -2.0],
    [-1.62, 2.87, -2.0],
    [-1.62, 3.13, -2.0]
], dtype='f4')

point_coordinates = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype='f4')
point_coordinates = np.tile(point_coordinates, (4, 1))

point_ids = np.array([
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
    [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
    [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
], dtype='f4')

point_color = np.array([0.96, 0.93, 0.83])

point_vertices = np.hstack([point_positions, point_coordinates, point_ids])

particle_vertex_buffer = context.buffer((particle_vertices.astype('f4').tobytes()))
line_vertex_buffer = context.buffer((line_positions.astype('f4').tobytes()))
point_vertex_buffer = context.buffer((point_vertices.astype('f4').tobytes()))

line_index_buffer = context.buffer((line_indices.astype('i4').tobytes()))

particle_vertex_array= context.vertex_array(particle_program, [(particle_vertex_buffer, '3f4 3f4 3f4', 'in_position', 'in_velocity', 'in_color')])
line_vertex_array = context.vertex_array(axis_program, [(line_vertex_buffer, '3f', 'in_position')], index_buffer=line_index_buffer)
point_vertex_array = context.vertex_array(point_program, [(point_vertex_buffer, '3f 2f 1f', 'in_position', 'in_coordinate', 'in_id')])

light = np.array([0.0, 0.0, 10.0], dtype='f4')
camera = np.array([1.0, 3.0, 10.0], dtype='f4')

projection = Matrix44.perspective_projection(45.0, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)
view = Matrix44.look_at(
    eye=camera,
    target=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0)
)

frame_buffer = context.simple_framebuffer((WINDOW_WIDTH, WINDOW_HEIGHT))
frame_buffer.use()

context.clear(0.1, 0.1, 0.1, 1.0, depth=1.0)
context.enable(mgl.DEPTH_TEST | mgl.PROGRAM_POINT_SIZE | mgl.BLEND)

particle_program['projection'].write(projection.astype('f4').tobytes())
particle_program['view'].write(view.astype('f4').tobytes())
particle_program['light'].value = light 
particle_program['camera'].value = camera

axis_program['projection'].write(projection.astype('f4').tobytes())
axis_program['view'].write(view.astype('f4').tobytes())
axis_program['color'].write(line_color.astype('f4').tobytes())

point_program['projection'].write(projection.astype('f4').tobytes())
point_program['view'].write(view.astype('f4').tobytes())
point_program['color'].write(point_color.astype('f4').tobytes())
point_program['center_point_mask'].value = 0
point_program['end_point_mask'].value = 1

start_time = time.perf_counter()

for i in range(FRAMES):
    particle_vertex_array.transform(particle_vertex_buffer, mode=mgl.POINTS)
    frame_buffer.clear(0.1, 0.1, 0.1, 1.0)

    line_vertex_array.render(mgl.LINES)
    point_vertex_array.render(mgl.TRIANGLES)
    particle_vertex_array.render(mgl.POINTS)

    image = Image.frombytes('RGB', frame_buffer.size, frame_buffer.read(), 'raw', 'RGB', 0, -1)
    image.save(f"frames/frame_{i:03d}.png")

    if((i+1) % int(FRAMES / 10) == 0): 
        print("Progress: " + str((i+1)/FRAMES*100.0) + "%")

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print("Time to render: " + str(elapsed_time))