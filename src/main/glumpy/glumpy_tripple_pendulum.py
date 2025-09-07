import numpy as np
from glumpy import app, gl, glm, gloo

vertex = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 a_position;
attribute vec4 a_fill;
attribute vec4 a_outline;

varying vec4 v_fill;
varying vec4 v_outline;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = 30.0;

    v_fill = a_fill;
    v_outline = a_outline;
}
"""

fragment = """
varying vec4 v_fill;
varying vec4 v_outline;

void main() {
    float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
    float edge = 0.1;
    float cutoff = 0.5;
    float outline = cutoff - edge;

    if (dist < outline) {
        gl_FragColor = v_fill;
    } else if (dist > cutoff) {
        discard;
    } else {
        gl_FragColor = v_outline;
    }
}
"""

def f(Y, l=1, g=9.81):
    t1, t2, t3, o1, o2, o3 = Y
    
    cos1 = np.cos(t1 - t2)
    cos2 = np.cos(t1 - t3)
    cos3 = np.cos(t2 - t3)

    sin1 = np.sin(t1 - t2)
    sin2 = np.sin(t1 - t3)
    sin3 = np.sin(t2 - t3)

    A = np.array([[ (7/3)*l, (3/2)*l*cos1, (1/2)*l*cos2 ],
                  [ (3/2)*l*cos1, (4/3)*l, (1/2)*l*cos3 ],
                  [ (1/2)*l*cos2, (1/2)*l*cos3, (1/3)*l ]], np.float32)
    
    b = np.array([[ (-3/2)*l*o2**2*sin1 - (1/2)*l*o3**2*sin2 - (5/2)*g*np.sin(t1) ],
                  [ (3/2)*l*o1**2*sin1 - (1/2)*l*o3**2*sin3 - (3/2)*g*np.sin(t2) ],
                  [ (1/2)*l*o1**2*sin2 + (1/2)*l*o2**2*sin3 - (1/2)*g*np.sin(t3) ]], np.float32)
    
    x = np.linalg.solve(A, b)
    
    return np.array([o1, o2, o3, x[0][0], x[1][0], x[2][0]])

def rk4(Y, dt):
    #k1 = f(Y)
    #k2 = f(Y + dt*k1/2)
    #k3 = f(Y + dt*k2/2)
    #k4 = f(Y + dt*k3)

    yk = Y
#   
    for _ in range(10):
        yk = Y + dt * f(yk)

    return yk

axis_points = 5
n = int(np.power(axis_points, 3))

state = np.zeros(n, [("a_position", np.float32, 3),
                     ("speed", np.float32, 3),
                     ("a_fill", np.float32, 4)])

x = np.linspace(-0.5, 0.5, axis_points)
y = np.linspace(-0.5, 0.5, axis_points)
z = np.linspace(-0.5, 0.5, axis_points)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
#points = 0.1 * np.random.randn(n,3)

state["a_position"] = points
colors = np.random.uniform(0.75, 1.00, (n, 4))
colors[:,3] = 1
state["a_fill"] = colors
state = state.view(gloo.VertexBuffer)

tripple_pendulum = gloo.Program(vertex, fragment, count=n)
tripple_pendulum.bind(state)

model = np.eye(4, dtype=np.float32)
view = glm.translation(0, 0, -5)
projection = glm.perspective(45.0, 1.0, 1.0, 100.0)

tripple_pendulum["u_model"] = model
tripple_pendulum["u_view"] = view
tripple_pendulum["u_projection"] = projection

tripple_pendulum["a_outline"] = 0, 0, 0, 1

window = app.Window(width=1024, height=1024,  color=(0.30, 0.30, 0.35, 1.00))

@window.event
def on_draw(dt):
    window.clear()
    tripple_pendulum.draw(gl.GL_POINTS)

@window.event
def on_resize(width,height):
    tripple_pendulum['u_projection'] = glm.perspective(45.0, width / float(height), 1.0, 1000.0)

@window.timer(1/60.)
def timer(dt):
    global state
    
    for i in range(n):
        y = np.concatenate([state[i]["a_position"], state[i]["speed"]])
        y_new = rk4(y, dt/10)

        state["a_position"][i] = y_new[:3]
        state["speed"][i] = y_new[3:]

app.run()