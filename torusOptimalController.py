import sympy as sp
import numpy as np
import osqp
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Physical Constants ---
m1, m2, m3 = 0.181, 0.181, 0.05  # Masses
l1, l2, l3 = 0.250, 0.250, 0.10  # Lengths
I3_val = 0.01                    # Moment of inertia for the wrist
g_val = 9.81

# --- Controller Gains ---
kp, ki, kd = 40.0, 0.0, 15.0
integral_error = np.zeros(3)

# --- Symbolic Derivation ---
q1, q2, q3 = sp.symbols('q1 q2 q3')
dq1, dq2, dq3 = sp.symbols('dq1 dq2 dq3')

# Forward Kinematics (Position + Orientation)
x = l1 * sp.sin(q1) + l2 * sp.sin(q1 + q2) + l3 * sp.sin(q1 + q2 + q3)
y = -l1 * sp.cos(q1) - l2 * sp.cos(q1 + q2) - l3 * sp.cos(q1 + q2 + q3)
phi = q1 + q2 + q3
z = sp.Matrix([x, y, phi])
J = z.jacobian(sp.Matrix([q1, q2, q3]))

# Dynamics (Mass Matrix M, Coriolis C, Gravity G)
# 2R portion
m11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * sp.cos(q2)
m12 = m2 * l2**2 + m2 * l1 * l2 * sp.cos(q2)
M22 = sp.Matrix([[m11, m12], [m12, m2 * l2**2]])

# Expand to 3R
M3 = sp.zeros(3, 3)
M3[:2, :2] = M22
M3[2, 2] = I3_val

c1 = -m2 * l1 * l2 * sp.sin(q2) * (2 * dq1 * dq2 + dq2**2)
c2 = m2 * l1 * l2 * sp.sin(q2) * dq1**2
C3 = sp.Matrix([c1, c2, 0])

y1 = -l1 * sp.cos(q1)
y2 = -l1 * sp.cos(q1) - l2 * sp.cos(q1 + q2)
V = m1 * g_val * y1 + m2 * g_val * y2 # Assuming wrist mass is small/neglected for G here
G3 = sp.Matrix([sp.diff(V, q1), sp.diff(V, q2), 0])

# Lambdify for performance
z_func = sp.lambdify((q1, q2, q3), z, 'numpy')
J_func = sp.lambdify((q1, q2, q3), J, 'numpy')
M_func = sp.lambdify((q1, q2, q3), M3, 'numpy')
C_func = sp.lambdify((q1, q2, q3, dq1, dq2, dq3), C3, 'numpy')
G_func = sp.lambdify((q1, q2, q3), G3, 'numpy')

def get_control_torques(q_curr, dq_curr, z_des, z_dot_des, dt):
    global integral_error
    z_c = z_func(*q_curr).flatten()
    jac = J_func(*q_curr)
    M = M_func(*q_curr)
    C = C_func(*q_curr, *dq_curr).flatten()
    G = G_func(*q_curr).flatten()

    error = z_des - z_c
    integral_error += error * dt
    ddz_des = kp * error + kd * (z_dot_des - jac @ dq_curr) + ki * integral_error

    # Optimization Setup (OSQP)
    # Variables: [tau1, tau2, tau3, ddq1, ddq2, ddq3] (6 variables)
    A_dyn = sparse.hstack([-sparse.eye(3), sparse.csc_matrix(M)])
    A_task = sparse.hstack([sparse.csc_matrix((3, 3)), sparse.csc_matrix(jac)])
    A_full = sparse.vstack([A_dyn, A_task]).tocsc()
    b_full = np.concatenate([(-C - G), ddz_des])

    H = sparse.diags([1.0, 1.0, 1.0, 0.1, 0.1, 0.1], format='csc') # Weights for tau and ddq
    f = np.zeros(6)

    prob = osqp.OSQP()
    prob.setup(H, f, A_full, b_full, b_full, verbose=False)
    res = prob.solve()
    return res.x[:3]

def robot_dynamics(state, t, z_des, z_dot_des, dt):
    q = state[:3]
    dq = state[3:]
    tau = get_control_torques(q, dq, z_des, z_dot_des, dt)
    
    M = M_func(*q)
    C = C_func(*q, *dq).flatten()
    G = G_func(*q).flatten()
    
    ddq = np.linalg.solve(M, (tau - C - G))
    return np.concatenate([dq, ddq])

# --- Simulation Execution ---
t_span = np.linspace(0, 10, 500)
dt = t_span[1] - t_span[0]
initial_state = [0.3, 0.2, 0.1, 0, 0, 0] # q1, q2, q3, dq...
z_desired = np.array([-0.3, 0.2, np.pi]) # [x, y, phi]
z_dot_desired = np.zeros(3)

sim_result = odeint(robot_dynamics, initial_state, t_span, args=(z_desired, z_dot_desired, dt))

# --- Animation ---
def get_arm_positions(q1, q2, q3):
    x1, y1 = l1 * np.sin(q1), -l1 * np.cos(q1)
    x2, y2 = x1 + l2 * np.sin(q1 + q2), y1 - l2 * np.cos(q1 + q2)
    x3, y3 = x2 + l3 * np.sin(q1 + q2 + q3), y2 - l3 * np.cos(q1 + q2 + q3)
    return [0, x1, x2, x3], [0, y1, y2, y3]

fig, ax = plt.subplots(figsize=(6, 6))
line, = ax.plot([], [], 'o-', lw=4)
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.7, 0.7)
ax.grid()
ax.set_title("3R Robot Arm with Wrist Control")

def update(frame):
    q = sim_result[frame, :3]
    xs, ys = get_arm_positions(*q)
    line.set_data(xs, ys)
    return line,
# Updated Animation setup
ani = FuncAnimation(
    fig, 
    update, 
    frames=len(t_span), 
    interval=5, 
    blit=True, 
    repeat=True,        # Ensure the animation loops
    repeat_delay=5000   # Pause for 5000ms (5 seconds) before restarting
)

plt.show()