import sympy as sp
import numpy as np
import osqp
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def robot_dynamics(state, t, tau_func):
    # Ensure state is 1D with 4 elements
    state = np.array(state).flatten()
    q_curr = state[:2]
    dq_curr = state[2:]
    
    # Get current torques
   # Ensure all are 1D arrays of size 2
    tau = tau_func(q_curr, dq_curr).flatten()

    M = M_func(q_curr[0], q_curr[1])
    C = C_func(q_curr[0], q_curr[1], dq_curr[0], dq_curr[1]).flatten()
    G = G_func(q_curr[0], q_curr[1]).flatten()

    # Now the subtraction will work
    ddq = (np.linalg.solve(M, (tau - C - G))).flatten()

    # Ensure return value is strictly 1D and length 4
    return np.concatenate([dq_curr.flatten(), ddq.flatten()])

def get_control_torques(q_curr, dq_curr, z_desired, z_dot_desired, kp, ki, kd, dt):
    # Global variable
    global integral_error

    # 1. Compute state and Jacobians (Ensure these return flat 1D arrays/matrices)
    z_curr = z_func(q_curr[0], q_curr[1]).flatten()
    J = J_func(q_curr[0], q_curr[1]) # This is likely (2,2)
    M = M_func(q_curr[0], q_curr[1]) # This is likely (2,2)
    C = C_func(q_curr[0], q_curr[1], dq_curr[0], dq_curr[1]).flatten()
    G = G_func(q_curr[0], q_curr[1]).flatten()

    # 2. PID Target
    max_integral = 5.0 # Tune this threshold
    error = np.array(z_desired) - z_curr
    integral_error += error * dt 
    integral_error = np.clip(integral_error, -max_integral, max_integral)
    ddz_des = kp * error + kd * (z_dot_desired - J @ dq_curr) + ki * integral_error

    # 3. Define Constraints
    # A_dyn: [-I, M] (2x4)
    A_dyn = sparse.hstack([-sparse.eye(2), sparse.csc_matrix(M)])
    # A_task: [0, J] (2x4)
    A_task = sparse.hstack([sparse.csc_matrix((2, 2)), sparse.csc_matrix(J)])
    
    # A_full (4x4)
    A_full = sparse.vstack([A_dyn, A_task]).tocsc()
    
    # b_full must be (4,)
    b_full = np.concatenate([(-C - G).flatten(), ddz_des.flatten()])
    
    # 4. Setup Solver correctly (Defined ONCE)
    prob = osqp.OSQP()
    
    # Define objective: Minimize torque (first 2 variables), Accelerations (last 2)
    # H must be (4, 4)
    H = sparse.diags([1.0, 1.0, 0.0, 0.0], format='csc') 
    f = np.zeros(4) 
    
    # A_full is (4, 4), b_full is (4,)
    # Setup ONLY once
    prob.setup(H, f, A_full, b_full, b_full, alpha=1.0, verbose=False)
    
    res = prob.solve()
    print(f"Torques: {res.x[:2]}, Status: {res.info.status}")
    
    # 5. Fallback/Return
    return res.x[:2].flatten()

# --- Global Scope ---
# Define this alongside your other parameters like kp, kd, t_start, etc.
integral_error = np.array([0.0, 0.0])

# Physical constants
m1, m2 = 0.181, 0.181 # Mass constants
l1, l2 = 0.250, 0.250 # Length constants
g_val = 9.81  # Gravity constant

# Time constants
t_start = 0
t_end = 10
t_num = 1000

# Initial conditions
q_q_dot_initial = [-np.pi/4, np.pi/2, 0, 0] # [q1, q2, dq1, dq2]

# Desired final task space position and velocity
z_desired = [0.25, 0.25]
z_dot_desired = [0, 0]

# PD controller constants
kp = 40.0
ki = 0.0
kd = 15.0

# Define symbols
q1, q2 = sp.symbols('q1 q2')
dq1, dq2 = sp.symbols('dq1 dq2')
ddq1, ddq2 = sp.symbols('ddq1 ddq2')
tau1, tau2 = sp.symbols('tau1 tau2')

q = sp.Matrix([q1, q2])
dq = sp.Matrix([dq1, dq2])
ddq = sp.Matrix([ddq1, ddq2])
tau = sp.Matrix([tau1, tau2])

# Jacobian (Correct)
# Ensure x and y match the potential energy definition
x = l1 * sp.sin(q1) + l2 * sp.sin(q1 + q2)
y = -l1 * sp.cos(q1) - l2 * sp.cos(q1 + q2)
z = sp.Matrix([x, y])
J = z.jacobian(sp.Matrix([q1, q2]))

# Mass Matrix elements (derived from kinetic energy)
# M = [[m11, m12], [m21, m22]]
m11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * sp.cos(q2)
m12 = m2 * l2**2 + m2 * l1 * l2 * sp.cos(q2)
m21 = m12
m22 = m2 * l2**2

M = sp.Matrix([[m11, m12], [m21, m22]])

# Coriolis/Gravity vector C = C(q, dq) * dq
# C_ij = sum_k ( Christoffel_ijk * dq_k )
# Here is the explicit form for a 2-link arm:
c1 = -m2 * l1 * l2 * sp.sin(q2) * (2 * dq1 * dq2 + dq2**2)
c2 = m2 * l1 * l2 * sp.sin(q2) * dq1**2

C = sp.Matrix([c1, c2])

# Potential Energy and Gravity (Correct)
y1 = -l1 * sp.cos(q1)
y2 = -l1 * sp.cos(q1) - l2 * sp.cos(q1 + q2)
V = m1 * g_val * y1 + m2 * g_val * y2
G = sp.Matrix([sp.diff(V, q1), sp.diff(V, q2)])

# LAMBDIFY ONCE
z_func = sp.lambdify((q1, q2), z, 'numpy')
J_func = sp.lambdify((q1, q2), J, 'numpy')
M_func = sp.lambdify((q1, q2), M, 'numpy')
C_func = sp.lambdify((q1, q2, dq1, dq2), C, 'numpy')
G_func = sp.lambdify((q1, q2), G, 'numpy')

# Simulation Setup
t_span = np.linspace(t_start, t_end, t_num)

# Calculate the duration
duration = t_end - t_start

# Calculate the time step
dt = duration / (t_num - 1)
initial_state = q_q_dot_initial # [q1, q2, dq1, dq2]

# Integration
# We pass a lambda that captures the state to get torques at each step
sim_result = odeint(lambda s, t: robot_dynamics(s, t, lambda q, dq: get_control_torques(q, dq, z_desired, z_dot_desired, kp, ki, kd, dt)), 
     initial_state, t_span)

# # Plotting the results
# plt.figure(figsize=(10, 5))
# plt.plot(t_span, sim_result[:, 0], label='Joint 1')
# plt.plot(t_span, sim_result[:, 1], label='Joint 2')
# plt.title("Joint Positions over Time")
# plt.legend()
# plt.show()

# Convert joint angles to end-effector positions
def get_arm_positions(q1, q2):
    # Link positions (using L1=0.25, L2=0.25)
    # Match the downward configuration: x = l*sin(q), y = -l*cos(q)
    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)
    x2 = x1 + l2 * np.sin(q1 + q2)
    y2 = y1 - l2 * np.cos(q1 + q2)
    return [0, x1, x2], [0, y1, y2]

def update(frame):
    q1, q2 = sim_result[frame, 0], sim_result[frame, 1]
    xs, ys = get_arm_positions(q1, q2)
    line.set_data(xs, ys)
    return line,

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
line, = ax.plot([], [], 'o-', lw=4)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.grid()

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_span), interval=10, blit=True)
plt.show()