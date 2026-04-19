import sympy as sp
import numpy as np
import osqp
import numpy as np
from scipy import sparse
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def robot_dynamics(state, t, tau_func):
    # Ensure state is 1D with 4 elements
    state = np.array(state).flatten()
    q_curr = state[:2]
    dq_curr = state[2:]
    
    # Get current torques
    tau = tau_func(q_curr, dq_curr)
    
    # Forward dynamics
    M = M_func(q_curr[0], q_curr[1])
    C = C_func(q_curr[0], q_curr[1], dq_curr[0], dq_curr[1])
    G = G_func(q_curr[0], q_curr[1]) # Calculate Gravity
    
    # Calculate ddq: M * ddq = tau - C - G
    # Solving for ddq: ddq = M^-1 * (tau - C - G)
    ddq = (np.linalg.solve(M, (tau - C.flatten() - G.flatten()))).flatten()
    
    return np.concatenate([dq_curr, ddq])

def get_control_torques(q_curr, dq_curr, z_desired, z_dot_desired):
    """
    Computes optimal torques using OSQP to track z_desired.
    """
    # 1. Compute task space state
    z_curr = np.array([np.cos(q_curr[0]) + np.cos(q_curr[0]+q_curr[1]), 
                       np.sin(q_curr[0]) + np.sin(q_curr[0]+q_curr[1])])
    
    J = J_func(q_curr[0], q_curr[1])
    
    # 1. PD Control for target task acceleration
    kp, kd = 10.0, 3.0
    z_curr = np.array([np.cos(q_curr[0]) + np.cos(q_curr[0]+q_curr[1]), 
                       np.sin(q_curr[0]) + np.sin(q_curr[0]+q_curr[1])])
    z_dot_curr = J @ dq_curr
    ddz_des = kp * (z_desired - z_curr) + kd * (z_dot_desired - z_dot_curr)
    
     # 2. Dynamics and Kinematics components
    M = M_func(q_curr[0], q_curr[1]).astype(float)
    C = C_func(q_curr[0], q_curr[1], dq_curr[0], dq_curr[1]).flatten()
    G = G_func(q_curr[0], q_curr[1]).flatten()
    J = J_func(q_curr[0], q_curr[1])
    
    # 3. Setup QP variables
    n = 4  # [tau1, tau2, ddq1, ddq2]
    H = sparse.eye(n) * 1e-3  # Now H is defined!
    f = np.zeros(n)
    
    # 4. Define Constraints (Dynamics + Task Space)
    # Dynamics: -tau + M*ddq = -C - G
    A_dyn = sparse.hstack([-sparse.eye(2), sparse.csc_matrix(M)])
    b_dyn = -C - G
    
    # Task Space: J*ddq = ddz_des
    # (Note: compute ddz_des using PD control as previously discussed)
    A_task = sparse.hstack([sparse.csc_matrix((2, 2)), sparse.csc_matrix(J)])
    b_task = ddz_des
    
    A_full = sparse.vstack([A_dyn, A_task])
    b_full = np.concatenate([b_dyn, b_task])
    
    # 5. Solve
    prob = osqp.OSQP()
    prob.setup(H, f, A_full, b_full, b_full, alpha=1.0)
    res = prob.solve()
    
    return res.x[:2]

# Convert joint angles to end-effector positions
def get_arm_positions(q1, q2):
    # Link positions (using L1=0.25, L2=0.25)
    x1 = 0.25 * np.cos(q1)
    y1 = 0.25 * np.sin(q1)
    x2 = x1 + 0.25 * np.cos(q1 + q2)
    y2 = y1 + 0.25 * np.sin(q1 + q2)
    return [0, x1, x2], [0, y1, y2]

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
line, = ax.plot([], [], 'o-', lw=4)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.grid()

def update(frame):
    q1, q2 = sim_result[frame, 0], sim_result[frame, 1]
    xs, ys = get_arm_positions(q1, q2)
    line.set_data(xs, ys)
    return line,

# Physical constants
m1, m2 = 0.181, 0.181
l1, l2 = 0.250, 0.250
g_val = 9.81  # Gravity constant

# Define symbols
q1, q2 = sp.symbols('q1 q2')
dq1, dq2 = sp.symbols('dq1 dq2')
ddq1, ddq2 = sp.symbols('ddq1 ddq2')
tau1, tau2 = sp.symbols('tau1 tau2')

q = sp.Matrix([q1, q2])
dq = sp.Matrix([dq1, dq2])
ddq = sp.Matrix([ddq1, ddq2])
tau = sp.Matrix([tau1, tau2])

# Example Task Space: End Effector Position (L1=1, L2=1)
x = 1 * sp.cos(q1) + 1 * sp.cos(q1 + q2)
y = 1 * sp.sin(q1) + 1 * sp.sin(q1 + q2)
z = sp.Matrix([x, y])

# Jacobian
J = z.jacobian(q)

# Symbolic variables
q1, q2 = sp.symbols('q1 q2')
dq1, dq2 = sp.symbols('dq1 dq2')

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

# Gravity vector (Partial derivative of Potential Energy V w.r.t q)
# V = (m1/2 + m2) * g * l1 * sin(q1) + (m2/2) * g * l2 * sin(q1 + q2)
g1 = (m1/2 + m2) * g_val * l1 * sp.cos(q1) + (m2/2) * g_val * l2 * sp.cos(q1 + q2)
g2 = (m2/2) * g_val * l2 * sp.cos(q1 + q2)

G = sp.Matrix([g1, g2])

# Export functions for use in simulation
from sympy.utilities.lambdify import lambdify
J_func = lambdify((q1, q2), J, 'numpy')
M_func = sp.lambdify((q1, q2), M, 'numpy')
C_func = sp.lambdify((q1, q2, dq1, dq2), C, 'numpy')
G_func = sp.lambdify((q1, q2), G, 'numpy')

print("Functions generated successfully.")

# Simulation Setup
t_span = np.linspace(0, 10, 1000)
initial_state = [np.pi/4, np.pi/4, 0, 0] # [q1, q2, dq1, dq2]

# Integration
# We pass a lambda that captures the state to get torques at each step
sim_result = odeint(lambda s, t: robot_dynamics(s, t, lambda q, dq: get_control_torques(q, dq, [0.5, 0.5], [0, 0])), 
                    initial_state, t_span)

# # Plotting the results
# plt.figure(figsize=(10, 5))
# plt.plot(t_span, sim_result[:, 0], label='Joint 1')
# plt.plot(t_span, sim_result[:, 1], label='Joint 2')
# plt.title("Joint Positions over Time")
# plt.legend()
# plt.show()

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_span), interval=10, blit=True)
plt.show()