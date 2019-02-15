"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

maintainer: Atsushi Sakai (@Atsushi_twi)
author: Shohei Fujii (@ompugao)
"""

import matplotlib.pyplot as plt
import copy
import cvxpy
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent.parent.joinpath("PathPlanning/CubicSpline")))

np.core.arrayprint._format_options['linewidth'] = 160

try:
    import cubic_spline_planner
except:
    raise

"""
u: -2.0 ~ 2.0
xdiff: ~1.0になってほしい
v: ~5.0
"""

NX = 5  # z = x, y, theta, dottheta, v
NU = 2  # a = [right, left]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.00001, 0.00001])   # input difference cost matrix
Q = np.diag([10.0, 10.0, 1, 0.5, 15.0])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5  # stop speed
MAX_TIME = 100.0  # max simulation time

# iterative paramter
MAX_ITER = 10  # Max iteration
DU_TH = 0.00001  # iteration finish param

TARGET_SPEED = 3.0  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 2.0  # [m]
WIDTH = 2.0  # [m]
# BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
# WB = 2.5  # [m]
WHEEL_RADIUS = 0.1 #[m]
WHEEL_MASS = 0.3 # [kg]
Iw = 1.0/2 * WHEEL_MASS * WHEEL_RADIUS * WHEEL_RADIUS

MASS = 10.0 # [kg]
Iv = 1.0/2 * MASS * 0.5 * 0.5
# Iv = 0.0
INPUT_GAIN = 1.0
DAMPING_GAIN = 0.000001

MAX_SPEED = 5  # maximum speed [m/s]
MIN_SPEED = -2  # minimum speed [m/s]

MAX_MOTOR_TORQUE = np.pi  # maximum motor angle [rad]
MAX_MOTOR_TORQUE_VEL = np.pi * 0.5  # maximum accel [m/ss]
show_animation = True

try:
    profile
except NameError as e:
    def profile(func):
        return func

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, thetadot=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.thetadot = thetadot
        self.v = v

    def __repr__(self, ):
        return '<State: x=%s, y=%s, theta=%s, thetadot=%s, v=%s>'%(self.x, self.y, self.theta, self.thetadot, self.v)

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

@profile
def get_linear_model_matrix(v, theta):
    """
    v: velocity
    theta: car direction
    """

    l = WIDTH/2.0
    A = np.identity(NX)
    A[0, 4] += DT * math.cos(theta)
    A[1, 4] += DT * math.sin(theta)
    A[2, 3] += DT * 1.0
    A[3, 3] += DT * (-2.0 * DAMPING_GAIN * l * l) / (WHEEL_RADIUS*WHEEL_RADIUS * Iv + l*l*2.0 * Iw)
    A[4, 4] += DT * (-2.0 * DAMPING_GAIN) / (WHEEL_RADIUS*WHEEL_RADIUS*MASS + 2.0*Iw)

    B = np.zeros((NX, NU))
    b1 = WHEEL_RADIUS*INPUT_GAIN*l / (WHEEL_RADIUS*WHEEL_RADIUS*Iv + l*l*2*Iw)
    b2 = WHEEL_RADIUS*INPUT_GAIN / (WHEEL_RADIUS*WHEEL_RADIUS*MASS + 2*Iw)
    B[3, 0] += DT * b1
    B[3, 1] += DT * (-b1)
    B[4, 0] += DT * b2
    B[4, 1] += DT * b2

    # check from here
    C = np.zeros(NX)
    # C[0] = DT * v * math.sin(theta) * theta
    # C[1] = - DT * v * math.cos(theta) * theta
    # C[3] = - v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

@profile
def plot_car(x, y, theta, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    radius = WIDTH/2.0
    circle = plt.Circle((x, y), radius, facecolor='w', edgecolor='r')
    plt.gcf().gca().add_artist(circle)
    direction = np.array([[x, x + math.cos(theta) * radius],
                          [y, y + math.sin(theta) * radius]])

    plt.plot(direction[0], direction[1], truckcolor)
    plt.plot(x, y, "*g")

@profile
def update_state(state, u):
    # input check
    u_copy = copy.deepcopy(u)
    u_copy[0] = np.clip(u_copy[0], -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)
    u_copy[1] = np.clip(u_copy[1], -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)

    # state.x = state.x + state.v * math.cos(state.theta) * DT
    # state.y = state.y + state.v * math.sin(state.theta) * DT
    #state.theta = state.theta + state.v / WB * math.tan(delta) * DT
    A, B, C = get_linear_model_matrix(state.v, state.theta)
    newz = A.dot(np.array([state.x, state.y, state.theta, state.thetadot, state.v]).T) + B.dot(u_copy.T)
    state.x        = newz[0]
    state.y        = newz[1]
    state.theta    = newz[2]
    state.thetadot = newz[3]
    state.v        = newz[4]

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state. v < MIN_SPEED:
        state.v = MIN_SPEED

    return state

@profile
def get_nparray_from_matrix(x):
    return np.array(x).flatten()

@profile
def calc_nearest_index(state, cx, cy, ctheta, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(ctheta[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

@profile
def predict_motion(x0, ou, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], theta=x0[2], thetadot=x0[3], v=x0[4])
    for i, u in enumerate(ou.T):
        state = update_state(state, u)
        xbar[0, i+1] = state.x
        xbar[1, i+1] = state.y
        xbar[2, i+1] = state.theta
        xbar[3, i+1] = state.thetadot
        xbar[4, i+1] = state.v

    return xbar

@profile
def iterative_linear_mpc_control(xref, x0, ou):
    """
    MPC contorl with updating operational point iteratively
    """

    if ou is None:
        ou = np.zeros((2,T))

    cmap = plt.get_cmap("tab10")
    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ou, xref)
        pou_r, pou_l = ou[0, :], ou[1, :]
        ou, ox, oy, otheta, othetadot, ov = linear_mpc_control(xref, xbar, x0)
        du = sum(abs(ou[0, :] - pou_r)) + sum(abs(ou[1, :] - pou_l))  # calc u change value

        plt.plot(ox, oy, color=cmap(i), marker="x", label="iteration")
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")
    print("iteration: %s"%i)

    return ou, ox, oy, otheta, othetadot, ov


@profile
def linear_mpc_control(xref, xbar, x0):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(xbar[4, t], xbar[2, t])
        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            # constraints += [cvxpy.abs(u[0, t + 1] - u[0, t]) <= MAX_MOTOR_TORQUE_VEL * DT]
            # constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_MOTOR_TORQUE_VEL * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[4, :] <= MAX_SPEED]
    constraints += [x[4, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_MOTOR_TORQUE]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_MOTOR_TORQUE]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    ou = np.zeros((NU, T))

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        otheta = get_nparray_from_matrix(x.value[2, :])
        othetadot = get_nparray_from_matrix(x.value[3, :])
        ov = get_nparray_from_matrix(x.value[4, :])
        ou[0,:] = get_nparray_from_matrix(u.value[0, :])
        ou[1,:] = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        ou, ox, oy, otheta, othetadot, ov = None, None, None, None, None, None

    return ou, ox, oy, otheta, othetadot, ov

@profile
def calc_ref_trajectory(state, cx, cy, ctheta, ck, sp, dl, pind):
    """
    cx: course x position list
    cy: course y position list
    ctheta: course theta position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]
    """

    #from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)
    ind, _ = calc_nearest_index(state, cx, cy, ctheta, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = ctheta[ind]
    xref[3, 0] = 0
    xref[4, 0] = sp[ind]

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = ctheta[ind + dind]
            xref[3, i] = 0.0
            xref[4, i] = sp[ind + dind]
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = ctheta[ncourse - 1]
            xref[3, i] = 0.0
            xref[4, i] = sp[ncourse - 1]

    return xref, ind

@profile
def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.sqrt(dx ** 2 + dy ** 2)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False

@profile
def do_simulation(cx, cy, ctheta, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    ctheta: course theta position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial theta compensation
    if state.theta - ctheta[0] >= math.pi:
        state.theta -= math.pi * 2.0
    elif state.theta - ctheta[0] <= -math.pi:
        state.theta += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    theta = [state.theta]
    thetadot = [state.thetadot]
    v = [state.v]

    t = [0.0]
    u = [np.zeros(2)]
    target_ind, _ = calc_nearest_index(state, cx, cy, ctheta, 0)

    ou = None

    ctheta = smooth_theta(ctheta)

    while MAX_TIME >= time:
        xref, target_ind = calc_ref_trajectory(
            state, cx, cy, ctheta, ck, sp, dl, target_ind)

        # xref contains
        #  xref[0, i] = cx[ind + dind] # x
        #  xref[1, i] = cy[ind + dind] # y
        #  xref[2, i] = sp[ind + dind] # speed profile
        #  xref[3, i] = ctheta[ind + dind] # theta

        x0 = [state.x, state.y, state.theta, state.thetadot, state.v]  # current state

        ou, ox, oy, otheta, othetadot, ov = iterative_linear_mpc_control(
            xref, x0, ou)

        state = update_state(state,  ou[:, 0])
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        theta.append(state.theta)
        thetadot.append(state.thetadot)
        v.append(state.v)
        t.append(time)
        u.append(ou[:, 0])

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        # if time > 15:
            # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

        if show_animation:  # pragma: no cover
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            print('State: %s'%state)
            print('Input: %s'%ou[:, 0])

            # print('Target: %d', target_ind)
            plot_car(state.x, state.y, state.theta)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[m/s]:" + str(round(state.v, 2)))
            plt.pause(0.0001)
            plt.cla()
            print(time)

    return t, x, y, theta, v, u

@profile
def calc_speed_profile(cx, cy, ctheta, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - ctheta[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

@profile
def smooth_theta(theta):

    for i in range(len(theta) - 1):
        dtheta = theta[i + 1] - theta[i]

        while dtheta >= math.pi / 2.0:
            theta[i + 1] -= math.pi * 2.0
            dtheta = theta[i + 1] - theta[i]

        while dtheta <= -math.pi / 2.0:
            theta[i + 1] += math.pi * 2.0
            dtheta = theta[i + 1] - theta[i]

    return theta

@profile
def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, ctheta, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, ctheta, ck

@profile
def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, ctheta, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, ctheta, ck

@profile
def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, ctheta, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    ctheta = [i - math.pi for i in ctheta]

    return cx, cy, ctheta, ck

@profile
def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, ctheta, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, ctheta, ck

@profile
def get_switch_back_course(dl):
    ax = np.array([0.0, 30.0, 6.0, 20.0, 35.0])/3.0
    ay = np.array([0.0, 0.0, 20.0, 35.0, 20.0])/3.0
    cx, cy, ctheta, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = np.array([35.0, 10.0, 0.0, 0.0])/3.0
    ay = np.array([20.0, 30.0, 5.0, 0.0])/3.0
    cx2, cy2, ctheta2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ctheta2 = [i - math.pi for i in ctheta2]
    cx.extend(cx2)
    cy.extend(cy2)
    ctheta.extend(ctheta2)
    ck.extend(ck2)

    return cx, cy, ctheta, ck
@profile
def get_forward_course_mini(dl):
    ax = np.array([0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0])/3.0
    ay = np.array([0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0])/3.0
    cx, cy, ctheta, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, ctheta, ck

def main():
    print(__file__ + " start!!")

    dl = 0.2 # course tick
    # cx, cy, ctheta, ck = get_straight_course(dl)
    # cx, cy, ctheta, ck = get_straight_course2(dl)
    # cx, cy, ctheta, ck = get_straight_course3(dl)
    # cx, cy, ctheta, ck = get_forward_course(dl)
    # cx, cy, ctheta, ck = get_switch_back_course(dl)
    cx, cy, ctheta, ck = get_forward_course_mini(dl)

    sp = calc_speed_profile(cx, cy, ctheta, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], theta=ctheta[0], v=0.0, thetadot=0.0)
    #initial_state = State(x=cx[0], y=1.0, theta=0.0, v=0.0, thetadot=0.0)

    t, x, y, theta, v, u = do_simulation(
        cx, cy, ctheta, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


def main2():
    print(__file__ + " start!!")

    dl = 0.2  # course tick
    cx, cy, ctheta, ck = get_straight_course2(dl)

    sp = calc_speed_profile(cx, cy, ctheta, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], theta=0.0, v=0.0, thetadot=0.0)

    t, x, y, theta, v, u = do_simulation(
        cx, cy, ctheta, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()

if __name__ == '__main__':
    # print('main')
    # import line_profiler
    # pr = line_profiler.LineProfiler()
    # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

    main()
    # main2()
