# Python simulation of a pendulum with air resistance (possible acc vector field)
# Emil Karlsson
# Start: 2021-02-03
"""
Idea
    Second order differential equation for a pendulum
    d2theta / dt2 + (mu * dtheta / dt) + (g / L * sin(theta)) = 0

    Seperate into two first order differential equations
    dtheta / dt = vel
    d2theta / dt2 = dvel / dt = acc 
    ==> acc + (mu * vel) + (g / L * sin(theta)) = 0

    vel = dtheta / dt
    acc = -(mu * vel) - (g / L * sin(theta))
    theta = [ theta, vel ]
    dtheta / dt = [ vel, acc ]

    define a function which returns the array of dtheta/dt. 
    which then can solve for dtheta/dt vy interating dtheta/dt from t_start to t_max (limits).

Comment
    Assumtion that mass = 1
"""

import numpy as np                                      # numpy
import matplotlib.pyplot as plt                         # plot
from matplotlib import animation                        # animation
from scipy.integrate import odeint                      # ode

# set x-axis ticks for fig to multiples of pi
def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

# innitial values
theta_start = np.pi/2                                   # starting value angle from y-negative axis
vel_start = 0                                           # starting value velocity
theta_vel_start = [theta_start, vel_start]              # starting value [ angle, velocity ]

lim = 5                                                 # figure x and y limits
step_dis = 0.2                                          # fig3 vector field step_dis value for x and y axis

dt = 0.01                                               # time shape length 
t_max = 150                                             # max x value
numframes = round(t_max/dt)                             # number of dt in t_max (rounded)     
t = np.linspace(0, t_max, numframes)                    # time space 

g = 9.82                                                # gravitational acc
mu = 0.1                                                # air resistance coefficience
L = 1                                                   # length of pendulum

extra = 0                                               # for when pendulum turns (only for positive turns)

# acceleration of pendulum
def acc(theta, vel):
    return - (mu * vel) - (g / L * np.sin(theta))

# ode 
def ode(theta, t, mu, g, L):   
    theta1 = theta[0]
    theta2 = theta[1]
    vel = theta2
    acceleration = acc(theta1, vel)
    dtheta_dt = [ vel, acceleration ]
    return dtheta_dt

# solve ode
solution = odeint(ode, theta_vel_start, t, args=(mu, g, L))

list_pos = solution[:, 0]                               # list to store angular position
list_vel = solution[:, 1]                               # list to store velocity
list_acc = []                                           # list to store acceleration

# add to list for acceleration
for i in range(len(solution)):
    list_acc.append(acc(solution[i][0], solution[i][1]))

"""
Print the values for angle, velocity and acceleration in a grid in the terminal.
"""
# Print values in grid (text)
def text():
    print("t\t", end="")
    print("Position", "Velocity", "Acceleration", sep="\t\t")
    for frame in range(len(list_pos)): # works to store other lists as well since they are the same lengths
        if frame == 0:
            print("0\t" + str(list_pos[frame]) + "\t" + str(list_vel[frame]) + "\t\t\t" + str(list_acc[frame]), end="\n")
        else:
            print(frame, list_pos[frame], list_vel[frame], list_acc[frame], sep="\t", end="\n")

"""
Figures
"""
fig = plt.figure(figsize=(14.5, 8.2))

gs = fig.add_gridspec(7, 6)

ax1 = fig.add_subplot(gs[:2, :], xlim=(-1, t_max+1), ylim=(-12, 12))                # value graph
ax2 = fig.add_subplot(gs[3:, :4], xlim=(-lim, lim + extra), ylim=(-lim, lim))       # vectorfield and pos/vel animation
ax3 = fig.add_subplot(gs[3:, 4:], xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))               # pendulum animation

"""
Graph the angle, velociity and acceleration functions on a graph (time-size).
"""
# pendulum values
ax1.set_xlabel("time")
ax1.set_ylabel("amplitude")

p1_1, = ax1.plot([], [])
p1_2, = ax1.plot([], [])
p1_3, = ax1.plot([], [])

ax1.legend((p1_1, p1_2, p1_3), ("Position (\u03F4)", "Velocity (d\u03F4)", "Acceleration (d2\u03F4)"), loc=1)
ax1.grid(True)

"""
Graph and animate (fades over time) the change in pendulums angle and position over time (t) in intervalls of dt (angle-velocity).
"""
#plt.rcParams['axes.facecolor'] = "black"                          # black background for 'spacey' effect :) (not working)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax2.set_xlabel("angle (\u03F4)")
ax2.set_ylabel("velocity (d\u03F4)")

# plot and color lines 
pendulum_line, = ax2.plot([], [], color="royalblue")                    # line 1 royalblue
pendulum_line2, = ax2.plot([], [], color="cornflowerblue")              # line 2 cornflowerblue
pendulum_line3, = ax2.plot([], [], color="lightskyblue")                # line 3 lightskyblue
pen, = ax2.plot([], [], color="royalblue", marker="o")                  # pen tip royalblue

# vector field
X, Y = np.meshgrid(np.arange(-lim, lim+extra, step_dis), np.arange(-lim, lim, step_dis), indexing='ij')
ax2.quiver(X, Y, Y, acc(X, Y), np.hypot(X, acc(X, Y)), units="x", width=0.03, scale=1/0.03)

"""
Pendulum animation
"""
ax3.set_xticks([], minor=False)
ax3.set_yticks([], minor=False)
ax3.set_aspect('equal')
ax3.set_xlabel("Pendulum")

pendulum, = ax3.plot([], [], lw=5)
ax3.axhline(y=0)
ax3.axvline(x=0)

"""
Animation
"""

# set all plot data to 0
def init():
    pendulum_line.set_data([], [])
    pendulum_line2.set_data([], [])
    pendulum_line3.set_data([], [])
    pen.set_data([], [])
    pendulum.set_data([],[])
    p1_1.set_data([], [])
    p1_2.set_data([], [])
    p1_3.set_data([], [])
    return pendulum_line, pendulum_line2, pendulum_line3, pen, pendulum, p1_1, p1_2, p1_3,

# animate order and timing 
def animate(frame):

    p1_1.set_data(t[:frame], list_pos[:frame])
    p1_2.set_data(t[:frame], list_vel[:frame])
    p1_3.set_data(t[:frame], list_acc[:frame])

    # line3
    if frame > 1000:
        x3 = list_pos[frame-1000: frame-749]
        y3 = list_vel[frame-1000: frame-749]
    elif frame > 749:
        x3 = list_pos[0: frame-749]
        y3 = list_vel[0: frame-749]
    else:
        x3 = []
        y3 = []

    # line2
    if frame > 750:
        x2 = list_pos[frame-750: frame-490]
        y2 = list_vel[frame-750: frame-490]
    elif frame > 490:
        x2 = list_pos[0: frame-490]
        y2 = list_vel[0: frame-490]
    else:
        x2 = []
        y2 = []

    # line1
    if frame > 500:
        x = list_pos[frame-500: frame]
        y = list_vel[frame-500: frame]
    else:        
        x = list_pos[0: frame]
        y = list_vel[0: frame]
    
    pen.set_data(list_pos[frame], list_vel[frame])

    pendulum_line.set_data(x, y)
    pendulum_line2.set_data(x2, y2) 
    pendulum_line3.set_data(x3, y3)

    # pendulum
    x4 = np.array([0, np.sin(list_pos[frame])])
    y4 = np.array([0, -np.cos(list_pos[frame])])
    pendulum.set_data(x4, y4)
    return pendulum_line, pendulum_line2, pendulum_line3, pen, pendulum, p1_1, p1_2, p1_3,

#text()
anim = animation.FuncAnimation(fig, animate, frames=numframes, init_func=init, interval=1, blit=True, repeat=False)  

plt.show()