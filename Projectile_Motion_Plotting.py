import numpy as np
import math
import matplotlib.pyplot as plt

def sqrt(x): return np.sqrt(x)

def draw_graph(x,y,xmin,xmax,sol):
    plt.plot(x,y,markersize=.8)
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.title('Bullet Trajectory')
    plt.xlim([xmin, 1.5*xmax])

def draw_trajectory(v0,sol):
    g = 9.8
    def sqrt(x): return np.sqrt(x)

    delx = 1
    dely = 4
    x = np.arange(0.0, delx, .01)

    theta1 = np.arctan((delx + sqrt(delx**2 - 2*(g*delx**2 / v0**2)*(g*delx**2/(2.0*v0**2) + dely)))/(g*delx**2/v0**2))
    theta2 = np.arctan((delx - sqrt(delx**2 - 2*(g*delx**2 / v0**2)*(g*delx**2/(2.0*v0**2) + dely)))/(g*delx**2/v0**2))

    theta = 0
    if sol == 1:
        theta = theta1
    elif sol == 2:
        theta = theta2

    y = math.tan(theta)*x - 0.5*g*x**2/(v0**2*math.cos(theta)**2)

    draw_graph(x,y,min(x),max(x),1)

sols_list = [1,2]
vel = 10
for s in sols_list:
    draw_trajectory(vel,s)

plt.legend(['solution 1','solution 2'])

vel_list = [10,15,20]
for v in vel_list:
    draw_trajectory(v,1)

plt.legend(['init velocity = 10 m/s','init velocity = 15 m/s','init velocity = 20 m/s'])
plt.savefig('./trajectory.pdf',bbox_inches='tight')
