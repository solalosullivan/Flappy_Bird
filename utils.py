import numpy as np

import matplotlib.pyplot as plt
from matplotlib import axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


def plot_state_value(environment, agent, title="State Value", size=(14, 14)):
    # Define x, y, and num_rows based on environment observation space
    X = np.arange(environment.observation_space[0].n)
    Y = np.arange(
        -environment.observation_space[1].n // 2,
        environment.observation_space[1].n // 2 + 1,
    )
    nb_rows = environment.observation_space[1].n + 1

    # Plot
    X, Y = np.meshgrid(X, Y)
    state_values = [max(agent.q[(u, v)]) for u, v in zip(np.ravel(X), np.ravel(Y))]
    Z = np.reshape(np.array(state_values), X.shape)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.PRGn, vmin=0)

    # Set the labels and title
    ax.set_xlabel("X distance")
    ax.set_ylabel("Y distance")
    ax.set_zlabel("State Value")
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    plt.show()


def plot_policy(environment, agent, title="Plot policy", size=(8, 8)):

    X = np.arange(environment.observation_space[0].n)
    Y = np.arange(
        -environment.observation_space[1].n // 2,
        environment.observation_space[1].n // 2 + 1,
    )

    X, Y = np.meshgrid(X, Y)

    # Get the maximum action value for each state
    Z = np.argmax(agent.q[(X, Y)], axis=2)

    fig, ax = plt.subplots(figsize=size)
    display = ax.imshow(
        Z, cmap=plt.get_cmap("PRGn", 2), vmin=0, vmax=1, extent=[0, 14, -11, 11]
    )

    plt.xticks(X)
    plt.yticks(Y)
    plt.gca().invert_yaxis()
    ax.set_xlabel("X distance")
    ax.set_ylabel("Y distance")
    ax.set_title(title)

    ax.grid(color="w", linestyle="-", linewidth=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(display, ticks=[0, 1], cax=cax)
    cbar.ax.set_yticklabels(["0 (idle)", "1 (flap)"])

    plt.show()


def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value, then update top and reset ties to zero
        # if a value is equal to top value, then add the index to ties
        if q_values[i] > top:
            top, ties = q_values[i], [i]
        elif q_values[i] == top:
            ties.append(i)

    # return a random selection from ties
    ind = np.random.choice(ties)

    return ind


def get_state(obs):
    return obs[0][0], obs[0][1]
