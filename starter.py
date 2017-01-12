import sys

import pylab as plb
import numpy as np
import mountaincar


class Network():

    def __init__(self, grid_shape=[90, 15], W=None):
        self.nx, self.nx_d = grid_shape
        if W is None:
            self.W = np.random.uniform(size=(3, self.nx * self.nx_d))
        else:
            r, c = W.shape
            assert r == 3 and c == (self.nx * self.nx_d), \
                "Incompatible W shape"
            self.W = W

        # Build grid
        x, self.sigma_x = np.linspace(-150, 30, self.nx, retstep=True)
        x_d, self.sigma_x_d = np.linspace(-15, 15, self.nx_d, retstep=True)
        self.X, self.X_d = np.meshgrid(x, x_d)

    def activation(self, x, x_d):
        # Activity of input neurons
        r = np.exp(-((self.X - x) / self.sigma_x)**2) * \
            np.exp(-((self.X_d - x_d) / self.sigma_x_d)**2)
        r = np.reshape(r, self.nx * self.nx_d, 1)

        # Activity of output neurons
        Q = self.W @ r

        return Q, r


class Agent():

    def __init__(self, mc=None, net=None, temp=1, learn_rate=1e-4,
                 reward_factor=0.95, el_tr_rate=0.5, temp_fun=None):
        self.mc = mountaincar.MountainCar() if mc is None else mc
        self.net = Network() if net is None else net
        self.temp = temp
        self.learn_rate = learn_rate
        self.reward_factor = reward_factor
        self.el_tr_rate = el_tr_rate
        self.temp_fun = temp_fun

    def learn(self, n_trials=100, n_steps=200, verbose=0):
        """ Learn to climb the hill with the SARSA(lambda) algorithm. """
        learning_curve = n_steps * np.ones(n_trials)

        for i in range(n_trials):

            if verbose:
                # Prepare for visualization:
                plb.ion()
                mv = mountaincar.MountainCarViewer(self.mc)
                mv.create_figure(n_steps, n_steps)
                plb.draw()

            # Initialization for new trial
            self.mc.reset()
            el_tr = np.zeros(self.net.W.shape)  # eligibility traces
            a, q, _ = self.choose_action()

            for j in range(n_steps):
                # Update exploration temperature
                if self.temp_fun is not None:
                    self.temp = self.temp_fun(self.temp, j)

                # Simulate timesteps
                self.mc.simulate_timesteps(n=100, dt=0.01)

                # Log
                if verbose:
                    mv.update_figure()
                    plb.draw()

                # Check for rewards
                if self.mc.R > 0.0:
                    if verbose:
                        print("\rGot reward at t = {}".format(self.mc.t))
                    learning_curve[i] = j
                    break

                # Choose next action
                a_prime, q_prime, r_prime = self.choose_action()

                # Calculate TD error
                delta = self.mc.R - (q - self.reward_factor * q_prime)

                # Update eligibility trace
                el_tr *= (self.reward_factor * self.el_tr_rate)
                el_tr[a_prime, :] += r_prime

                # Update network weights
                self.net.W -= self.learn_rate + delta * el_tr

                # Normalize the weights to avoid numerical overflow
                self.net.W /= np.max(self.net.W)

                # Change old varible for new one
                q = q_prime

            if verbose:
                input("Press ENTER to continue...")

        return learning_curve

    def choose_action(self):

        # Compute action probabilities
        Q, r = self.net.activation(self.mc.x, self.mc.x_d)
        p = np.exp(np.clip(Q / self.temp, -500, 500))  # action probabilities
        p = p / np.sum(p)
        cmf = np.cumsum(p)  # cumulative mass function

        # Take action
        u = np.random.uniform()  # random number for action choice
        action = np.argmax(u <= cmf)  # choose action by comparing with the cmf
        force = action - 1
        self.mc.apply_force(force)

        return action, Q[action], r


def plot_q_values(agent):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    x = np.linspace(-150, 30, 100)
    x_d = np.linspace(-15, 15, 100)
    X, X_d = np.meshgrid(x, x_d)
    Q1 = np.zeros(X.shape)
    Q2 = np.zeros(X.shape)
    Q3 = np.zeros(X.shape)

    # Populate Q matrices
    for i in range(len(x_d)):
        for j in range(len(x)):
            Q, _ = agent.net.activation(x[i], x_d[j])
            Q1[i, j] = Q[0]
            Q2[i, j] = Q[1]
            Q3[i, j] = Q[2]

    # Plot
    fig = plt.figure(figsize=(9, 9))

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax.plot_surface(X, X_d, Q1, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_title('Action 1 (force to the left)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    fig.colorbar(surf1, shrink=0.5, aspect=10)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax.plot_surface(X, X_d, Q2, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_title('Action 2 (no force)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    fig.colorbar(surf2, shrink=0.5, aspect=10)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax.plot_surface(X, X_d, Q3, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_title('Action 3 (force to the right)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    fig.colorbar(surf3, shrink=0.5, aspect=10)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.show()


def plot_vector_field(agent):
    """ Got the idea from
    stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields
    """
    x = np.linspace(-150, 30, 20)
    y = np.linspace(-15, 15, 6)
    X, X_d = np.meshgrid(x, y)
    DX = np.zeros(X.shape)
    DY = np.zeros(X.shape)
    E = agent.mc._energy(X, X_d)  # Energy in each point of the grid

    # Populate DX matrix
    for i in range(len(y)):
        for j in range(len(x)):
            Q, _ = agent.net.activation(x[j], y[i])
            if np.argmax(Q) == 0:
                DX[i, j] = -1
            if np.argmax(Q) == 2:
                DX[i, j] = 1

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(E, extent=[X.min(), X.max(), X_d.min(), X_d.max()])
    ax.quiver(X, X_d, DX, DY, width=0.0025, scale=0.25, scale_units='x')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax.set(aspect=1, title='Directions with highest Q-value')
    ax.set_xlabel('x [m/s]')
    ax.set_ylabel('dx/dt [m/s]')
    plt.show()


if __name__ == "__main__":
    agent = Agent(temp=1e2, el_tr_rate=0.0, learn_rate=0.01)
    agent.learn(n_trials=10, verbose=1)

    # plot_q_values(agent)
    # input("Press ENTER to continue...")

    plot_vector_field(agent)
    input("Press ENTER to continue...")

    plb.show()
