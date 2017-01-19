"""
Learning module for the mountain-car problem.
"""

import sys

import pylab as plb
import numpy as np
import mountaincar
import matplotlib.pyplot as plt


class Network():
    """
    Neural network used for approximating the Q-values.

    Attributes
    ----------
    grid_shape : array_like
        Dimensions of the grid of neurons in the input layer.
    W : array
        A #(actions)-by-#(input neurons) matrix with synaptic weights.

    Methods
    -------
    activation(x, x_d)
        Returns the activations of each layer in response to state s=(x, x_d).
        r : #(input neurons)-by-1 array
            Activity in the input layer
        Q : 3-by-1 array
            Activity in the output layer

    """

    def __init__(self, grid_shape=(20, 20), W=None):
        self.nx, self.nx_d = grid_shape
        if W is None:
            self.W = np.random.uniform(size=(3, self.nx * self.nx_d))
            # self.W = np.zeros((3, self.nx * self.nx_d))
            # self.W = np.ones((3, self.nx * self.nx_d))
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
        r = np.exp(-((self.X - x) / self.sigma_x)**2 -
                   ((self.X_d - x_d) / self.sigma_x_d)**2)
        r = np.reshape(r, self.nx * self.nx_d, 1)

        # Activity of output neurons
        Q = self.W @ r

        return Q, r


class Agent():
    """
    Agent class implementing SARSA(lambda) for the mountain-car problem.

    Parameters
    ----------
    mc : mountaincar.MountainCar
        The mountain-car object implemented in the mountaincar module.
    net : Network
        The network used to approximate the Q-values in SARSA(lambda).
    temp : float
        Exploration temperature parameter for the policy probabilities.
    learn_rate : float
        Learning rate for the weights of the neural net.
    reward_factor : float
        Reward factor in SARSA(lambda).
    el_tr_rate : float
        Eligibility trace decay rate in SARSA(lambda).
    temp_fun : callable
        Function to change 'temp' at each iteration of each trial.

    Methods
    -------
    learn(n_trials, n_steps, verbose)
        Runs 'n_trials' of SARSA(lambda) with at most 'n_steps' steps.
        Returns a learning rate vector.
        If verbose, the learning procedure will be logged at each step.
    choose_action()
        Chooses next action to take (force to apply) based on the current
        state-action pair.
        Returns the action, as well as the neural net activations that led
        to it.

    """

    def __init__(self, mc=None, net=None, temp=None, learn_rate=1e-2,
                 reward_factor=0.95, el_tr_rate=None, temp_fun=None):
        self.mc = mountaincar.MountainCar() if mc is None else mc
        self.net = Network() if net is None else net
        self.temp0 = 0.1 if temp is None else temp
        self.learn_rate = learn_rate
        self.reward_factor = reward_factor
        self.el_tr_rate = 0.95 if el_tr_rate is None else el_tr_rate
        self.temp_fun = temp_fun

    def learn(self, n_trials=100, n_steps=10000, verbose=0):
        learning_curve = n_steps * np.ones(n_trials)

        for i in range(n_trials):
            if verbose:
                # Prepare for visualization
                plb.ion()
                mv = mountaincar.MountainCarViewer(self.mc)
                mv.create_figure(n_steps, n_steps)
                plb.draw()

            # Initialization for new trial
            self.mc.reset()
            self.temp = self.temp0
            el_tr = np.zeros(self.net.W.shape)  # eligibility traces
            a, q, r = self.choose_action()

            # Update exploration temperature
            if self.temp_fun is not None:
                self.temp = self.temp_fun(self.temp0, 0, i, n_trials)

            for j in range(n_steps):

                # Update eligibility traces
                el_tr *= (self.reward_factor * self.el_tr_rate)
                el_tr[a, :] += r

                # Simulate timesteps
                self.mc.simulate_timesteps(n=100, dt=0.01)

                # Choose next action
                a_prime, q_prime, r_prime = self.choose_action()

                # Calculate TD error
                delta = self.mc.R + (self.reward_factor * q_prime) - q

                # Update network weights
                deltaW = self.learn_rate * delta * el_tr
                self.net.W += deltaW

                # Log
                if verbose:
                    mv.update_figure()
                    plb.draw()
                    print("tau = {}".format(self.temp))
                    print("a_prime = {}".format(a_prime))
                    print("q_prime = {}".format(q_prime))
                    print("delta = {}".format(delta))
                    print("||deltaW|| = {}".format(np.linalg.norm(deltaW)))
                    print("max(|deltaW|) = {}".format(np.max(np.abs(deltaW))))
                    sys.stdout.flush()

                # Change old varibles for new ones
                a = a_prime
                q = q_prime
                r = r_prime

                # Check for rewards
                if self.mc.R > 0.0:
                    if verbose:
                        print("\rGot reward at t = {}".format(self.mc.t))
                        sys.stdout.flush()
                    learning_curve[i] = j
                    break

            if verbose:
                input("Press ENTER to continue...")
                sys.stdout.flush()

        return learning_curve

    def choose_action(self):
        # Compute network activations
        Q, r = self.net.activation(self.mc.x, self.mc.x_d)

        if self.temp == 0:  # Greedy policy
            action = np.argmax(Q)
        else:  # Compute action probabilities
            if self.temp == np.inf:
                p = np.ones(Q.shape)
            else:
                p = np.exp(np.clip(Q / self.temp, -500, 500))

            p = p / np.sum(p)
            cmf = np.cumsum(p)  # cumulative mass function

            # Take action
            u = np.random.uniform()  # random number for action choice
            action = np.argmax(u <= cmf)

        force = action - 1
        self.mc.apply_force(force)

        return action, Q[action], r


def plot_q_values(agent, f=None):
    """ Plot Q-values of the agent on state space, for each action. """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
    fig = plt.figure(figsize=(9, 9)) if f is None else f

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax.plot_surface(X, X_d, Q1, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_title('Action 1 (force to the left)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    fig.colorbar(surf1, shrink=0.5, aspect=10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax.plot_surface(X, X_d, Q2, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_title('Action 2 (no force)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    fig.colorbar(surf2, shrink=0.5, aspect=10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax.plot_surface(X, X_d, Q3, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_title('Action 3 (force to the right)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    fig.colorbar(surf3, shrink=0.5, aspect=10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return fig, ax


def plot_vector_field(agent):
    """
    Vector field of most likely actions, overlaid on the energy landscape

    Notes
    -----
    Got the idea from
    stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields

    """
    x = np.linspace(-160, 40, 200)
    y = np.linspace(-20, 20, 60)
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

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(E, extent=[X.min(), X.max(), X_d.min(), X_d.max()])
    ax.quiver(X[::10, ::10], X_d[::10, ::10], DX[::10, ::10],
              DY[::10, ::10], width=0.0025, scale=0.25, scale_units='x')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax.set(aspect=1, title='Directions with highest Q-value')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('dx/dt [m/s]')
    return fig, ax


def plot_learning_curves(lc, fig=None):
    """ Plot average and standard deviation of the input curves. """
    lc_mean = np.mean(lc, axis=0)
    lc_std = np.std(lc, axis=0)
    trials = np.array(range(len(lc_mean))) + 1

    if fig is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = fig.axes()
    ax.set_title("Learning curve")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Excape latency")
    ax.grid()
    ax.fill_between(trials, lc_mean - lc_std, lc_mean + lc_std, alpha=0.1,
                    color="g")
    ax.plot(trials, lc_mean, 'o-', color="g")

    return fig, ax


def plot_weights(agent, f=None):
    """ Plot NN weights corresponding to each action """
    nx_d = agent.net.nx_d
    nx = agent.net.nx

    W_backward = agent.net.W[0, :].reshape(nx_d, nx)
    W_neutral = agent.net.W[1, :].reshape(nx_d, nx)
    W_forward = agent.net.W[2, :].reshape(nx_d, nx)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(18, 6)) if f is None else f

    i = 0
    for choice in ['backward', 'neutral', 'forward']:
        i += 1
        ax = fig.add_subplot(1, 3, i)
        im = ax.imshow(eval('W_' + choice))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set(aspect=1, title=('W for ' + choice + ' action'))

    return fig, ax


def exp_temp_decay(t_0, t_end, curr_step, n_steps):
    r"""
    Exponentially decrease the exploration temperature.

    Parameters
    ----------
    t_0: float
        Initial exploration temperature
    t_end : float
        Final exploration temperature
    curr_step : int
        Current step number
    curr_step : int
        Maximum number of steps

    Returns
    -------
    t : float
        Current exploration temperature

    """
    if t_0 == np.inf:
        return t_0
    else:
        epsilon = min(1e-3, (1 / t_0))
        return t_0 * ((t_end + epsilon) / t_0) ** (curr_step / n_steps)


def lin_temp_decay(t_0, t_end, curr_step, n_steps):
    r"""
    Linearly decrease the exploration temperature.

    Parameters
    ----------
    t_0: float
        Initial exploration temperature
    t_end : float
        Final exploration temperature
    curr_step : int
        Current step number
    curr_step : int
        Maximum number of steps

    Returns
    -------
    t : float
        Current exploration temperature

    """
    if n_steps < 2:
        return t_0
    else:
        return t_0 + curr_step * ((t_end - t_0) / (n_steps - 1))


def job(n_agents, n_trials, n_steps, temp, temp_fun, el_tr_rate, W):
    """ Batch job for the function batch_agents.

    Notes
    -----
    This has to be done on the top level of the module for
    multiprocessing.Pool() to work properly.

    """
    net = Network(W=W)
    agent = Agent(net=net, temp=temp, el_tr_rate=el_tr_rate,
                  temp_fun=temp_fun)
    return agent.learn(n_trials=n_trials, n_steps=n_steps)


def batch_agents(n_agents=16, n_trials=100, n_steps=10000, temp=None,
                 temp_fun=None, el_tr_rate=None, W=None):
    """
    Train in parallel a number of agents and record their learning curves.

    Parameters
    ----------
    n_agents : int
        Number of agents to train.

    """
    from multiprocessing import Pool

    # Run jobs in parallel
    p = Pool(4)
    args = (n_agents, n_trials, n_steps, temp, temp_fun, el_tr_rate, W)
    results = [p.apply_async(job, args) for a in range(n_agents)]

    # Record learning curves
    learning_curves = n_steps * np.ones((n_agents, n_trials))
    for i in range(len(results)):
        learning_curves[i, :] = results[i].get()

    return learning_curves


if __name__ == "__main__":
    """ Interactive visualization of a couple of learning trials """
    agent = Agent()

    plt.ion()

    # Most likely actions (initially)
    plot_vector_field(agent)
    input("Press ENTER to continue...")
    sys.stdout.flush()

    agent.learn(n_trials=15, verbose=1)

    # Most likely actions (after training)
    plot_vector_field(agent)
    input("Press ENTER to continue...")
    sys.stdout.flush()

    plb.show()
