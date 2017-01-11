import sys

import pylab as plb
import numpy as np
import mountaincar


class Network():

    def __init__(self, grid_shape=[20, 20], W=None):
        self.nx, self.nx_d = grid_shape
        if W is None:
            self.W = np.random.uniform(size=(self.nx, self.nx_d, 3))
        else:
            r, c, d = W.shape
            assert r == self.nx and c == self.nx_d and d == 3, \
                "Incompatible W shape"
            self.W = W

        x = np.linspace(-150, 30, self.nx)
        self.sigma_x = abs(x[0] - x[1])

        x_d = np.linspace(-15, 15, self.nx_d)
        self.sigma_x_d = abs(x_d[0] - x_d[1])

        self.X, self.X_d = np.meshgrid(x, x_d)

    def activation(self, x, x_d):
        # Activity of input neurons (r.shape = grid_shape):
        r = np.exp(-((self.X - x) / self.sigma_x)**2) * \
            np.exp(-((self.X_d - x_d) / self.sigma_x_d)**2)

        # Activity of output neurons (q.shape = (3,)):
        Q = np.sum(self.W * r, axis=(1, 2))

        return Q, r


class Agent():

    def __init__(self, mc=None, net=None, temp=1, learn_rate=0.01,
                 reward_factor=0.95, el_tr_rate=0.5, temp_fun=None):
        self.mc = mountaincar.MountainCar() if mc is None else mc
        self.net = Network() if net is None else net
        self.temp = temp
        self.learn_rate = learn_rate
        self.reward_factor = reward_factor
        self.el_tr_rate = el_tr_rate
        self.temp_fun = temp_fun

    def learn(self, n_trials=100, n_steps=200, verbose=0):
        learning_curve = np.zeros(n_trials)

        for i in range(n_trials):

            if verbose:
                # Prepare for visualization:
                plb.ion()
                mv = mountaincar.MountainCarViewer(self.mc)
                mv.create_figure(n_steps, n_steps)
                plb.draw()

            self.mc.reset()
            el_tr = np.zeros(self.net.W.shape)  # eligibility traces
            a, q, _ = self.choose_action()

            for j in range(n_steps):
                # Update exploration temperature
                if self.temp_fun is not None:
                    self.temp = self.temp_fun(self.temp, j)

                # Simulate timesteps
                self.mc.simulate_timesteps(n=100, dt=0.01)

                if verbose:
                    print('\rt = {}'.format(self.mc.t))
                    sys.stdout.flush()
                    mv.update_figure()
                    plb.draw()

                # Check for rewards
                if self.mc.R > 0.0:
                    if verbose:
                        print("\rGot reward at t = {}".format(self.mc.t))
                    learning_curve[i] = j
                    break

                # Choose action
                a_prime, q_prime, r_prime = self.choose_action()

                # Calculate TD error
                delta = self.mc.R - (q - self.reward_factor * q_prime)

                # Update eligibility trace
                el_tr *= (self.reward_factor * self.el_tr_rate)
                el_tr[:, :, a_prime] += r_prime

                # Update network weights
                self.net.W += self.learn_rate + delta * el_tr

                q = q_prime

        return learning_curve

    def choose_action(self):

        # Compute action probabilities
        Q, r = self.net.activation(self.mc.x, self.mc.x_d)
        p = np.exp(Q / self.temp)  # action probabilities
        p = p / np.sum(p)
        cdf = np.cumsum(p)

        # Take action
        u = np.random.uniform()  # random number for action choice
        if u <= cdf[0]:
            self.mc.apply_force(-1)  # apply force to the left
            action = 0
        elif u > cdf[0] and u <= cdf[1]:
            self.mc.apply_force(0)  # apply no force
            action = 1
        else:
            self.mc.apply_force(1)  # apply force to the right
            action = 2

        return action, Q[action], r


if __name__ == "__main__":
    a = Agent()
    a.learn(verbose=1)
    plb.show()
