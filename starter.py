import sys

import pylab as plb
import numpy as np
import mountaincar


class DummyAgent():
    """ A not-so-good agent for the mountain-car task. """

    def __init__(self, mountain_car=None, parameter1=3.0):
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

    def visualize_trial(self, n_steps=200):
        """
        Do a trial without learning, with display on.

        Parameters
        ----------
        n_steps : int
            Number of steps in the simulation
        """

        # Prepare for visualization:
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        self.mountain_car.reset()  # Make sure the mountain-car is reset

        for n in range(n_steps):
            # Log:
            print '\rt =', self.mountain_car.t,
            sys.stdout.flush()

            # Choose random action:
            self.mountain_car.apply_force(np.random.randint(3) - 1)

            # Simulate timesteps:
            self.mountain_car.simulate_timesteps(100, 0.01)

            # Ipdate visualization:
            mv.update_figure()
            plb.draw()

            # Check for rewards
            if self.mountain_car.R > 0.0:
                print "\rreward obtained at t = ", self.mountain_car.t
                break

    def learn(self):
        # TODO
        pass

if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
