"""
Mountain-car module for the reinforcement learning miniproject.
"""

import pylab as plb
import numpy as np


class MountainCar():
    """
    A mountain-car problem.

    For the miniproject, you are not meant to change the default parameters
    (mass of the car, etc.)

    Usage:
        >>> mc = MountainCar()

        Set the agent to apply a rightward force (positive in x)
        >>> mc.apply_force(+1) # Actual value doesn't mattter, only the sign

        Run an "agent time step" of 1s with 0.01 s integration time step
        >>> mc.simulate_timesteps(n = 100, dt = 0.01)

        Check the state variables of the agent, and the reward
        >>> print mc.x, mc.x_d, mc.R

        At some point, one might want to reset the position/speed of the car
        >>> mc.reset()
    """

    def __init__(self, g=10.0, d=100.0, H=10., m=10.0,
                 force_amplitude=3.0, reward_amplitude=1.,
                 reward_threshold=0.0):
        self.g = g  # Gravitational constant
        self.d = d  # Minima location
        self.H = H  # Height of the saddle point
        self.m = m  # Mass of the car
        self.force_amplitude = force_amplitude  # Force applied by engine
        self.reward_amplitude = reward_amplitude  # Value of the reward
        self.reward_threshold = reward_threshold  # Thresh. for getting reward
        self.reset()  # Reset car variables

    def reset(self):
        """ Reset the mountain car to a random initial position. """
        self.x = 80 * np.random.rand() - 130.0  # Set pos. to range [-130; -50]
        self.x_d = 10.0 * np.random.rand() - 5.0  # Set x_dot to range [-5; 5]
        self.R = 0.0  # Reset reward
        self.t = 0.0  # Reset time
        self.F = 0.0  # Reset applied force

    def apply_force(self, direction):
        """
        Apply a force to the car.

        Only three values are possible:
            right (if direction > 0),
            left (direction < 0) or
            no force (direction = 0).
        """
        self.F = np.sign(direction) * self.force_amplitude

    def _h(self, x):
        """ Landscape function h in x. """
        num = (x - self.d)**2 * (x + self.d)**2
        den = (self.d**4 / self.H) + x**2
        return num / den

    def _h_prime(self, x):
        """ First derivative of the landscape function h in x. """
        c = self.d**4 / self.H
        num = 2 * x * (x**2 - self.d**2) * (2 * c + self.d**2 + x**2)
        den = (c + x**2)**2
        return num / den

    def _h_second(self, x):
        """ Second derivative of the landscape function h in x. """
        c = self.d**4 / self.H
        num = - 2 * c**2 * (self.d**2 - 3 * x**2)
        num = num + c * (-self.d**4 + 6 * self.d**2 * x**2 + 3 * x**4)
        num = num + 3 * self.d**4 * x**2
        num = num + x**6
        num = 2 * num
        den = (c + x**2)**3
        return num / den

    def _energy(self, x, x_d):
        """
        Total energy of the car.

        Note
        ----
        v and x dot are not the same; v includes the y direction.

        """
        a = self.g * self._h(x)
        b = 0.5 * (1 + self._h_prime(x)**2) * x_d**2
        return self.m * (a + b)

    def simulate_timesteps(self, n=1, dt=0.1):
        """ Car dynamics for n timesteps of length dt. """
        for i in range(n):
            self._simulate_single_timestep(dt)
        self.t += n * dt
        self.R = self._get_reward()  # Check for rewards

    def _simulate_single_timestep(self, dt):
        """ Car dynamics for a single timestep. """

        # Second derivative of x (horizontal acceleration):
        a = np.arctan(self._h_prime(self.x))
        b = self.F / self.m
        c = self.g + self._h_second(self.x) * self.x_d**2
        x_dd = np.cos(a) * (b - np.sin(a) * c)

        # Update position and velocity:
        self.x += self.x_d * dt + 0.5 * x_dd * dt**2
        self.x_d += x_dd * dt

    def _get_reward(self):
        """ Check for, and return, reward. """

        if self.R > 0.0:  # If there's already a reward, we stick to it
            return self.R

        if self.x >= self.reward_threshold:  # Have we crossed the threshold?
            return self.reward_amplitude

        return 0.0  # Otherwise, no reward


class MountainCarViewer():
    """
    Display the state of a MountainCar instance.

    Usage:
        >>> mc = MountainCar()

        >>> mv = MoutainCarViewer(mc)

        Turn matplotlib's "interactive mode" on and create figure
        >>> plb.ion()
        >>> mv.create_figure(n_steps = 200, max_time = 200)

        This forces matplotlib to draw the fig. before the end of execution
        >>> plb.draw()

        Simulate the MountainCar, visualizing the state
        >>> for n in range(200):
        >>>     mc.simulate_timesteps(100,0.01)
        >>>     mv.update_figure()
        >>>     plb.draw()
    """

    def __init__(self, mountain_car):
        assert isinstance(mountain_car, MountainCar), \
            'Argument to MoutainCarViewer() must be a MountainCar instance.'
        self.mountain_car = mountain_car

    def create_figure(self, n_steps, max_time, f=None):
        """
        Create a figure showing the progression of the car.

        Call update_car_state subsequently to update this figure.

        Parameters
        ----------
        n_steps : int
            Number of times update_car_state will be called.
        max_time: float
            Time that the trial will last (to scale the plots).
        f : pylab.figure
            (Optional) figure in which to create the plots.
        """

        self.f = plb.figure() if f is None else f

        # Create attributes to store the arrays
        self.times = np.zeros(n_steps + 1)
        self.positions = np.zeros((n_steps + 1, 2))
        self.forces = np.zeros(n_steps + 1)
        self.energies = np.zeros(n_steps + 1)

        # Fill in initial values
        self.i = 0
        self._get_values()

        # Create the energy landscape plot
        self.ax_position = plb.subplot(2, 1, 1)
        self._plot_energy_landscape(self.ax_position)
        self.h_position = self._plot_positions()

        # Create the force plot
        self.ax_forces = plb.subplot(2, 2, 3)
        self.h_forces = self._plot_forces()
        plb.axis(xmin=0, xmax=max_time,
                 ymin=-1.1 * self.mountain_car.force_amplitude,
                 ymax=1.1 * self.mountain_car.force_amplitude)

        # Create the energy plot
        self.ax_energies = plb.subplot(2, 2, 4)
        self.h_energies = self._plot_energy()
        plb.axis(xmin=0, xmax=max_time,
                 ymin=0.0, ymax=1000.)

    def update_figure(self):
        """
        Update the figure.

        Notes
        -----
        Assumes the figure has already been created with create_figure().
        """

        # Increment
        self.i += 1
        assert self.i < len(self.forces), \
            "update_figure() was called too many times."

        # Get new values from the car
        self._get_values()

        # Update plots
        self._plot_positions(self.h_position)
        self._plot_forces(self.h_forces)
        self._plot_energy(self.h_energies)

    def _get_values(self):
        """ Retrieve the relevant car variables for the figure. """
        self.times[self.i] = self.mountain_car.t
        self.positions[self.i, 0] = self.mountain_car.x
        self.positions[self.i, 1] = self.mountain_car.x_d
        self.forces[self.i] = self.mountain_car.F
        self.energies[self.i] = self.mountain_car._energy(
            self.mountain_car.x, self.mountain_car.x_d)

    def _plot_energy_landscape(self, ax=None):
        """ Plot energy landscape for the mountain car in 2D.

        Parameters
        ----------
        ax : axes instance
            (Optional) Axes of the plot

        Returns
        -------
        ax : axes instance
            Axes of the plot

        Notes
        -----
        Use plot_energy_landscape to let the module decide whether you have
        the right modules for 3D plotting.
        """
        # Grid coordinates in the x-x_dot space
        X = np.linspace(-160, 160, 61)
        XD = np.linspace(-20, 20, 51)
        X, XD = np.meshgrid(X, XD)

        # Energy in each point of the grid
        E = self.mountain_car._energy(X, XD)

        # Display energy grid as an image
        if ax is None:
            plb.figure()
            ax = plb.axes()
        C = ax.contourf(X, XD, E, 100)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\dot x$')
        cbar = plb.colorbar(C)
        cbar.set_label('$E$')

        return ax

    def _plot_positions(self, handles=None):
        """ Plot position and trajectory of the car in state space. """

        # Choose point color according to force direction:
        color = ['r', 'w', 'g'][1 + int(np.sign(self.mountain_car.F))]

        if handles is None:  # Create plots
            handles = []  # List to keep plot objects
            handles.append(plb.plot(
                np.atleast_1d(self.positions[:self.i + 1, 0]),
                np.atleast_1d(self.positions[:self.i + 1, 1]),
                ',k'
            )[0])
            handles.append(plb.plot(
                np.atleast_1d(self.positions[self.i, 0]),
                np.atleast_1d(self.positions[self.i, 1]),
                'o' + color,
                markeredgecolor='none',
                markersize=9,
            )[0])
            return tuple(handles)
        else:  # Update plots
            handles[0].set_xdata(np.atleast_1d(self.positions[:self.i + 1, 0]))
            handles[0].set_ydata(np.atleast_1d(self.positions[:self.i + 1, 1]))
            handles[1].set_xdata(np.atleast_1d(self.positions[self.i, 0]))
            handles[1].set_ydata(np.atleast_1d(self.positions[self.i, 1]))
            handles[1].set_color(color)
            return handles

    def _plot_forces(self, handle=None):
        """ Plot force applied by the car vs. time. """
        if handle is None:  # Create plots
            handle = plb.plot(
                np.atleast_1d(self.times[:self.i + 1]),
                np.atleast_1d(self.forces[:self.i + 1]),
                ',k',
            )[0]
            plb.xlabel('$t$')
            plb.ylabel('$F$')
            return handle
        else:  # Update plots
            handle.set_xdata(np.atleast_1d(self.times[:self.i + 1]))
            handle.set_ydata(np.atleast_1d(self.forces[:self.i + 1]))
            return handle

    def _plot_energy(self, handle=None):
        """ Plot energy of the car vs. time. """
        if handle is None:  # Create plots
            handle = plb.plot(
                np.atleast_1d(self.times[:self.i + 1]),
                np.atleast_1d(self.energies[:self.i + 1]),
                'k',
                linewidth=0.5
            )[0]
            plb.xlabel('$t$')
            plb.ylabel('$E$')
            return handle
        else:  # Update plots
            handle.set_xdata(np.atleast_1d(self.times[:self.i + 1]))
            handle.set_ydata(np.atleast_1d(self.energies[:self.i + 1]))
            return handle
