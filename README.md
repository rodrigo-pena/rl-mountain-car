# rl-mountain-car
Reinforcement learning solution with the [SARSA(\lambda)][1] algorithm for the mountain-car problem.

**Goal**: help an under-powered car find its way up a steep hill. Since the car has not enough power to directly climb the hill, it has to learn to swing back-and-forth to gain enough momentum to reach the summit.

**Implementation**: 2-layer neural network with reward-modulated plasticity.

## Requirements
Make sure you have the follwing installed:

* [matplotlib](http://matplotlib.org)
* [numpy](http://www.numpy.org)
* [sys](https://docs.python.org/3/library/sys.html)
* [argparse](https://docs.python.org/3/library/argparse.html)

Alternatively, you can simply run this command to install those dependencies:

```sh
pip install -r requirements.txt
```


## Running the code
The best way to get started is by running from the terminal the command

```sh
python starter.py
```

This will trigger an interactive view of the learning trials, using default parameters.
The vector field plots will show, before and after training, the direction of the most likely action at evenly-spaced points in the s = (x [m], dx/dt [m/s]) state space of the car. The vectors are overlaid on a contour plot of the total energy of the car as a function of its state. The remaining plots (one for each trial) will depict the trajectories the car took in the state space, the force directions it applied, as well as the total energy at each step of the trial.

Alternatively, one could check the jupyter notebook [experiments.ipynb](https://github.com/rodrigo-pena/rl-mountain-car/blob/master/experiments.ipynb) or the script [experyments.py](https://github.com/rodrigo-pena/rl-mountain-car/blob/master/experiments.py) for example of code usage and for reproduction of the figures in the report.

## Notes
Developed as a mini-project for the course CS-434 "Unsupervised and Reinforcement Learning in Neural Networks", Fall 2016, EPFL.

## References
1. Richard S. and Barto, Andrew G. Reinforcement Learning: An Intro- duction. MIT Press, 1998. ISBN 0262193981. [URL](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html).

[1]: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node77.html

