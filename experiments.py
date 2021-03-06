"""
Experiment module for reproducing the figures in the report.

Usage (terminal):
    Set eligibility trace rate to zero:
    $ python experiments.py el_tr_rate 0

    Set intitial exploration temperature to 1000 and decrease it linearly
    with time:
    $ python experiments.py temp 1 --temp_fun

Notes
-----
Whenever a parameter is set via terminal, the remaining parameters stay on
their default value.

To set temp to infinity, just call the tep experiment with a value greater than
1e6, e.g.,
    $ python experiments.py temp 10000000

"""

import starter as st
import numpy as np
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Experiments on the agent")
    parser.add_argument("exp_type", choices=['temp', 'el_tr_rate', 'w'],
                        help='parameter upon which to experiment')
    parser.add_argument("val", type=float, help='value of the parameter')
    parser.add_argument("-tf", "--temp_fun", action='store_true',
                        help='decrease exploration temperature with time')
    args = parser.parse_args()

    # Set figure name
    figname = args.exp_type + '=' + str(args.val)

    # Deal with optional argument
    if args.temp_fun:
        temp_fun = st.lin_temp_decay
        figname += 'with_temp_decay'
    else:
        temp_fun = None

    if args.exp_type == 'temp':  # Experiment on exploration temperature
        val = np.inf if args.val > 1e6 else args.val
        learning_curves = st.batch_agents(temp=val)

    if args.exp_type == 'el_tr_rate':  # Experiment on eligibility trace decay
        learning_curves = st.batch_agents(el_tr_rate=args.val)

    if args.exp_type == 'w':  # Experiment on weight initialization
        learning_curves = st.batch_agents(W=(args.val * np.ones((3, 20 * 20))))

    # learning_curves = np.ones((10, 100))

    fig, _ = st.plot_learning_curves(learning_curves)
    fig.savefig(figname + '.pdf', bbox_inches='tight')
