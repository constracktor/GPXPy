# N-linked Mass-spring-damper Simulator

This simulator models a non-linear system of coupled mass spring damper carts.
Non-linearity is introduced into the system by springs with non-linear force
profiles. Multisine or APRBS control signals apply force onto the first cart.
The output of the system resembles the position of the last cart. White noise
can be added to account for real-world measurement errors. The data is sampled
at a given frequency.

The simulator allows to generate training and test data for GPXPy.
Before running the simulator modify the `config.json` file to configure
the number of data samples to generate. More configuration parameters
are described in `comment.txt`.

The simulator requires a recent version of Python (>=3.2) and the packages 
numpy, scipy, and matplolib. The code can be run either directly with 
`python3 generate_msd_data.py` or in an specifically created virtual
environment with `./run_msd.sh`.
