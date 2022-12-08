# Abstraction

## Steps to get running
First clone with submodules installed:

` git clone --recurse-submodules https://github.com/simonalford42/abstraction`

If you've already cloned, try

` git submodule update --init --recursive `

to get the submodules added.

Then install the requirements:

` python -m pip install -r requirements.txt`

Then you should be good to go.

If you want to install pycolab code do `cd lib/pycolab` and `python setup.py develop`. Using `develop` ensures that changes to the pycolab code take effect.
