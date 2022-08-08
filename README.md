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

Note to self: to reinstall pycolab after editing its contents, do `python -m pip uninstall pycolab` and then `cd lib/pycolab && python setup.py install`. To check if works open a shell and do `from pycolab.examples.research.box_world import box_world as bw`, and check that your changes worked.
