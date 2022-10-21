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

If you want to edit pycolab code, do `python -m pip uninstall pycolab` and then `cd lib/pycolab && python setup.py develop`. Edit: there was a way to install pycolab code in "edit mode" where changes to the pcyolab code automatically get updated. I forget what I did there, will update this next time I install somewhere.
