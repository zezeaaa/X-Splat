# Marks simple_knn as a regular package.
#
# Upstream ships this directory with only a .gitkeep, relying on `setup.py develop`
# putting the source root on sys.path. Modern pip installs editable projects through
# a PEP 660 import finder instead, which only resolves a mapped directory when it is
# a regular package - so without this file `import simple_knn._C` fails after an
# apparently successful install.
