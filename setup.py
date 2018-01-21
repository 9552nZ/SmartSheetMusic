from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

# np_lib = os.path.dirname(numpy.__file__)
# np_inc = [os.path.join(np_lib, 'core/include')]

# ext_modules = [Extension("dtw.fast",["src/fast.pyx"], include_dirs=np_inc)]

setup(description='Dynamic Time Warping',
      cmdclass = {'build_ext' : build_ext},
      include_dirs = [np.get_include()],
      ext_modules = [Extension("dtw_fast", ["dtw_fast.pyx"])]
      )