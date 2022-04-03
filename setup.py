import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import versioneer
import numpy as np


if __name__ == "__main__":

    if sys.version_info < (3, 5):
        raise ImportError("requires Python>=3.5")

    cython_paths = [
        "chromo/mc/move_funcs.pyx",
        "chromo/polymers.pyx",
        "chromo/fields.pyx",
        "chromo/binders.pyx",
        "chromo/util/bead_selection.pyx",
        "chromo/util/linalg.pyx",
        "chromo/mc/mc_sim.pyx",
        "chromo/mc/moves.pyx"
    ]

    extensions = [
        Extension(
            path.split(".")[0].replace("/", "."),
            sources=[path],
            language="c++",
            extra_compile_args=["-std=c++17"]
        ) for path in cython_paths
    ]

    setup(name="chromo",
          author="Spakowitz Lab",
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          url="https://github.com/SpakowitzLab/chromo",
          license="",
          long_description=open("README.md").read(),
          classifiers=["Intended Audience :: Science/Research",
                       'Intended Audience :: Developers',
                       'Development Status :: 1 - Pre-Alpha',
                       'License :: OSI Approved :: MIT License',
                       "Natural Language :: English",
                       "Programming Language :: Python :: 3 :: Only",
                       "Programming Language :: Fortran",
                       "Topic :: Scientific/Engineering :: Chemistry",
                       "Topic :: Scientific/Engineering :: Physics",
                       ],
          packages=find_packages(include=["chromo"]),
          install_requires=["numpy", "pandas"],
          include_dirs=[
                np.get_include(), "chromo", "chromo/util", "chromo/mc"
            ],
          ext_modules=cythonize(
              extensions,
              annotate=True,
              compiler_directives={"cdivision": False}
            )
          )
