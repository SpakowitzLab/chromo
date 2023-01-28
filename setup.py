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
            language="c++"
        ) for path in cython_paths
    ]

    setup(
        name="chromo",
        author="Spakowitz Lab (Joseph Wakim, Bruno Beltran, Andrew Spakowitz)",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        url="https://github.com/SpakowitzLab/chromo",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Development Status :: 1 - Pre-Alpha",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Cython",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Physics",
        ],
        packages=["chromo", "chromo.mc", "chromo.util"],
        install_requires=[
            "bioframe~=0.3.3",
            "Cython~=0.29.32",
            "IPython~=8.4.0",
            "karma_sphinx_theme~=0.0.8",
            "matplotlib~=3.5.3",
            "nbsphinx~=0.8.8",
            "numba~=0.56.0",
            "numpy~=1.22.4",
            "pandas~=1.4.3",
            "scikit_learn~=1.1.2",
            "scipy~=1.9.1",
            "setuptools~=61.2.0",
            "sphinx~=4.4.0",
            "sphinx_theme~=1.0",
            "sphinx_rtd_theme~=1.0.0"
        ],
        extras_require = {
            "dev": [
                "pytest~=6.0.1",
                "check-manifest~=0.48"
            ]
        },
        include_dirs=[
            np.get_include(), "chromo", "chromo/util", "chromo/mc",
            "analyses", "analyses/adsorption", "analyses/characterizations",
            "analyses/cmaps", "analyses/equilibration", "analyses/modifications"
        ],
        ext_modules=cythonize(
            extensions,
            annotate=True,
            compiler_directives={"cdivision": False}
        )
    )
