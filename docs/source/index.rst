.. chromo documentation master file, created by
   sphinx-quickstart on Thu Sep 17 12:55:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Chromo: Physics-based Chromatin Simulator
=========================================
Joseph Wakim, Bruno Beltran, Andrew Spakowitz

|

Quickstart
----------

1. **Download Chromo Source Code.** The Chromo source code can be downloaded from the `Spakowitz Lab GitHub <https://github.com/SpakowitzLab>`_.

2. **Load Conda Environment.**
   Using `Anaconda <https://docs.anaconda.com>`_, load and activate the :code:`chromo` virtual environment from :code:`environment.yml`.

.. parsed-literal::

   $ conda env create -f environment.yml
   $ conda activate chromo


3. **Install Chromo Package.**
   We need to add Chromo to the site-packages of your Python installation.
   Enter the command from the root directory of Chromo.

.. parsed-literal::

   $ pip install -e .

4. **Compile Cython Modules.**
   This codebase uses cython to improve runtime.
   Use the command below from Chromo's root directory to compile the code.

.. parsed-literal::

   $ python setup.py build_ext --inplace

5. **Run an Example Simulation (Optional).**
   To run an example simulation, navigate to the :code:`doc/example` and execute one of the example notebooks.
   For more details behind executing a Jupyter notebook from the terminal, visit the `Nbconvert <https://nbconvert.readthedocs.io/en/latest/index.html#>`_ docs.

.. parsed-literal::

   $ cd doc/examples
   $ jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --output <Output_Notebook_Name> --execute <Example_Notebook_Name>

.. Tip:: To recompile the Sphinx documentation, you will also need to run :code:`conda install pandoc`.

|

Examples
--------

**Specify Reader Proteins.**
Default properties for HP1 and PRC1 are implemented; these binders can be called by name.
New reader proteins can be defined by specifying their physical properties.

.. parsed-literal::

   hp1 = chromo.binders.get_by_name('HP1')
   prc1 = chromo.binders.get_by_name('PRC1')
   custom_binder = ReaderProtein(
      name="new_binder",
      sites_per_bead=2,
      bind_energy_mod=-1,
      bind_energy_no_mod=1,
      interaction_energy=-0.2,
      chemical_potential=-1.5,
      interaction_radius=3
   )
   binders = chromo.binders.make_binder_collection([hp1, prc1, custom_binder])

|

**Specify Chemical Modifications & Initial Binding States.**
Each reader protein requires a pattern of histone modifications dictating binding.
These patterns can be specified directly or read from a file.
Initial reader protein binding states may be trivially defined to match the chemical modifications.

.. parsed-literal::

   H3K9me3 = Chromatin.load_seqs(["path/to/H3K9me3/sequence"])
   H3K27me3 = Chromatin.load_seqs(["path/to/H3K27me3/sequence"])
   custom_mod = np.zeros((len(H3K0me3))
   chemical_mods = np.column_stack((H3K9me3, H3K27me3, custom_mod))
   states = chemical_mods.copy()

|

**Define Polymer.**
Homopolymers can be instantiated with basic dimensions and an initial configuration.
For example, here we define a 1000-beads stretchable, shearable wormlike chain homopolymer, with beads spaced by 25-units and a persistence length of 100-units.
The polymer is initialized along a Gaussian random walk.

.. parsed-literal::

   num_beads = 1000
   bead_spacing = 25
   lp = 100
   polymer = SSWLC.gaussian_walk_polymer(
      "poly_1",
      num_beads,
      bead_spacing,
      lp=lp
   )

To instantiate chromatin, specify basic dimensions, histone modification patterns, and an initial configuration.
The polymer below is confined to a sphere with a 900-unit radius.
Each bead is modified at three sites, and initial reader proptein binding states match the initial modifications.

.. parsed-literal::

   chromatin = Chromatin.confined_gaussian_walk(
       'Chr-1',
       num_beads,
       bead_spacing,
       states=states,
       confine_type="Sphere",
       confine_length=900,
       binder_names=np.array(['HP1', 'PRC1', 'Custom']),
       chemical_mods=chemical_mods,
       chemical_mod_names=np.array(['H3K9me3', 'H3K27me3', 'custom_mod'])
   )

|

**Define Uniform Density Field.**
The density field for the polymer and its binders is instantiated as a grid of discrete voxels.
The width and number of voxels in each dimension of the field must be specified.

.. parsed-literal::

   x_width = 1000
   y_width = x_width
   z_width = x_width

   n_bins_x = 100
   n_bins_y = n_bins_x
   n_bins_z = n_bins_x

   udf = UniformDensityField(
       polymers = [polymer],
       binders = binders,
       x_width = x_width,
       nx = n_bins_x,
       y_width = y_width,
       ny = n_bins_y,
       z_width = z_width,
       nz = n_bins_z
   )

.. Tip:: See demos for setup, execution, and analysis of full simulations.

|

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction

   Abstract <abstract>

   Code Overview <code_structure>

   Monte Carlo Simulations <mc_sim>


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Theory

   Polymer Models <poly_models>

   Field Theoretic Treatment of Interactions <field>


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   Semiflexible, Unconfined Homopolymer <semiflexible_homopolymer_OUT.ipynb>

   Flexible, Confined Homopolymer <flexible_confined_homopolymer_OUT.ipynb>

   Chromatin + HP1 <one_mark_chromatin_OUT.ipynb>

   Chromatin + HP1 + PRC1 <two_mark_chromatin_OUT.ipynb>

   Ring Polymer <looped_loosely_confined_OUT.ipynb>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Additional Resources

   Modules <modules>

   References <references>


Indices and tables
==================

:ref:`genindex`

:ref:`modindex`
