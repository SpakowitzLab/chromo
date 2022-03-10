# Chromo

Physics-based simulator of chromatin architecture.

## Authors
Joseph Wakim, Bruno Beltran, Andrew Spakowitz

## Abstract
Chromatin organization is essential for differential gene expression in humans, enabling a diverse array of cell types to develop from the same genetic code. The architecture of chromatin is dictated by patterns of epigenetic marks–chemical modifications to DNA and histones–which must be reestablished during each cell replication cycle. Dysregulation of chromatin organization has drastic medical consequences and can contribute to aging, obesity, and cancer progression. During this study, we develop a Monte Carlo simulator called “Chromo” to investigate factors affecting chromatin organization. The simulator proposes and evaluates random configurational changes to chromatin, which include geometric transformations and effector protein binding/unbinding to the biopolymer. By iteratively sampling configurations, Chromo generates snapshots of energetically favorable chromatin architectures. Unlike previous models fit to experimental data, Chromo is rooted in fundamental polymer theory, allowing us to predict biophysical mechanisms governing chromatin organization. By leveraging a computationally efficient field theoretic approach, we can simulate full human chromosomes with nucleosome-scale resolution. To confirm the validity of our simulator, we reproduce theoretical chain statistics for flexible and semiflexible homopolymers and recapitulate heterochromatin compartments expected in chromatin contact maps. Using Chromo, we will evaluate how the quality of structural prediction changes with the simulation of additional epigenetic marks. We will also identify which combinations of marks best explain the architecture of the chromosome.

## Quickstart
Setup is completed in the terminal from the root directory of this package (containing this file). This simulator is ready for use in two simple steps:

1. **Install dependencies.** The python packages required by this simulator are listed in `requirements.txt`.
   1. `python -m pip install --upgrade pip`
   2. `pip install -r requirements.txt`

2. **Compile cython modules.** This codebase uses cython to improve runtime.
   1. python setup.py build_ext --inplace`

## Code Structure

We use an object-oriented approach to organize components of our model. Components of the system are defined in a hierarchical manner, supporting efficiency and modularity. The class structure of the codebase is illustrated by the diagram below. The accompanying sphinx documentation includes descriptions of each class and their attributes.

<img src="doc/source/figures/chromo_UML.png">

## Examples
We've prepared example simulations for a variety of polymer modeling scenarios, which are presented as jupyter notebooks at the `doc/examples` subdirectory.
