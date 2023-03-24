# Chromo

Physics-based simulator of chromatin architecture.

## Authors
Joseph Wakim, Bruno Beltran, Angelika Hirsch, Andrew Spakowitz

## Quickstart

The physics-based chromatin simulator is available in the [“Chromo” repository](https://github.com/AngelikaH123/chromo) of the Spakowitz Lab GitHub account. Clone this repository to your local machine.

`git clone git@github.com:AngelikaH123/chromo.git`

Install the package dependencies and environment of the simulator by running the command:

'bash make_all.sh'

During installation, all Cython code required by the Monte Carlo algorithm will be compiled. This may take several
minutes. Once the package has been installed, the simulator is ready for use. Navigate to `chromo` from the root directory to find the run_simulation.py file for initializing the polymer object. Use the following command to run the file:

'python run_simulation.py'

This will generate the polymer object and compare it to a reference image, so the user can assess if the generated polymer is sufficiently similar to the control. 


## Abstract
Chromatin organization is essential for differential gene expression in humans, enabling a diverse array of cell types to develop from the same genetic code. The architecture of chromatin is dictated by patterns of epigenetic marks–chemical modifications to DNA and histones–which must be reestablished during each cell replication cycle. Dysregulation of chromatin organization has drastic medical consequences and can contribute to aging, obesity, and cancer progression. During this study, we develop a Monte Carlo simulator called “Chromo” to investigate factors affecting chromatin organization. The simulator proposes and evaluates random configurational changes to chromatin, which include geometric transformations and effector protein binding/unbinding to the biopolymer. By iteratively sampling configurations, Chromo generates snapshots of energetically favorable chromatin architectures. Unlike previous models fit to experimental data, Chromo is rooted in fundamental polymer theory, allowing us to predict biophysical mechanisms governing chromatin organization. By leveraging a computationally efficient field theoretic approach, we can simulate full human chromosomes with nucleosome-scale resolution. To confirm the validity of our simulator, we reproduce theoretical chain statistics for flexible and semiflexible homopolymers and recapitulate heterochromatin compartments expected in chromatin contact maps. Using Chromo, we will evaluate how the quality of structural prediction changes with the simulation of additional epigenetic marks. We will also identify which combinations of marks best explain the architecture of the chromosome.

## Code Structure

We use an object-oriented approach to organize components of our model. Components of the system are defined in a hierarchical manner, supporting efficiency and modularity. The class structure of the codebase is illustrated by the diagram below. The accompanying sphinx documentation includes descriptions of each class and their attributes.

<img src="docs/source/figures/chromo_UML.png">

## Examples
We've prepared example simulations for a variety of polymer modeling scenarios, which are presented as jupyter notebooks at the `simulations/examples` subdirectory.
