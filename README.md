# Alphorn_simulation
Numerical Simulation of the sound of an Alphorn, valid as Semester Project 2024-25 at EPFL

## Repository infos

The repository is organized in the following way:
* _bell_preprocessing_: folder that contains all the pipeline needed to create `final_bell.csv` and `mouthpiece.csv` files.
* _data_mesh_generation.py_: file for constructiong centerlines and radii data to be used by Fabio Marcinno mesh implementation program. The most important function of this file is `assemble_mesh_data`, whose flexibility allows to change:
    * the geometrical properities of the first two conical parts of the instrument (lengths and radius of intersection)
    * the presence of absence of mouthpiece

   Mesh are constructed and stored in the folder `mesh_data` with a labeled name (e.g.`mesh_data1.csv`, `mesh_data2.csv`, ...). In this folder, the already present files `final_bell.csv` and `mouthpiece.csv` are obtained by image preprocessing, and are used for constructing the entire mesh.
* _solver.py_: contains functions for solving Helmholz equation and compute results. In particular:
    * _solve_helmholtz_: solves PDEs and stores input impedance results in file. Labeling of output file is coherent with .msh file used for computations.
    * _compute_resonance_freqs_: computes resonance frequencies starting from file of results.
    * _plot_results_: saves input impadance plots in .png file.
    * _main_: where to change mesh file, frequency ranges and input velocity.
* _mesh_data_: folder with .csv files for mesh implementation.
* _alphorn_meshes_: folder with .msh files with meshes to be used by Dolfinx. Here we have already 3 meshes:
    * _mesh1.msh_: coarse mesh
    * _mesh2.msh_: middle mesh
    * _mesh3.msh_: fine mesh

   All of them are related to 32 mm intersection radius and 447.09 mm hemispherical output radius
* _results_: folder with .csv files containing computed input impedance data (real part, imaginary part, modululs)
* _plots_: folder with .png files with input impedance plots

**Important**: .msh file names should end with a number. In this way, file with results and plots will be generated accordingly to that number and a easy mapping between input and ouput can be made.


## How to set up Docker image on VSCode

_solver.py_ needs a PETSc version which supports `complex128`. I made use of a [(Docker Image)](https://hub.docker.com/r/dolfinx/lab) specifically created for this type of problems. If working on Linux, after having installed Docker with the commands:

```
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

You need to pull the docker image:
```
docker pull dolfinx/lab:nightly
```

If you are programming on VSCode, then you just need to run the `run_docker.sh` bash file for running the container. The file `.devcontainer/devcontainer.json` will do the rest.

## How to run

_data_mesh_generation.py_ can be run locally with only **numpy**, **warnings** and **os** libraries. Simply change main file data to construct different mesh data files.

_solver.py_ can be run inside the Docker image just by running the `run_solver.py` bash file. Simply change main file data to construct different numerical problems.
