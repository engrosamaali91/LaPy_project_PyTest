# LaPy_project_PyTest
The repository is associated with test cases using pytest

# Test Cases for TriaMesh Class

This `test_tria_mesh.py` file contains a comprehensive set of test cases for the `TriaMesh` class, which is responsible for working with triangle meshes. These test cases cover various methods and functionalities provided by the `TriaMesh` class to ensure its correctness and robustness.

    Checking for free vertices
    Removing free vertices
    Orienting the triangle mesh
    Computing vertex normals
    Computing the average edge length
    Applying a normal offset
    Extracting the boundary loops
    Checking for manifoldness
    Computing the boundary triangle mesh
    Computing the volume of the boundary mesh

Each test case validates the expected behavior of the corresponding method in the TriaMesh class.

## Running Tests

To run the test cases, you can use your preferred testing framework. We recommend using `pytest`. To install it, you can use the following command:

```bash
pip install pytest
```

Once pytest is installed, you can run the tests using the following command:
```bash
pytest test_tria_mesh.py
```


# Test cases for TetMesh class

The test cases cover various functionalities of the TetMesh class, including:

    Checking for free vertices
    Removing free vertices
    Orienting the tetrahedral mesh
    Computing average edge length
    Extracting the boundary surface
    Computing the volume of the boundary surface

Each test case validates the expected behavior of the corresponding method in the TetMesh class.


## Running Tests
```bash
pytest test_tet_mesh.py
```

# Test Case: Visualize First Non-Constant Eigenfunction

This test case focuses on visualizing the first non-constant eigenfunction on a cube mesh and comparing it with a reference image. The purpose of this test is to ensure that the visualization function produces results that are visually similar to the expected reference image.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (>= 3.8)
- Required Python packages: `numpy`, `pytest`, `lapy`, `matplotlib`, `opencv-python`

You can install the required packages using the following command in your virtual environment:

```bash
pip install numpy pytest lapy matplotlib imageio opencv-python
```
## Running Tests
```bash
pytest test_plot_tria_mesh.py
```




# Test cases for ShapeDNA Module
The ShapeDNA module provides functions for computing, normalizing, and analyzing shape descriptors for both tria and tet meshes. 

## Test Cases

We have provided comprehensive unit test cases for the ShapeDNA module to ensure its correctness. Each test case verifies specific functionality and compares computed values with expected results using tolerance thresholds. The test cases cover:

1. Computing eigenvalues and eigenvectors.
2. Normalizing eigenvalues using the 'geometry' method.
3. Linear reweighting of eigenvalues.
4. Computing distances between eigenvalues.

## Running Tests

To run the test cases, ensure you have the required dependencies installed. You can then use a testing framework like `pytest`:

```bash
pip install pytest
pytest test_shape_DNA.py
```


# TriaMesh Geodesics

This repository showcases the usage and testing of the TriaMesh Geodesics functionality provided by the `lapy` library.
It demonstrates various operations such as heat diffusion, Laplacian eigenfunctions, and geodesics on triangular meshes.

## Test Cases


The repository includes tests to ensure the correctness of the implemented functionalities. The tests cover the following scenarios:

- **TriaMesh Geodesics**: Tests the identity of reference and generated images.
- **Laplacian Geodesics**: Tests Laplacian eigenfunctions and solver settings.
- **Heat Diffusion Shape**: Tests the shape of heat diffusion results.
- **Geodesics Format**: Tests matrix format, solver settings, max distance, and computed values.

## Running Tests

1. Place the expected outcomes data in a JSON file named `expected_outcomes.json` or retrive data from 'expected_outcomes.json' file by placing it in same directory as of your source code.
2. Run the following command:
```bash
pip install pytest
pytest test_TriaMesh_Geodesics.py
```







