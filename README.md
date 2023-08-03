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

