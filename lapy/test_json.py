import json 
import numpy as np
import pytest

from lapy import TriaMesh, TetMesh
from lapy import shapedna
import os

# Loading data
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
vtk_filename_1 = "cubeTria.vtk"
vtk_filename_2 = "cubeTetra.vtk"
tria = TriaMesh.read_vtk(os.path.join(data_dir, vtk_filename_1))
tet = TetMesh.read_vtk(os.path.join(data_dir, vtk_filename_2))



# compute eigenvalues and eigenvectors for tria mesh
ev = shapedna.compute_shapedna(tria, k=3)
ev['Eigenvectors']
ev['Eigenvalues']


def test_normalize_ev_geometry():
    with open('expected_outcomes.json', 'r') as f:
        expected_outcomes = json.load(f)
    
    expected_normalized_values = np.array(expected_outcomes['expected_outcomes']['normalize_ev_geometry']['expected_normalized_values'])
    tolerance = 1e-4
    
    normalized_eigenvalues = shapedna.normalize_ev(tria, ev['Eigenvalues'], method="geometry")
    
    assert np.allclose(normalized_eigenvalues, expected_normalized_values, atol=tolerance)
    assert normalized_eigenvalues.dtype == np.float32

