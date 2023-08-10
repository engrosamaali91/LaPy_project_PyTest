from lapy import TriaMesh, TetMesh
from lapy import shapedna
import os
import numpy as np
import json
import pytest

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


@pytest.fixture
def loaded_data():
    """
    Load and provide the expected outcomes data from a JSON file.
    
    Returns:
        dict: Dictionary containing the expected outcomes data.
    """
    with open('expected_outcomes.json', 'r') as f:
        expected_outcomes = json.load(f)
    return expected_outcomes

def test_compute_shapedna(loaded_data):
    """
    Test the compute_shapedna function for a triangular mesh.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the computed eigenvalues do not match the expected values within tolerance.
        AssertionError: If the dtype of the computed eigenvalues is not float32.
    """
    expected_Eigenvalues = np.array(loaded_data["expected_outcomes"]["test_compute_shapedna"]["expected_eigenvalues"])
    tolerance = loaded_data["expected_outcomes"]["test_compute_shapedna"]["tolerance"]
    assert np.allclose(ev['Eigenvalues'], expected_Eigenvalues, atol=tolerance)
    assert ev['Eigenvalues'].dtype == np.float32

def test_normalize_ev_geometry(loaded_data):
    """
    Test the normalize_ev() function using the 'geometry' method for a triangular mesh.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the normalized eigenvalues do not match the expected values within tolerance.
        AssertionError: If the dtype of the normalized eigenvalues is not float32.
    """
    expected_normalized_values = np.array(loaded_data["expected_outcomes"]["test_normalize_ev_geometry"]["expected_normalized_values"])
    tolerance = loaded_data["expected_outcomes"]["test_normalize_ev_geometry"]["tolerance"]
    normalized_eigenvalues = shapedna.normalize_ev(tria, ev['Eigenvalues'], method="geometry")
    assert np.allclose(normalized_eigenvalues, expected_normalized_values, atol=tolerance)
    assert normalized_eigenvalues.dtype == np.float32


def test_reweight_ev(loaded_data):
    """
    Test the reweighted_ev() function and validate the data type of the reweighted eigenvalues.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the reweighted eigenvalues do not match the expected values within tolerance.
        AssertionError: If the dtype of the reweighted eigenvalues is not float32.
    """
    expected_reweighted_values = np.array(loaded_data["expected_outcomes"]["test_reweight_ev"]["expected_reweighted_values"])
    tolerance = loaded_data["expected_outcomes"]["test_reweight_ev"]["tolerance"]
    reweighted_eigenvalues =  shapedna.reweight_ev(ev['Eigenvalues'])
    tolerance = 1e-4
    assert np.allclose(reweighted_eigenvalues, expected_reweighted_values, atol=tolerance)    

def test_compute_distance(loaded_data):
    """
    Test the compute_distance() function for eigenvalues and validate the computed distance.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the computed distance does not match the expected value.
    """
    expected_compute_distance = loaded_data["expected_outcomes"]["test_compute_distance"]["expected_compute_distance"]
    # compute distance for tria eigenvalues (trivial case)
    computed_distance = shapedna.compute_distance(ev["Eigenvalues"], ev["Eigenvalues"])
    assert computed_distance == expected_compute_distance

# Repeat testing steps for a tetrahedral mesh
# compute eigenvalues and eigenvectors for tet mesh
evTet = shapedna.compute_shapedna(tet, k=3)
evTet['Eigenvectors']
evTet['Eigenvalues']

def test_compute_shapedna_tet(loaded_data):
    """
    Test the compute_shapedna function for a tetrahedral mesh.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the computed eigenvalues do not match the expected values within tolerance.
        AssertionError: If the dtype of the computed eigenvalues is not float32.
    """
    expected_eigen_values = np.array(loaded_data["expected_outcomes"]["test_compute_shapedna_tet"]["expected_eigen_values"])
    tolerance = loaded_data["expected_outcomes"]["test_compute_shapedna_tet"]["tolerance"]
    evTet = shapedna.compute_shapedna(tet, k=3)
    assert np.allclose(evTet['Eigenvalues'], expected_eigen_values, atol=tolerance)
    assert evTet['Eigenvalues'].dtype == np.float32
    
def test_normalize_ev_geometry_tet(loaded_data):
    """
    Test the normalize_ev() function using the 'geometry' method for a tetrahedral mesh.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the normalized eigenvalues do not match the expected values within tolerance.
        AssertionError: If the dtype of the normalized eigenvalues is not float32.
    """
    expected_normalized_values = np.array(loaded_data["expected_outcomes"]["test_normalize_ev_geometry_tet"]["expected_normalized_values"])
    tolerance = loaded_data["expected_outcomes"]["test_normalize_ev_geometry_tet"]["tolerance"]
    # volume / surface / geometry normalization of tet eigenvalues
    normalized_eigenvalues = shapedna.normalize_ev(tet, evTet['Eigenvalues'], method="geometry")
    
    assert np.allclose(normalized_eigenvalues, expected_normalized_values, atol=tolerance)
    assert normalized_eigenvalues.dtype == np.float32


def test_reweight_ev_tet(loaded_data):
    """
    Test the reweighted_ev() function for tetrahedral meshes and validate the data type of the reweighted eigenvalues.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the reweighted eigenvalues do not match the expected values within tolerance.
        AssertionError: If the dtype of the reweighted eigenvalues is not float32.
    """
    expected_reweighted_values = np.array(loaded_data["expected_outcomes"]["test_reweight_ev_tet"]["expected_reweighted_values"])
    tolerance = loaded_data["expected_outcomes"]["test_reweight_ev_tet"]["tolerance"]
    # Linear reweighting of tet eigenvalues
    reweighted_eigenvalues =  shapedna.reweight_ev(evTet['Eigenvalues'])
    assert np.allclose(reweighted_eigenvalues, expected_reweighted_values, atol=tolerance)
    

def test_compute_distance_tet(loaded_data):
    """
    Test the compute_distance() function for eigenvalues of tetrahedral meshes and validate the computed distance.
    
    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.
        
    Raises:
        AssertionError: If the computed distance does not match the expected value.
    """
    # compute distance for tria eigenvalues (trivial case)
    computed_distance = shapedna.compute_distance(evTet["Eigenvalues"], evTet["Eigenvalues"])
    expected_compute_distance = loaded_data["expected_outcomes"]["test_compute_distance_tet"]["exp_compute_distance"]
    
    # Compare the computed distance with the expected distance using a tolerance
    assert computed_distance == expected_compute_distance


