# imports
from lapy import TriaMesh, TetMesh
from lapy import shapedna
import os
import numpy as np

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

def test_compute_shapedna():

    """
    This function performs unit testing for the compute_shapedna() function for tria mesh by
    comparing the computed Eigenvalues with expected values using a specified tolerance.
    Additionally, it asserts that the dtype of the computed Eigenvalues is float32.
    """
    expected_Eigenvalues = np.array([-4.0165149e-05,  4.1696410e+00,  4.1704664e+00])
    tolerance = 1e-4  # Adjust the tolerance value as needed
    
    # assert np.isclose(ev['Eigenvalues'], expected_Eigenvalues, tolerance).all()
    assert np.allclose(ev['Eigenvalues'], expected_Eigenvalues, atol=tolerance)
    assert ev['Eigenvalues'].dtype == np.float32

def test_normalize_ev_geometry():

    """
    This function performs unit testing for the normalize_ev() function using the
    'geometry' method. It compares the normalized eigenvalues with expected values.
    It also compares whether the data type of the normalized is float32
    """
    expected_normalized_values = np.array([-2.4099089e-04, 2.5017845e+01, 2.5022799e+01])
    tolerance = 1e-4
    
    normalized_eigenvalues = shapedna.normalize_ev(tria, ev['Eigenvalues'], method="geometry")
    
    assert np.allclose(normalized_eigenvalues, expected_normalized_values, atol=tolerance)
    assert normalized_eigenvalues.dtype == np.float32


def test_reweight_ev():

    """
    This function performs unit testing for reweighted_ev() function.
    It also compares whether the data type of the reweighted eigenvalues is float32
    """
    expected_reweighted_values = np.array([-4.01651487e-05,  2.08482051e+00,  1.39015547e+00])
    original_eigenvalues = np.copy(ev['Eigenvalues'])
    reweighted_eigenvalues =  shapedna.reweight_ev(ev['Eigenvalues'])
    tolerance = 1e-4
    assert np.allclose(reweighted_eigenvalues, expected_reweighted_values, atol=tolerance)
    

def test_compute_distance():
    """
    This function performs unit testing for compute_distance() function.
    """
    # compute distance for tria eigenvalues (trivial case)
    computed_distance = shapedna.compute_distance(ev["Eigenvalues"], ev["Eigenvalues"])
    expected_compute_distance = 0.0
    assert computed_distance == expected_compute_distance

# Repeat testing steps for a tetrahedral mesh
# compute eigenvalues and eigenvectors for tet mesh
evTet = shapedna.compute_shapedna(tet, k=3)
evTet['Eigenvectors']
evTet['Eigenvalues']

def test_compute_shape_dna():
    """
    This functoin performs unit testing for compute_shapedna function for tet mesh
    """
    tolerance = 1e-4
    expected_eigen_values = np.array([8.4440224e-05, 9.8897915e+00, 9.8898811e+00])
    # evTet = shapedna.compute_shapedna(tet, k=3)
    assert np.allclose(evTet['Eigenvalues'], expected_eigen_values, atol=tolerance)
    assert evTet['Eigenvalues'].dtype == np.float32
    
def test_normalize_ev_geometry_tet():

    """
    This function performs unit testing for the normalize_ev() function for tet mesh using the
    'geometry' method. It compares the normalized eigenvalues with expected values.
    It also compares whether the data type of the normalized is float32
    """
    expected_normalized_values = np.array([8.4440224e-05, 9.8897915e+00, 9.8898811e+00])
    tolerance = 1e-4
    # evTet = shapedna.compute_shapedna(tet, k=3)
    # volume / surface / geometry normalization of tet eigenvalues
    normalized_eigenvalues = shapedna.normalize_ev(tet, evTet['Eigenvalues'], method="geometry")
    
    assert np.allclose(normalized_eigenvalues, expected_normalized_values, atol=tolerance)
    assert normalized_eigenvalues.dtype == np.float32


def test_reweight_ev_tet():

    """
    This function performs unit testing for reweighted_ev() function for tet meshes.
    It also compares whether the data type of the reweighted eigenvalues is float32
    """
    expected_reweighted_values = np.array([8.44402239e-05, 4.94489574e+00, 3.29662704e+00])
    # Linear reweighting of tet eigenvalues
    reweighted_eigenvalues =  shapedna.reweight_ev(evTet['Eigenvalues'])
    tolerance = 1e-4
    assert np.allclose(reweighted_eigenvalues, expected_reweighted_values, atol=tolerance)
    

def test_compute_distance_tet():
    """
    This function performs unit testing for compute_distance() function.
    """
    # compute distance for tria eigenvalues (trivial case)
    computed_distance = shapedna.compute_distance(evTet["Eigenvalues"], evTet["Eigenvalues"])
    expected_compute_distance = 0.0
    
    # Compare the computed distance with the expected distance using a tolerance
    assert computed_distance == expected_compute_distance
