from lapy import TriaMesh, plot, Solver, heat
import os
import numpy as np
import plotly.io as pio
pio.renderers.default = "sphinx_gallery"
from lapy.plot import plot_tria_mesh
import cv2
import hashlib
import pytest
import json

images_dir = os.path.join(os.path.dirname(__file__),"..", "images")
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")


# Load the images
real_image_path = os.path.join(images_dir, "TriaMesh_Geodesics_reference.png")
generated_image_path = os.path.join(images_dir, "TriaMesh_Geodesics_generated.png")

real_img_data = cv2.imread(real_image_path)
gen_img_data = cv2.imread(generated_image_path)


filename = "square-mesh.off"
T = TriaMesh.read_off(os.path.join(data_dir, filename))

# q = T.tria_qualities()
# Uncomment following code to get visulaize generated plot
# plot_tria_mesh(
#     T,
#     plot_edges=True,
#     tfunc=q,
#     html_output=True,
#     )


# Calculate and compare image checksums
def calculate_checksum(img_data, algorithm="md5"):
    """
    Calculate the checksum hash value of the given image data.
    
    Parameters:
        img_data (numpy.ndarray): Image data as a NumPy array.
        algorithm (str, optional): Hash algorithm to use (default is "md5").
        
    Returns:
        str: Hexadecimal hash value of the image data.
    """
    hash_object = hashlib.new(algorithm)
    hash_object.update(img_data)
    return hash_object.hexdigest()

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


def test_TriaMesh_Geodesics(loaded_data):
    real_image_checksum = loaded_data['expected_outcomes']['test_TriaMesh_Geodesics']['real_img_checksum']

    generated_image_checksum = calculate_checksum(gen_img_data)
    assert real_image_checksum == generated_image_checksum, "Images are not identical based on checksums."



# Laplace
real_image_path = os.path.join(images_dir, "Laplace_Geodesics_reference.png")
generated_image_path = os.path.join(images_dir, "Laplace_Geodesics_generated.png")

real_img_data_laplace = cv2.imread(real_image_path)
gen_img_data_laplace = cv2.imread(generated_image_path)


def test_Laplase_Geodesics(loaded_data):
    # compute first eigenfunction
    fem = Solver(T, lump=True)
    eval, evec = fem.eigs()
    vfunc = evec[:,1]

    # also get A,B (lumped), and inverse of B (easy as it is diagonal due to lumping)
    A, B = fem.stiffness, fem.mass
    Bi = B.copy()
    Bi.data **= -1

    # Uncomment this line to visualize the plot
    # plot_tria_mesh(T,Bi*(A*vfunc),plot_edges=True, html_output = True)

    # or this because this is the same as eigenvalues times the eigen function 
    # plot_tria_mesh(T,eval[1]*vfunc,plot_edges=True)
    # or this 

    # from lapy.diffgeo import compute_gradient
    # from lapy.diffgeo import compute_divergence
    # grad = compute_gradient(T,vfunc)
    # divx = -compute_divergence(T,grad)
    # plot_tria_mesh(T,Bi*divx,plot_edges=True)


    real_image_checksum = loaded_data['expected_outcomes']['test_Laplase_Geodesics']['real_img_checksum']
    generated_image_checksum = calculate_checksum(gen_img_data_laplace)
    assert real_image_checksum == generated_image_checksum, "Images are not identical based on checksums."
    assert B.sum() == 1
    assert Bi is not B

# Geodesics
from lapy.diffgeo import compute_gradient
from lapy.diffgeo import compute_divergence

bvert = T.boundary_loops()
u = heat.diffusion(T, bvert, m=1)
# plot_tria_mesh(T,u,plot_edges=True,plot_levels=True, html_output=True)

def test_heat_diffusion_shape():
    expected_shape = (415,)
    assert u.shape == expected_shape



# compute gradient of heat diffusion
tfunc = compute_gradient(T,u)

# normalize gradient
X = -tfunc / np.sqrt((tfunc**2).sum(1))[:,np.newaxis]
X = np.nan_to_num(X)
divx = compute_divergence(T,X)
# print(divx.shape)



# compute distance
from scipy.sparse.linalg import splu
useCholmod = True
try:
    from sksparse.cholmod import cholesky
except ImportError:
    useCholmod = False

fem = Solver(T,lump=True)
A, B = fem.stiffness, fem.mass

H=-A
b0=divx

# solve H x = b0
# we don't need the B matrix here, as divx is the intgrated divergence
print("Matrix Format now: "+H.getformat())
if useCholmod:
    print("Solver: cholesky decomp - performance optimal ...")
    chol = cholesky(H)
    x = chol(b0)
else:
    print("Solver: spsolve (LU decomp) - performance not optimal ...")
    lu = splu(H)
    x = lu.solve(b0)

# remove shift
x = x-min(x)

Bi = B.copy()
vf = fem.poisson(-Bi*divx)
vf = vf - min(vf)
# print(vf)


print(max(abs(vf-x)))
from lapy.diffgeo import compute_geodesic_f
gf = compute_geodesic_f(T,u)

def test_Geodesics_format(loaded_data):
    """
    Test if matrix format, solver settings, max distance,
    and computed values match the expected outcomes.

    Parameters:
    - loaded_data (dict): Dictionary containing loaded test data.

    Raises:
    - AssertionError: If any test condition is not met.
    """
    expected_matrix_format = loaded_data['expected_outcomes']['test_Geodesics_format']['expected_matrix_format']
    assert H.getformat() == expected_matrix_format
    assert useCholmod == False, "Solver: cholesky decomp - performance optimal ..."
    expected_max_x = loaded_data['expected_outcomes']['test_Geodesics_format']['max_distance']
    expected_sqrt_2_div_2 = loaded_data['expected_outcomes']['test_Geodesics_format']['expected_sqrt_2_div_2']
    assert np.isclose(max(x), expected_max_x)
    computed_sqrt_2_div_2 = np.sqrt(2) / 2
    assert np.isclose(computed_sqrt_2_div_2, expected_sqrt_2_div_2)
    expected_max_abs_diff = loaded_data['expected_outcomes']['test_Geodesics_format']['expected_max_abs_diff']
    computed_max_abs_diff = max(abs(gf-x))
    assert np.allclose(computed_max_abs_diff, expected_max_abs_diff)



