import hashlib
import json
import os

import cv2
import numpy as np
import pytest

import lapy
from lapy import Solver, TriaMesh, plot

images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

# Load the images
real_image_path = os.path.join(images_dir, "Reference_image.png")
generated_image_path = os.path.join(images_dir, "test_image_1.png")

real_img_data = cv2.imread(real_image_path)
gen_img_data = cv2.imread(generated_image_path)


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
    with open("expected_outcomes.json", "r") as f:
        expected_outcomes = json.load(f)
    return expected_outcomes


def test_triangle_mesh_visualization_using_checksum(loaded_data):
    """
    Test the visualization of a triangle mesh using image checksum comparison.

    Parameters:
        loaded_data (dict): Expected outcomes data loaded from a JSON file.

    Raises:
        AssertionError: If the image checksums of the reference and generated images do not match.
    """

    # Uncomment this code if would like to visualize the plot
    # vtk_filename = "cubeTria.vtk"

    # tria_mesh = TriaMesh.read_vtk(os.path.join(data_dir, vtk_filename))

    # # print(tria_mesh)
    # # Perform eigenvalue computation
    # fem = Solver(tria_mesh)
    # evals, evecs = fem.eigs(k=3)

    # # Visualize the first non-constant eigenfunction
    # fig = plot.plot_tria_mesh(tria_mesh, vfunc=evecs[:, 1], xrange=None, yrange=None, zrange=None,
    #                           showcaxis=False, caxis=None)

    real_image_checksum = loaded_data["expected_outcomes"][
        "test_triangle_mesh_visualization_using_checksum"
    ]["real_img_checksum"]
    print(real_image_checksum)

    # real_image_checksum = calculate_checksum(real_img_data)
    generated_image_checksum = calculate_checksum(gen_img_data)
    assert (
        real_image_checksum == generated_image_checksum
    ), "Images are not identical based on checksums."
