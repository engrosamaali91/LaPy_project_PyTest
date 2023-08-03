import numpy as np
import pytest
import lapy
from lapy import TriaMesh, Solver, plot
from lapy import io

import matplotlib.pyplot as plt
import imageio
import os
import cv2 as cv




def test_visualize_first_eigenfunction():
    '''
    Test visualization of the first non-constant eigenfunction on a cube mesh
    '''
    # .. is used to move up one level in the directory structure 
    # from the location of your test script (test_plot_tria_mesh.py),
    # which should take you to the LaPy folder where the data folder and images folder should resides. 
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    images_dir = os.path.join(os.path.dirname(__file__),"..", "images")

    vtk_filename = "cubeTria.vtk"

    tria_mesh = TriaMesh.read_vtk(os.path.join(data_dir, vtk_filename))


    # print(tria_mesh)
    # Perform eigenvalue computation
    fem = Solver(tria_mesh)
    evals, evecs = fem.eigs(k=3)

    # Visualize the first non-constant eigenfunction
    fig = plot.plot_tria_mesh(tria_mesh, vfunc=evecs[:, 1], xrange=None, yrange=None, zrange=None,
                              showcaxis=False, caxis=None)

    gen_img_filename = "test_image_1.png"
    img_path = os.path.join(images_dir, gen_img_filename)
    gen_img_data = cv.imread(img_path)
    # cv.imshow('Generated Image', gen_img_data)
    
    ref_img_filename = "Reference_image.png"
    img_path = os.path.join(images_dir, ref_img_filename)
    ref_img_data = cv.imread(img_path)
    # cv.imshow('Reference Image', ref_img_data)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    # Compare the images using an image similarity metric (e.g., mean squared error)
    diff = np.abs(gen_img_data - ref_img_data)
    mse = np.mean(diff)

    # Define a threshold for image similarity
    similarity_threshold = 100  # Adjust the threshold based on your use case

    # Perform assertion based on image similarity
    assert mse < similarity_threshold, f"Images are not similar (MSE: {mse})"


# test_visualize_first_eigenfunction()



# Below code is for reference only, this is testable with absolute path only

# def visualize_first_eigenfunction():
#     '''
#     Visualize the first non-constant eigenfunction on a cube mesh and save the plot as an image
#     '''

#     # Load the mesh
#     tria_mesh = TriaMesh.read_vtk("/home/ashrafo/LaPy/data/cubeTria.vtk")


#     # Perform eigenvalue computation
#     fem = Solver(tria_mesh)
#     evals, evecs = fem.eigs(k=3)


#     print(f"Eigen values:", evals)
#     print(50*"*")
#     print(f"Eigen vectors:", evecs)

#     # Visualize the first non-constant eigenfunction
#     fig = plot.plot_tria_mesh(tria_mesh, vfunc=evecs[:, 1], xrange=None, yrange=None, zrange=None,
#                               showcaxis=False, caxis=None)
    

#     gen_img_filename = "test_image_1.png"
#     img_path = os.path.join("/home/ashrafo/LaPy/lapy/images", gen_img_filename)
#     # fig.write_image(img_path, format='png')

#     # Load the generated image using OpenCV and display it
#     gen_img_data = cv.imread(img_path)
#     # cv.imshow('Generated Image', gen_img_data)

#     ref_img_filename = "Reference_image.png"
#     img_path = os.path.join("/home/ashrafo/LaPy/lapy/images", ref_img_filename)

#     # Load the reference image using OpenCV and display it
#     Ref_img_data = cv.imread(img_path)
#     # cv.imshow('Reference Image', Ref_img_data)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()


#     # Calculate mean squared error (MSE) between generated and referemce image
#     mse = np.mean((gen_img_data - Ref_img_data)**2)
#     print(f"mse, {mse}")

#     # Define a threshold for image similarity, this would be a design decision
#     similarity_threshold  = 100 
#     # mse = 101
#     # Perform assertionbased on image similarity 
#     assert mse < similarity_threshold , f"Images are not similar (MSE: {mse})"

# # Call the function to visualize the plot
# visualize_first_eigenfunction()  


