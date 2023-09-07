
import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import lapy



# Structural Similarity Index (SSIM) methods take into account the actual pixel values 
# and their spatial relationships to determine how similar two images are in terms
# of appearance.
images_dir = os.path.join(os.path.dirname(__file__),"..", "images")

# Load the images
real_image_path = os.path.join(images_dir, "Reference_image.png")
generated_image_path = os.path.join(images_dir, "test_image_1.png")

# Load the images
real_image_path = os.path.join(images_dir, "Reference_image.png")
generated_image_path = os.path.join(images_dir, "test_image_1.png")

real_img_data = cv2.imread(real_image_path)
gen_img_data = cv2.imread(generated_image_path)

# Convert images to grayscale (SSIM works on grayscale images)
real_gray = cv2.cvtColor(real_img_data, cv2.COLOR_BGR2GRAY)
gen_gray = cv2.cvtColor(gen_img_data, cv2.COLOR_BGR2GRAY)

# Calculate SSIM
ssim_score = compare_ssim(real_gray, gen_gray)
print("ssim_score {}".format(ssim_score))
# Set a threshold for similarity
threshold = 0.95

#  SSIM provides a value between -1 and 1, where 1 indicates identical images, 
# 0 indicates no similarity, and -1 indicates complete dissimilarity.

if ssim_score >= threshold:
    print("Images are very similar based on SSIM.")
else:
    print("Images are not very similar based on SSIM.")
