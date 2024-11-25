import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_and_analyse_images(path_to_images_folders):
    """
    Function to load and analyse the images, Displays 10 xray images of each class with it's size
    Args:
        path_to_images_folders(List): path to all the 3 images folders
    Returns:
        None
    """
    for folder in path_to_images_folders:
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"Images of {folder}:\n")
        for image_file in images[:5]:
            image_path = os.path.join(folder, image_file)
    
            image = Image.open(image_path)
    
            plt.imshow(image)
            plt.axis('off') 
            plt.title(f"Image: {image_file}\nSize: {image.size}")
            plt.show()
    return None


def grayscale_and_resize(path_to_images_folders, path_to_save_images_folder, size):
    """
    Function to grayscale and resize images to specific size
    Args:
        path_to_images_folders(List): path to all the 3 images folders
        path_to_save_images_folder(String): path to new folder where processed images are saved
        size(Tuple): size to which images are resized
    Returns:
        None
    """
    for folder in path_to_images_folders:
        new_folder = os.path.join(path_to_save_images_folder, os.path.basename(folder))
        os.makedirs(new_folder, exist_ok=True)
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for image_file in images:
            image_path = os.path.join(folder, image_file)
            
            image = Image.open(image_path)
    
            # Grayscale
            gray_image = image.convert('L')
    
            # Resize
            resized_image = gray_image.resize(size)
    
            # Save image
            new_image_path = os.path.join(new_folder, image_file)
            resized_image.save(new_image_path)
            
        print(f"\nImages of {folder} class:")
        print(f"Original size: {image.size} (Width x Height in pixels)")
        print(f"New size: {resized_image.size} (Width x Height in pixels)")
        print(f"All {len(images)} images are grayscaled, resized and saved in {new_folder}")
        return None


def normalize_and_brighten(path_to_images_folders, path_to_save_images_folder, brightness_factor):
    """
        Function to normalize and brighten images, prepares images for model
    Args:
        path_to_images_folders(List): path to all the 3 images folders
        path_to_save_images_folder(String): path to new folder where processed images are saved
        brightness_factor(Integer): size to which images are resized
    Returns:
        None
    """
    for folder in path_to_images_folders:
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for image_file in images:
            image_path = os.path.join(folder, image_file)
            
            image = Image.open(image_path)
            
            image_array = np.array(image)

            # Normalize
            normalized_image = image_array / 255.0
    
            # Brighten
            brightened_image = normalized_image + (brightness_factor / 255.0)
            brightened_image = np.clip(brightened_image, 0, 1)
            final_image = (brightened_image * 255).astype(np.uint8)
            final_image = Image.fromarray(final_image)
            
            # save image
            new_image_path = os.path.join(folder, image_file)
            final_image.save(new_image_path)
            
        print(f"\nImages of {folder} class:")
        print(f"All {len(images)} images are normalized, brightened and saved")
        return None