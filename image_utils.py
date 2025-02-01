import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(file_path):
    """
    Load a color image and convert it to a NumPy array.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        np.array: Image data as a NumPy array
    """
    # Open the image using PIL (Python Imaging Library)
    img = Image.open(file_path)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    return img_array


def edge_detection(image_array):
    """
    Perform edge detection on an input image using Sobel operators.
    
    Args:
        image_array (np.array): Input color image array with shape (height, width, 3)
        
    Returns:
        np.array: Edge magnitude array with shape (height, width)
    """
    # Convert to grayscale by averaging the color channels
    grayscale = np.mean(image_array, axis=2)
    
    # Define Sobel filters for vertical and horizontal edges
    kernel_y = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ])
    
    kernel_x = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    
    # Apply convolution with zero padding to maintain image size
    edge_x = convolve2d(grayscale, kernel_x, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(grayscale, kernel_y, mode='same', boundary='fill', fillvalue=0)
    
    # Compute edge magnitude using the Pythagorean theorem
    edge_mag = np.sqrt(np.square(edge_x) + np.square(edge_y))
    
    return edge_mag
