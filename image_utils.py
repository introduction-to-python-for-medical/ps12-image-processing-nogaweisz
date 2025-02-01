import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(file_path):
    """
    Load a color image and convert it to a NumPy array.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        np.array: Image data as a NumPy array, or None if loading fails
    """
    try:
        # Check if file path is provided
        if not file_path:
            raise ValueError("File path cannot be empty")
            
        # Open the image using PIL (Python Imaging Library)
        img = Image.open(file_path)
        
        # Convert the image to RGB mode if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        
        # Verify the array has the correct shape (height, width, 3)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Invalid image format. Expected RGB image, got shape {img_array.shape}")
            
        return img_array
        
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def edge_detection(image_array):
    """
    Perform edge detection on an input image using Sobel operators.
    
    Args:
        image_array (np.array): Input color image array with shape (height, width, 3)
        
    Returns:
        np.array: Edge magnitude array with shape (height, width), or None if input is invalid
    """
    # Validate input
    if image_array is None:
        print("Error: Input image array is None")
        return None
        
    try:
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
        
        return edge_mag
        
    except Exception as e:
        print(f"Error performing edge detection: {str(e)}")
        return None
