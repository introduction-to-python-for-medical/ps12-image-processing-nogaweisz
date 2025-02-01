import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
from skimage.feature import canny  # If you want to use Canny for edge detection

def load_image(image_path):
    try:
        return plt.imread(image_path)  # Load image as a NumPy array
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def edge_detection(image):
    # For demonstration, we'll use the Canny edge detector here
    return canny(image)

def main():
    image_path = "orig_img.jpg" 
    
    try:
        # Step 1: Load image
        original_image = load_image(image_path)
        if original_image is None:
            raise ValueError("Failed to load image")

        # Step 2: Apply noise reduction
        clean_image = median(original_image, ball(3))

        # Step 3: Detect edges
        edge_result = edge_detection(clean_image)

        # Step 4: Normalize edge magnitudes to 0-255 range
        edge_normalized = ((edge_result - edge_result.min()) / 
                         (edge_result.max() - edge_result.min()) * 255).astype(np.uint8)

        # Step 5: Save the result
        edge_image = Image.fromarray(edge_normalized)
        edge_image.save('edge_result.png')

        print("Edge detection completed successfully")
        return True

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return False

if __name__ == "__main__":
    main()

