import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection

def main():
    # Step 1: Load the image
    try:
        image = load_image("orig_img.jpg")
        print("Image loaded successfully")
        print(f"Image shape: {image.shape}")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return

    # Step 2: Apply median filter for noise suppression
    clean_image = median(image, ball(3))
    print("Noise suppression completed")

    # Step 3: Perform edge detection
    edge_mag = edge_detection(clean_image)
    print("Edge detection completed")

    # Step 4: Compute histogram for threshold selection
    plt.figure(figsize=(10, 4))
    plt.hist(edge_mag.ravel(), bins=256)
    plt.title('Edge Magnitude Histogram')
    plt.xlabel('Edge Magnitude')
    plt.ylabel('Frequency')
    plt.show()

    # Step 5: Convert to binary image using threshold
    # You can adjust this threshold based on the histogram
    threshold = np.percentile(edge_mag, 90)  # Using 90th percentile as default
    edge_binary = (edge_mag > threshold).astype(np.uint8) * 255
    print(f"Binary conversion completed with threshold: {threshold:.2f}")

    # Step 6: Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(clean_image)
    plt.title('Noise Suppressed Image')
    
    plt.subplot(133)
    plt.imshow(edge_binary, cmap='gray')
    plt.title('Edge Detection Result')
    
    plt.tight_layout()
    plt.show()

    # Step 7: Save the result
    try:
        edge_image = Image.fromarray(edge_binary)
        edge_image.save('edge_detection_result.png')
        print("Edge detection result saved as 'edge_detection_result.png'")
    except Exception as e:
        print(f"Error saving image: {str(e)}")

if _name_ == "_main_":
    main()
