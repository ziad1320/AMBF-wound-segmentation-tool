import cv2
import numpy as np
import os
import glob

def process_batch():
    # Find all files matching the pattern in the current folder
    image_files = glob.glob("raw_images/image_*.jpeg")
    
    if not image_files:
        print("No images found matching the pattern 'image_*.jpeg'.")
        return

    # Loop over every image found
    for image_path in image_files:
        # Extract the base name (e.g., 'image_1' from 'image_1.jpeg')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create a dedicated folder for this specific image
        output_dir = f"processing_patch_7/processing_steps_{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {image_path} -> Saving to {output_dir}/")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading {image_path}. Skipping.")
            continue

        # Save Step 1
        cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), image)

        # 1. Solid Phantom Mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([130, 40, 40]) 
        upper_pink = np.array([170, 255, 255])
        raw_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        solid_phantom_mask = np.zeros_like(raw_mask)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(solid_phantom_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Save Step 2
        cv2.imwrite(os.path.join(output_dir, "2_solid_phantom_mask.jpg"), solid_phantom_mask)

        # 2. ROI
        roi = cv2.bitwise_and(image, image, mask=solid_phantom_mask)

        # 3. Grayscale & CLAHE 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        
        # Save Step 3
        cv2.imwrite(os.path.join(output_dir, "3_clahe_roi.jpg"), equalized)

        # 4. Canny Edge Detection
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # 5. Internal Edge Isolation
        kernel_erode = np.ones((25, 25), np.uint8) 
        inner_zone_mask = cv2.erode(solid_phantom_mask, kernel_erode, iterations=1)
        internal_edges = cv2.bitwise_and(edges, edges, mask=inner_zone_mask)
        
        # Save Step 4
        cv2.imwrite(os.path.join(output_dir, "4_internal_edges.jpg"), internal_edges)

        # 6. Morphological Merging
        kernel_close = np.ones((15, 15), np.uint8)
        connected_edges = cv2.morphologyEx(internal_edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Save Step 5
        cv2.imwrite(os.path.join(output_dir, "5_connected_edges.jpg"), connected_edges)

        # 7. Final Extraction
        final_mask = np.zeros_like(gray)
        wound_contours, _ = cv2.findContours(connected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if wound_contours:
            largest_wound = max(wound_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_wound) > 50:
                cv2.drawContours(final_mask, [largest_wound], -1, 255, thickness=cv2.FILLED)

        # 8. Dilation (Expanding the mask for the suturing margin)
        # The larger the numbers in the brackets, the wider the margin will be.
        kernel_expand = np.ones((10, 10), np.uint8) 
        final_mask = cv2.dilate(final_mask, kernel_expand, iterations=1)

        # Save Step 6
        cv2.imwrite(os.path.join(output_dir, "6_final_wound_mask.jpg"), final_mask)
        
    print("Batch processing complete.")

# Run the batch process
if __name__ == "__main__":
    process_batch()