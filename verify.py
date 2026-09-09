import cv2
import os

def verify_alignment_batch():
    # 1. Create a new folder to hold all the results
    output_dir = "alignment_patch_2"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all alignment checks to: {output_dir}/\n")

    # 2. Loop through folder numbers 1 to 8
    for i in range(1, 9):
        # Dynamically build the folder and file paths for each iteration
        folder_path = f"processing_patch_7/processing_steps_image_{i}"
        original_path = f"{folder_path}/1_original.jpg"
        mask_path = f"{folder_path}/6_final_wound_mask.jpg"

        # Sanity check: Ensure the files exist before trying to open them
        if not os.path.exists(original_path) or not os.path.exists(mask_path):
            print(f"Skipping image {i}: Missing files in {folder_path}")
            continue

        # 3. Load the images
        original = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if original is None or mask is None:
            print(f"Error loading images for iteration {i}. Skipping.")
            continue

        if original.shape[:2] != mask.shape:
            print(f"Error: Size mismatch for iteration {i}. Skipping.")
            continue

        # 4. Create overlay and paint mask coordinates green
        overlay_image = original.copy()
        overlay_image[mask == 255] = [0, 255, 0]

        # 5. Blend the images
        alpha = 0.6 
        beta = 0.4
        blended_result = cv2.addWeighted(original, alpha, overlay_image, beta, 0)
        
        # 6. Save the result to the new folder with a dynamic filename
        output_filename = f"{output_dir}/alignment_check_{i}.jpg"
        cv2.imwrite(output_filename, blended_result)
        print(f"Successfully processed and saved: {output_filename}")

    print("\nBatch verification complete.")

# Run the script
verify_alignment_batch()