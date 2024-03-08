import cv2
import os
import numpy as np

def extract_colored_outer_contour(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each binary image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
            # Read the binary image
            binary_path = os.path.join(input_folder, filename)
            binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)           
            
            # Use a suitable thresholding method to create a binary image
            _, thresh = cv2.threshold(binary_image, 20, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image; RETR_EXTERNAL retrieves only the extreme outer contours
            contours, _ = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area to keep only larger contours (human-sized)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 25]

            # Create an empty black image for contours
            contour_image = np.zeros_like(binary_image)

            # Convert the black image to RGB
            contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)
            
            # Draw the filtered contours on the black image
            #cv2.drawContours(contour_image, filtered_contours, -1, (255), thickness=2) 
            
            # Draw the filtered contours on the RGB image
            for i, contour in enumerate(filtered_contours):
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.drawContours(contour_image_rgb, [contour], -1, color, thickness=2)  

            # Bitwise AND operation to extract the regions with contours from the original image
            #result = cv2.bitwise_and(binary_image, binary_image, mask=contour_image)

            # Save the result
            #output_path = os.path.join(output_folder, f"outer_contour_{filename}")
            #cv2.imwrite(output_path, result)
            
            # Save the result
            output_path = os.path.join(output_folder, f"colored_outer_contour_{filename}")
            cv2.imwrite(output_path, contour_image_rgb)

    print("Colored outer contour extraction completed.")

input_folder_path = "../../data/interim/masks"
output_folder_path = "../../data/interim/outlines"

extract_colored_outer_contour(input_folder_path, output_folder_path)