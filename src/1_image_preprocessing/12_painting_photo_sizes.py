# libraries ----
import os
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import math

# extract information on photos from paintings ----
def get_image_size_resolution(image_path):
    """
    Get the width, height, and resolution of an image.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - Tuple[int, int, Tuple[float, float]]: The width, height, and resolution of the image.
      Returns None if there's an error processing the image.
    """
    try:
        with Image.open(image_path) as img:
            width_pixels, height_pixels = img.size
            resolution = img.info.get("dpi", (0, 0))
            return height_pixels, width_pixels, resolution
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def pixels_to_cm(pixels, dpi):
    """
    Convert pixel dimensions to centimeters.

    Parameters:
    - pixels (int): Number of pixels.
    - dpi (float): Dots per inch, representing the resolution.

    Returns:
    - float or None: Converted value in centimeters. Returns None if dpi is 0.

    Explanation:
    This function converts pixel dimensions to centimeters using the provided dots per inch (dpi) value.
    If dpi is not zero, it calculates the size in inches and then converts it to centimeters using the conversion factor (1 inch = 2.54 cm).
    If dpi is zero, it returns None since conversion is not possible without resolution information.
    """
    if dpi != 0:
        inches = pixels / dpi
        cm = inches * 2.54
        return cm
    else:
        return None

def process_images_in_folder(folder_path):
    """
    Process images in a folder, extracting size, resolution, and converting dimensions to centimeters.

    Parameters:
    - folder_path (str): Path to the folder containing images.

    Returns:
    - pandas.DataFrame: A DataFrame containing information about each processed image.

    Explanation:
    This function iterates through the images in the specified folder, filters based on file extension,
    and extracts size and resolution information. It then converts the dimensions to centimeters using the
    pixels_to_cm function and prints the information for each image. Finally, it creates a DataFrame from
    the collected data and returns it.

    The DataFrame columns include:
    - 'Image': File name of the image.
    - 'Height_pixels': Height of the image in pixels.
    - 'Width_pixels': Width of the image in pixels.
    - 'Resolution': Tuple representing the resolution of the image.
    - 'Height_cm': Height of the image in centimeters.
    - 'Width_cm': Width of the image in centimeters.
    """    
    data = []
     
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            size_resolution = get_image_size_resolution(image_path) #height, width, resolution
            
            if size_resolution:
                height_pixels, width_pixels, resolution = size_resolution
                
                # In the typical structure of resolution tuples, the first value usually 
                # represents the horizontal resolution (width), and the second value 
                # represents the vertical resolution (height). Therefore, resolution[0] 
                # corresponds to the width, and resolution[1] corresponds to the height
                width_cm = pixels_to_cm(width_pixels, resolution[0])
                height_cm = pixels_to_cm(height_pixels, resolution[1])
                                
                if width_cm is not None and height_cm is not None:
                    print(f"Image: {filename}, Size: {height_pixels} x {width_pixels} pixels, Resolution: {resolution} dpi, Size in cm: {height_cm:.2f} x {width_cm:.2f} cm") 
                else:
                    print(f"Image: {filename}, Size: {height_pixels} x {width_pixels} pixels, Resolution: {resolution} dpi, Size in cm: Not available (dpi is zero)")

                if width_cm is not None and height_cm is not None:
                    data.append({
                        'Image': filename,
                        'Height_pixels': height_pixels,
                        'Width_pixels': width_pixels,
                        'Resolution': resolution,
                        'Height_cm': height_cm,
                        'Width_cm': width_cm
                    })
                else:
                    data.append({
                        'Image': filename,
                        'Height_pixels': height_pixels,
                        'Width_pixels': width_pixels,
                        'Resolution': resolution,
                        'Height_cm': None,
                        'Width_cm': None
                    })
    # Create a DataFrame from the list and return it
    return pd.DataFrame(data)

def extract_label(name):
    """
    Extract a label from a filename between the first underscore (_) and the first dot (.).

    Parameters:
    - name (str): Filename from which to extract the label.

    Returns:
    - str: Extracted label.

    Explanation:
    This function takes a filename as input and extracts the text between the first underscore (_) and the
    first dot (.). It finds the start and end indices for the extraction and returns the substring between them.
    """
    start_index = name.find('_') + 1
    end_index = name.find('.', start_index)
    return name[start_index:end_index]

# Provide the path to image folder
input_folder  = '../../data/raw/Bridgewater/'

# Provide the path to output folder
output_folder = '../../data/raw/corrections/'

# Process images in the specified folder
nr_images = len([filename for filename in os.listdir(input_folder) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Number of images in the input folder: {nr_images}")

image_data_df = process_images_in_folder(input_folder)

# create ratio variable on height/ width
image_data_df['ratio_image'] = image_data_df['Height_cm'] / image_data_df['Width_cm']
image_data_df.info()

# make the name of each photo nicer (and prepare for merging with data on paintings)
image_data_df['Image'] = image_data_df['Image'].apply(extract_label)

# External information on paintings ----
# read in data
painting_data_df = pd.read_excel("../../data/info/Index.xlsx")
painting_data_df.head()

# Joining data ----
# filter for Bridgewater paintings
paintings_photos_joined = painting_data_df[painting_data_df['folder'] == 'Raphael_Bridgewater'] \
    .drop(columns=['folder', 'id'], axis=1) \
    .merge(
        right=image_data_df,
        how='left',
        left_on='image_name',
        right_on='Image'
    ) \
    .assign(matching_ratios = lambda x: x['ratio_painting'] / x['ratio_image'])

# Display the resulting ratio comparison
# Matching Ratios describe percentages indicating the degree of agreement between the 
# dimensions of the real painting and the digital image. A value of 1 indicates a perfect
# match, i.e. a high degree of correspondence between the real painting and its digital 
# representation in terms of length and width ratios. On the other hand, a lower matching
# ratio may imply that there are discrepancies, potentially due to cropping or inaccuracies
# in the recorded dimensions of the painting or photo 
print(paintings_photos_joined[['ratio_painting', 'ratio_image', 'matching_ratios']])
   
paintings_photos_joined.info()

######################################################################################################
# if necessary, correct the size of the photos so that they match the size of the paintings
# this goes to the expense of quality loss

painting_measurement_ratios = paintings_photos_joined['ratio_painting'].tolist()
# Replace NaN values with 1
painting_measurement_ratios = [1 if math.isnan(x) else x for x in painting_measurement_ratios]
len(painting_measurement_ratios)

# height and width of photos in a list
print(paintings_photos_joined[['Height_cm', 'Width_cm']])

size_photo_values = paintings_photos_joined.apply(lambda row: (row['Height_cm'], row['Width_cm']), axis=1).tolist()
size_photo_values = [(round(x[0], 2), round(x[1], 2)) for x in size_photo_values]
print(size_photo_values)

num_images = len(painting_measurement_ratios)

# Process each photo in the input folder
for index in range(min(len(painting_measurement_ratios), len(size_photo_values), num_images)):
    filename = os.listdir(input_folder)[index]
    
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct the full path for the input and output images
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the original photo with the highest quality
        original_photo = Image.open(input_path)
      
        # Find the corresponding painting measurement ratio
        ratio = painting_measurement_ratios[index]

        # Get the original size_photo values
        original_height,original_width = size_photo_values[index]

        print(f"Original Dimensions: height {original_height} x width {original_width}")
               
        # Calculate adjustment factors for width and height
        width_adjustment_factor = round(original_width / (original_height * ratio), 2)
        height_adjustment_factor = round(original_height / (original_width * ratio), 2)

        # Choose the side with less change
        if width_adjustment_factor < height_adjustment_factor:
            adjusted_width = original_width
            adjusted_height = round(original_width * ratio, 2)
        else:
            adjusted_height = original_height
            adjusted_width = round(original_height * ratio, 2)
        
        print(f"Adjusted Dimensions: height {adjusted_height} x x width {adjusted_width}")
        
        # Check if the adjusted ratio matches the painting measurement ratio
        adjusted_ratio = round(adjusted_height / adjusted_width,2)

        if adjusted_ratio == ratio:
            print("Adjusted ratio matches the painting measurement ratio.")
        
        else:
            print("Warning: Adjusted ratio does not match the painting measurement ratio.")
            print(f"Painting measurement ratio: {ratio}, Adjusted ratio: {adjusted_ratio}")

        # Round the adjusted dimensions to two decimal places for precision
        adjusted_width = round(adjusted_width, 2)
        adjusted_height = round(adjusted_height, 2)

        # Resize the photo with a higher-quality interpolation method
        adjusted_photo = original_photo.resize((int(adjusted_width), int(adjusted_height)), Image.ANTIALIAS)

        # Save the adjusted photo in JPEG format with higher quality
        adjusted_photo.save(output_path, quality=100,subsampling=0)
                
        # Plot the adjusted image using PIL
        plt.figure(figsize=(4, 4))  # Adjust the figsize as needed

        # Adjusted image
        plt.imshow(adjusted_photo)
        plt.title('Adjusted Image')
        plt.axis('off')

        plt.show()

print("Adjustment and saving complete.")