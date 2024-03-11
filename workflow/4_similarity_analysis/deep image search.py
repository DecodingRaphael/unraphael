# https://github.com/TechyNilesh/DeepImageSearch

from DeepImageSearch import Load_Data, Search_Setup

# Load images from a folder
image_list = Load_Data().from_folder(['../data/outp/'])

image_list

st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=100)

# Index the images
st.run_index()

# Get metadata
metadata = st.get_image_metadata_file()

# Add new images to the index
#st.add_images_to_index(['image_path_1', 'image_path_2'])

# Get similar images
st.get_similar_images(image_path=image_list[0], number_of_images=5)

# Plot similar images
st.plot_similar_images(image_path=image_list[0], number_of_images=10)
#st.plot_similar_images(image_path="../data/outp/output_0_Edinburgh_Nat_Gallery.jpg", number_of_images=5)

# Update metadata
metadata = st.get_image_metadata_file()