# sources to read!
# https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/
# https://docs.ultralytics.com/guides/isolating-segmentation-objects/#isolate-with-black-pixels-sub-options
# https://stackoverflow.com/questions/76870102/how-to-save-segmented-images-or-masks-using-segment-anything-model-sam
# https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9

from PIL import ImageDraw
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# segmentation ----
model = YOLO("yolov8x-seg.pt")  # for segmentation (huge)
#model = YOLO("yolov8m-seg.pt")  # for segmentation (medium)

# detection ----
#model = YOLO("yolov8x.pt")     # for detection (huge)

# classification ----
#model = YOLO("yolov8x-cls.pt") # for classification (huge)

# pose ----
model = YOLO("yolov8x-pose.pt") # for classification (huge)


# Run inference
#res = model.predict("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg", save = True, save_txt=True)
res = model.predict("../../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg", save = True, save_txt=True)

#img_segmented = cv2.imread("runs/segment/predict2/output_8_London_OrderStJohn.jpg")
#rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
#plt.imshow(rgb, cmap = plt.cm.Spectral)

result = res[0] # get first image result
len(result.boxes)
len(result.masks) # 2 objects detected

# multiple segmentation masks in a single binary image (baby and madonna) ----
img = (res[0].cpu().masks.data[0].numpy() * 255).astype("uint8")

height,width = img.shape
masked = np.zeros((height, width), dtype="uint8")

num_masks = len(res[0].cpu().masks.data)

for i in range(num_masks):
    masked = cv2.add(masked,(res[0].cpu().masks.data[i].numpy() * 255).astype("uint8"))

# Convert NumPy array to PIL Image
mask_img = Image.fromarray(masked,"L")
mask_img

# Save the combined mask image
mask_img.save("../../data/interim/b_w/0.jpg")

    

# first object (baby) ----
mask1 = result.masks[0]
mask = mask1.data[0].numpy()
polygon = mask1.xy[0]

mask_img = Image.fromarray(mask,"I")
mask_img

# draw polygon of baby on image
img_baby = Image.open("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")
draw = ImageDraw.Draw(img_baby)
draw.polygon(polygon,outline=(0,255,0), width = 3)
img_baby

# second object (madonna) ----
mask2 = result.masks[1]
mask = mask2.data[0].numpy()
polygon = mask2.xy[0]
mask_img = Image.fromarray(mask,"I")
mask_img

# draw polygon of madonna and baby on image
img_madonna = Image.open("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")
draw = ImageDraw.Draw(img_madonna)
draw.polygon(polygon,outline=(0,255,0), width=3)
img_madonna

# iterate detection results 
for idx, r in enumerate(res):  # Iterate over the indices and results
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem

    # iterate each object contour 
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        # Create binary mask
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask 
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Isolate object with transparent background (when saved as PNG)
        isolated = np.dstack([img, b_mask])
        
        x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        
        # Ensure the coordinates are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        
        print(f'Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}')
        
        #iso_crop = isolated[y1:y2, x1:x2]
        
        # Save isolated object to file
        _ = cv2.imwrite(f'../../data/interim/segments/{img_name}_segment{idx}_{label}-{ci}.png', isolated)
        
        print(f'Original Image Dimensions: {img.shape}')
        print(f'Isolated Image Dimensions: {isolated.shape}')


########################################################################################
#  apply the segmentation and mask combination to all images in a folder.

# Path to the folder containing input images
input_folder_path = "../../data/interim/aligned"

# Path to the folder where you want to save the combined mask images
output_folder_path = "../../data/interim/masks"

# Iterate through each image in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
        # Predict using YOLO model
        img_path = os.path.join(input_folder_path, filename)
        res = model.predict(img_path, save=True, save_txt=True)

        # Combine segmentation masks into a single binary image
        img = (res[0].cpu().masks.data[0].numpy() * 255).astype("uint8")
        height, width = img.shape
        masked = np.zeros((height, width), dtype="uint8")

        num_masks = len(res[0].cpu().masks.data)

        for i in range(num_masks):
            masked = cv2.add(masked, (res[0].cpu().masks.data[i].numpy() * 255).astype("uint8"))

        # Convert NumPy array to PIL Image
        mask_img = Image.fromarray(masked, "L")

        # Save the combined mask image
        output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_combined_mask.jpg")
        mask_img.save(output_path)

print("Segmentation and mask combination for all images completed.")