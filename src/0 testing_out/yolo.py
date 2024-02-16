# https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9
from ultralytics import YOLO
from PIL import Image
from PIL import ImageDraw

model = YOLO("yolov8m-seg.pt")

results = model.predict("cat_dog.jpg")
results = model.predict("raphael.png")
results = model.predict("baby2.jpg")

result = results[0]

masks = result.masks
len(masks)

mask1 = masks[1]

mask = mask1.data[0].numpy()
polygon = mask1.xy[0]

mask_img = Image.fromarray(mask,"I")
mask_img

polygon

img = Image.open("baby2.jpg")
draw = ImageDraw.Draw(img)
draw.polygon(polygon,outline=(0,255,0), width=5)
img

mask2 = masks[1]
mask = mask2.data[0].numpy()
polygon = mask2.xy[0]
mask_img = Image.fromarray(mask,"I")
mask_img

draw.polygon(polygon,outline=(0,255,0), width=5)
img

