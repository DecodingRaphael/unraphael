
!pip install git+https://github.com/openai/CLIP.git
!pip install open_clip_torch
!pip install sentence_transformers

# libraries ----
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image

## reading data (color format)
image1 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 1)
image2 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 1)

# plot the original
plt.imshow(image1) 
plt.imshow(image2)

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2):
    #orig_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    #copy_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    orig_img = image1
    copy_img = image2
    img1 = imageEncoder(orig_img)
    img2 = imageEncoder(copy_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

print(f"similarity Score: ", round(generateScore(image1, image2), 2))
#similarity Score: 90.69

# The similarity between images is computed based on the cosine similarity  of these feature vectors. 
# To improve the accuracy, we can preprocess the images


