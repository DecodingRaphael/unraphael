# https://github.com/DatumLearning/Image-search-engine/blob/main/video_front_end.py
# https://github.com/DatumLearning
# https://github.com/DatumLearning/Streamlit_full_course


import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist

@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    processed_images = np.load("processed_images.npy", allow_pickle=True)
    return all_vecs, all_names, processed_images

vecs, names, processed_images = read_data()

_ , fcol2 , _ = st.columns(3)

scol1 , scol2 = st.columns(2)

ch = scol1.button("Start / change")
fs = scol2.button("find similar")

if ch:
    random_name = names[np.random.randint(len(names))]
    print("RANDOM NAME ")
    print(random_name)
    fcol2.image(Image.open( random_name))
    st.session_state["disp_img"] = random_name
    st.write(st.session_state["disp_img"])
    print("------------------------------------")
    print(st.session_state["disp_img"])

if fs:
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    
    print("----------idx------------------------------------")
    print(idx)
    
    target_vec = vecs[idx]
    fcol2.image(Image.open(st.session_state["disp_img"]))
    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]
    
    # Display similar images with names and similarity scores below each image
    for i, col in enumerate([c1, c2, c3, c4, c5]):
        similar_idx = top5[i]
        similar_name = names[similar_idx]

        # Find the correct similarity score using processed_images
        for score, image_id1, image_id2 in processed_images:
            if image_id1 == idx and image_id2 == similar_idx:
                similarity_score = round(score * 100, 2)
                break

        col.image(Image.open(similar_name), caption=f"Name: {similar_name}", use_column_width=True)
        col.write(f"Similarity Score: {similarity_score}")
                  
    # Display similar images with names below each image
    #c1.image(Image.open(names[top5[0]]), caption=f"Name: {names[top5[0]]}", use_column_width=True)
    #c2.image(Image.open(names[top5[1]]), caption=f"Name: {names[top5[1]]}", use_column_width=True)
    #c3.image(Image.open(names[top5[2]]), caption=f"Name: {names[top5[2]]}", use_column_width=True)
    #c4.image(Image.open(names[top5[3]]), caption=f"Name: {names[top5[3]]}", use_column_width=True)
    #c5.image(Image.open(names[top5[4]]), caption=f"Name: {names[top5[4]]}", use_column_width=True)
                    
    #c1.image(Image.open(names[top5[0]]))
    #c2.image(Image.open(names[top5[1]]))
    #c3.image(Image.open(names[top5[2]]))
    #c4.image(Image.open(names[top5[3]]))
    #c5.image(Image.open(names[top5[4]]))