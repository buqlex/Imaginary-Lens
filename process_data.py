import fnmatch
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
import pickle
from transformers import CLIPImageProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


def find_images_in_folder(folder_path, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.gif')):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for ext in image_extensions:
            for filename in fnmatch.filter(files, ext):
                full_filepath = os.path.join(root, filename)
                image_files.append((full_filepath, filename))
    return image_files


pwd = os.getcwd()
relative_path = os.path.join(pwd, 'static/')
paths = find_images_in_folder(relative_path)


def embed_image(image_path_or_paths):
    if isinstance(image_path_or_paths, str):
        image_paths = [image_path_or_paths]
    else:
        image_paths = image_path_or_paths

    pil_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = image_processor(images=pil_images, return_tensors="pt")
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # Normalize embeddings
    return embeddings.numpy().tolist()


full_paths = [p[0] for p in paths]

processed_paths = []
embeddings = []


def write_data():
    np_embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(np_embeddings.shape[1])
    index.add(np_embeddings)

    path_to_index = {p: i for i, p in enumerate(processed_paths)}
    index_to_path = {i: p for i, p in enumerate(processed_paths)}

    with open('data.pkl', 'wb') as f:
        pickle.dump((index, np_embeddings, path_to_index, index_to_path), f)


i = 0
batch_size = 10
total_images = len(full_paths)

while i < total_images:
    batch = full_paths[i:i + batch_size]
    embeddings += embed_image(batch)
    processed_paths += batch
    i += batch_size
    if i % 100 == 0:
        write_data()
write_data()
