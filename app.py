from flask import Flask, render_template, request
import os
import numpy as np
import fnmatch
import torch
import faiss
import pickle
import random
from transformers import CLIPTokenizerFast, CLIPImageProcessor, CLIPModel
import nltk
from nltk.corpus import stopwords
from translate import Translator

app = Flask(__name__)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")


def weighted_rand(spec):
    i, sum = 0, 0
    r = random.random()
    for i in spec:
        sum += spec[i]
        if r <= sum: return int(i)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        index, np_embeddings, path_to_index, index_to_path = pickle.load(f)
    return index, np_embeddings, path_to_index, index_to_path


faiss_index, np_embeddings, path_to_index, index_to_path = load_data('data.pkl')


def ann_search(embedding, k=5):
    k = min(k, faiss_index.ntotal)
    embedding = np.array(embedding).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(embedding, k)
    return distances, indices


def find_images_in_folder(folder_path, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.gif')):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for ext in image_extensions:
            for filename in fnmatch.filter(files, ext):
                full_filepath = os.path.join(root, filename).split('static/')[1]
                image_files.append((full_filepath, filename))
    return image_files


def search(embedding, k=5):
    embedding = np.array(embedding).astype('float32').reshape(1, -1)
    return faiss_index.search(embedding, k)


def find_nearest_paths(input_path, k=100):
    input_path = os.path.join(os.getcwd(), 'static/') + input_path
    input_embedding = np_embeddings[path_to_index[input_path]]
    distances, indexes = search(input_embedding, k)
    nearest_paths = [index_to_path[idx] for idx in indexes.flatten()]
    return [p.split('static/')[1] for p in nearest_paths]


def embed_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # Normalize
        return text_features.numpy().tolist()[0]


def embed_texts(texts):
    inputs = text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # Normalize
        return text_features.numpy().tolist()


def extract_key_terms(query):
    stop_words = ['in', 'the', 'a', 'an', 'on', 'at', 'of', 'with', 'by', 'for', 'from', 'to', 'and', 'or', 'but']
    terms = [word for word in query.split() if word.lower() not in stop_words]
    return terms


def translate_ru_to_en(text):
    translator = Translator(from_lang="ru", to_lang="en")
    translation = translator.translate(text)
    return translation


@app.route('/search', methods=['GET'])
def query_search():
    query = translate_ru_to_en(request.args.get('query'))
    accuracy = request.args.get('accuracy', '50')  # Default to 50%
    threshold = float(accuracy) / 100  # Convert to [0,1] scale

    query_embedding = embed_text(query)
    distances, indexes = ann_search(query_embedding, 50)

    key_terms = extract_key_terms(query)
    if key_terms:
        term_embeddings = np.array(embed_texts(key_terms))
    else:
        term_embeddings = np.array([])

    filtered_images = []
    image_accuracies = {}

    for idx in indexes.flatten():
        if idx == -1:
            continue
        image_path = index_to_path[idx].split('static/')[1]
        image_embedding = np_embeddings[idx]

        if key_terms:
            similarities = np.dot(term_embeddings, image_embedding)
            acc_percentages = similarities * 100  # Convert to percentage
            if np.all(similarities >= threshold):
                filtered_images.append(image_path)
                image_accuracies[image_path] = {term: acc for term, acc in zip(key_terms, acc_percentages)}

    return render_template('similar_images.html', images=filtered_images, image_accuracies=image_accuracies,
                           weighted_rand=weighted_rand)


@app.route('/similar-images')
def similar_images():
    image_path = request.args.get('image_path')
    similar_image_paths = find_nearest_paths(image_path)
    return render_template('similar_images.html', images=similar_image_paths, weighted_rand=weighted_rand)


@app.route('/nearest_images')
def nearest_images():
    input_path = request.args.get('path')
    nearest_image_paths = find_nearest_paths(input_path)
    return render_template('index.html', images=nearest_image_paths)


@app.route('/')
def index():
    all_image_paths = [p.split('static/')[1] for p in index_to_path.values()]
    return render_template('similar_images.html', images=all_image_paths, weighted_rand=weighted_rand)


if __name__ == '__main__':
    app.run(debug=True)
