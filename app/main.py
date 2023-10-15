import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from utils import load_and_preprocess_image, cluster_to_img, iter
from glob import glob
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
import config

if __name__ == "__main__":

  path = config.ds_path
  n_samples = config.n_samples
  n_clusters = config.n_clusters
  p = config.p
  n = config.n
  alpha = config.alpha
  height, width = config.height, config.width
  image_paths = glob(path)

  try:
    embeddings = np.loadtxt("app/embeds.txt")
  except:
    # VGG16 is used as pretrained CNN model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, 3))
    model = keras.Sequential()
    
    for layer in vgg_model.layers:
      model.add(layer)
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, name='embeddings'))

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    batch_size = 16
    image_ds = image_ds.batch(batch_size)
    image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    embeddings = np.concatenate([model.predict(images) for images in image_ds], axis=0)

    np.savetxt("app/embeds.txt", embeddings)

  km = KMeans(n_clusters=n_clusters).fit(embeddings)
  clusters = np.zeros(n_samples)

  for cluster in range(n_clusters):
    for image in np.where(km.labels_ == cluster):
      clusters[image] = cluster

  if 'summ' not in st.session_state:
    st.session_state.summ=0
    for i in range(n):
      st.session_state.summ += p*(alpha**i)

  if 'prev_votes' not in st.session_state:
    st.session_state.prev_votes = np.floor(n_clusters*np.random.rand(4, n)-0.000001)
    st.session_state.boundary = np.zeros(n)

  if 'uvote' not in st.session_state:
    st.session_state.uvote = np.random.randint(n_clusters, size=4)

  st.title("Apparel Recommender")

  col1, col2, col3 = st.columns(3)
  options = ["Like", "Dislike"]

  col1.image(cluster_to_img(clusters, image_paths, st.session_state.uvote[0]), "First Apparel", width=200)
  col2.image(cluster_to_img(clusters, image_paths, st.session_state.uvote[1]), "Second Apparel", width=200)
  col1.image(cluster_to_img(clusters, image_paths, st.session_state.uvote[2]), "Third Apparel", width=200)
  col2.image(cluster_to_img(clusters, image_paths, st.session_state.uvote[3]), "Fourth Apparel", width=200)

  with col3.form("my_form", clear_on_submit=True):

    st.write("### Choose what you like! :sunglasses")
    ip = [0, 0, 0, 0]
    ip[0] = 1 if st.radio("Liked first apparel?", options, key=1) == options[0] else -1
    ip[1] = 1 if st.radio("Liked second apparel?", options, key=2) == options[0] else -1
    ip[2] = 1 if st.radio("Liked third apparel?", options, key=3) == options[0] else -1
    ip[3] = 1 if st.radio("Liked forth apparel?", options, key=4) == options[0] else -1

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit", type="primary")

  if submitted:
    submitted = False
    st.session_state.uvote = np.array(ip) * st.session_state.uvote
    neg = (st.session_state.uvote < 0).sum()

    if neg == 4:
      uvote = np.random.randint(n_clusters,size=4)
    elif neg == 3:
      temp = st.session_state.uvote[st.session_state.uvote > 0][0]
      uvote = np.array([temp, temp, temp, temp])
    elif neg == 2:
      temp1 = st.session_state.uvote[st.session_state.uvote > 0][0]
      temp2 = st.session_state.uvote[st.session_state.uvote > 0][1]
      uvote = np.array([temp1, temp2, temp1, temp2])
    elif neg == 1:
      temp1 = st.session_state.uvote[st.session_state.uvote > 0][0]
      temp2 = st.session_state.uvote[st.session_state.uvote > 0][1]
      temp3 = st.session_state.uvote[st.session_state.uvote > 0][2]
      uvote = np.array([temp1, temp2, temp3, temp3])

    st.session_state.uvote, st.session_state.prev_votes = iter(clusters, st.session_state.summ, st.session_state.prev_votes, st.session_state.uvote)