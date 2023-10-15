from PIL import Image
import numpy as np
import tensorflow as tf
import random
import config

height = config.height
width = config.width
n_clusters = config.n_clusters
p = config.p
n = config.n
alpha = config.alpha

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [height, width])
    image /= 255.0  # normalize to [0,1] range
    return image

# This function does all preprocessing needed
def preprocess(image_path):
  content_img = Image.open(image_path).resize((width, height))

  content_arr = np.asarray(content_img, dtype='float32')
  if len(content_arr.shape) < 3:
    content_arr = np.array([content_arr, content_arr, content_arr]).reshape(width, height, 3)
  
  content_arr = np.expand_dims(content_arr, axis=0)
  return content_arr

def sampling_inner(clusters, arr):
  a = np.random.rand(1)[0]
  x = (a-0.00001) // 0.25
  subset = arr[int(x)]
  
  b = np.random.rand(1)[0]
  if(b> 0.75):
    subset = np.random.randint(n_clusters)
    while(len(np.where(clusters == subset)) == 0):
      subset = np.random.randint(n_clusters)
  return subset
  
def sampling_outer(clusters, summ, prev_votes):
  a = summ * np.random.rand(1)[0]
  sub = 0
  add = p
  for i in range(n):
    if(a < add):
      sub = sampling_inner(clusters, prev_votes[:,i])
      return sub      
    else:
      add+=p*alpha**i
  return sub

def iter(clusters, summ, prev_votes, uvote):
  ##uvote is a array containing 4 subset number
  curr = [0]*4
  
  temp = np.zeros((4,n))
  temp[:,1:]=prev_votes[:,:-1]
  temp[:,0]= uvote
  prev_votes = temp
  for i in range(4):
    curr[i] = sampling_outer(clusters, summ, prev_votes)
  return curr, prev_votes

def cluster_to_img(clusters, image_paths, cluster):
  arr = np.where(clusters == cluster)[0]
  try:
    img_idx = random.choice(arr)
  except:
    # breakpoint()
    img_idx = random.choice(range(len(image_paths)))
  return preprocess(image_paths[img_idx]).astype(int)[0]