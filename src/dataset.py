import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from numpy.lib.npyio import save

def unpickle_file(file):
  with open(file, 'rb') as fo:
    dict_data = pickle.load(fo, encoding='bytes')
  return dict_data


def rgb_from_1darray(array):
  r = array[:1024]
  g = array[1024:2048]
  b = array[2048:3072]

  rgb = list()

  for i in range(1024):
    rgb.append( np.asarray([r[i], g[i], b[i]]) )
        
  np_img = np.asarray(rgb)
  np_img_reshaped = np.reshape(np_img, (32,32,3))

  return np_img_reshaped


def images_from_batch_file():
  files = list(Path('../cifar-10-batches-py').glob('**/*data_batch_*'))
  data_rgb = list()
  
  for f in files:
    print(f.name)

    data_dict = unpickle_file(f)
    
    data_1d = data_dict[b'data']
    labels = data_dict[b'labels']

    for array_data in data_1d:
      data_rgb.append(rgb_from_1darray(array_data))

    save_images(data_rgb, labels)

    break


def save_images(images, labels):
  count = 0
  
  for im_array, label in zip(images, labels):
    count += 1
    Image.fromarray(im_array).save(f'../img/cifar/{label}/{count}.png')


if __name__ == '__main__':
  images_from_batch_file()