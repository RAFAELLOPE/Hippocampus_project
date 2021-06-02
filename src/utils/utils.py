import os
from os import listdir
from os.path import isfile, join
from medpy.io import load
import numpy as np
import nibabel as nib

def med_reshape(image, new_shape):
  """
  This function reshapes 3D data to new dimension paddiing with zeros
  and leaving the content in the top-left corner

  Arguments:
    image {array} -- 3D array of pixel data
    new_shape {3-tuple} -- expected output shape
  
  Returns:
  3D array of desired shape, padded with zeroes
  """

  reshaped_image = np.full(new_shape, np.min(image))

  assert new_shape >= image.shape

  x_max = image.shape[0]
  y_max = image.shape[1]
  z_max = image.shape[2]

  reshaped_image[:x_max, :y_max, :z_max] = image

  return reshaped_image



def LoadHippocampusData(root_dir, y_shape, z_shape):
  '''
  This function loads our dataset from disk into memory,
  reshaping output to common size.

  Arguments:
    volume {Numpy array} -- 3D array representing the volume

  Returns:
    Array of dictionaries with data stored in seg and imag fields as
    Numpy arrays of sahpe [AXIAL_WIDTH, Y_SAHPE, Z_SHAPE]
  '''
  image_dir = os.path.join(root_dir, 'imagesTr')
  label_dir = os.path.join(root_dir, 'labelsTr')

  images = [f for f in os.listdir(image_dir) if (os.path.isfile(join(image_dir, f))) and f[0] != '.']

  out = []
  for f in images:
    # We would benefit here from mmap load  method here if dataset does not fit into
    # memory. Images are loaded here using MedPy's load method
    image, _ = load(os.path.join(image_dir, f))
    label, _ = load(os.path.join(label_dir, f))
    normalized_img = np.zeros(image.shape, dtype=np.single)

    # Normalize image to [0,..,1]

    for slice_ix in range(image.shape[0]):
      slc = image[slice_ix].astype(np.single)/np.max(image[slice_ix])
      #mean = np.mean(image[slice_ix].astype(np.single))
      #std = np.std(image[slice_ix].astype(np.single))
      #slc = (image[slice_ix].astype(np.single) - mean)/std
      normalized_img[slice_ix] = slc

    # We need to reshape data since CNN tensors that represent minibatches.
    # In our case will be stacks of slices and stacks need to be of the same size.

    reshaped_image = med_reshape(normalized_img, new_shape=(normalized_img.shape[0], y_shape, z_shape))
    reshaped_label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

    out.append({"image":reshaped_image, "seg":reshaped_label, "filename":f})
  
  print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
  return np.array(out)


def Dice3d(a, b):
  '''
  This will compute Dice Similarity coefficient for two 3-dimensional volumes.
  Volumes are expected to be of the same size. We are expecting binary masks
  0's are treated as background and anything else is counted as data.
  '''
  if len(a.shape) != 3 or len(b.shape) != 3:
    raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

  if a.shape != b.shape:
    raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")
  
  #Since we have two classes we will do it separately
  smooth = 0.0001
  intersection = np.sum(a*b)
  volumes = np.sum(a**2) + np.sum(b**2)
  dice = (2.*float(intersection)) / (float(volumes) + smooth)
  return dice

    

def Jaccard3d(a, b):
  '''
  This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes.
  Volumes are expected to be of the same size. We are expecting binary masks.
  0's are treated as backgrouond and anything else is counted as data
  '''
  if len(a.shape) != 3 or len(b.shape) != 3:
    raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

  if a.shape != b.shape:
    raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")
  
  smooth = 0.0001
  intersection = np.sum(a*b)
  union = np.sum(a**2) + np.sum(b**2) - intersection
  jaccard = (float(np.abs(intersection))) / (float(np.abs(union)) + smooth)
  return jaccard