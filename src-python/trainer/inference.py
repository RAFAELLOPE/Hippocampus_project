import numpy as np
import torch
from model.UNet_model import UNet
import torch.nn.functional as F



class UNetInferenceAgent:
  """
  Stores model and parameters and some methods to handle inferencing
  """
  def __init__(self, parameter_file_path='', model=None, device='cpu', patch_size=64):
    self.model = model
    self.patch_size = patch_size
    self.device = device

    if model is None:
      self.model = UNet(num_classes = 3)
    
    if parameter_file_path:
      self.model.load_state_dict(torch.load(parameter_file_path,
                                            map_location=self.device))
      
    self.model.to(device)
  
  def single_volume_inference_unpadded(self, volume):
    """
    Runs inference on a single volume of arbitrary patch size,
    padding it ot the conformat size first
    
    Arguments:
            volume {Numpy array} -- 3D array representing the volume

    Returns:
            3D NumPy array with prediction mask
    """
    new_shape = (volume.shape[0], self.patch_size, self.patch_size)
    reshaped_volume = np.full(new_shape, np.min(volume))
    assert new_shape >= volume.shape

    x_max = volume.shape[0]
    y_max = volume.shape[1]
    z_max = volume.shape[2]

    reshaped_volume[:x_max, :y_max, :z_max] = volume
        
    self.model.eval()
    pred_slices = []
      
    for s in range(reshaped_volume.shape[0]):
      slc = torch.tensor(reshaped_volume[s])
      slc = slc.unsqueeze(0).unsqueeze(0).to(self.device).type(torch.float)
      pred = self.model(slc)
      prediction = torch.argmax(F.softmax(pred, dim=1), dim=1)
      prediction_numpy = prediction.cpu().detach().numpy()
      pred_slices.append(prediction_numpy[0])
      pred_volume = np.array(pred_slices)
    
    return pred_volume

  def single_volume_inference(self, volume):
    """
    Runs inference on a single volume of conformant patch size
    """
    self.model.eval()
    # Assuming volume is a numpy array of shape [X Y Z] and we need to slice X axis
    slices = []
    for s in range(volume.shape[0]):
      slc = torch.tensor(volume[s])
      slc = slc.unsqueeze(0).unsqueeze(0).to(self.device).type(torch.float)
      pred = self.model(slc)
      prediction = F.softmax(pred, dim=1)
      prediction_numpy = np.squeeze(prediction.cpu().detach().numpy())
      slices.append(prediction_numpy)
    return (np.array(slices))