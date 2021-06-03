# A Configuration class for image and model parameters 
class Config:
  """
  Holds configuration parameters
  """
  def __init__(self):
    self.name = "Basic_unet"
    self.root_dir = 'root_dir'
    self.n_epochs = 50
    self.learning_rate = 0.0001
    self.batch_size = 16
    self.patch_size= 64
    self.test_results_dir = 'result_dir'