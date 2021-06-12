import torch
from torch import optim
import time
import os
from model.UNet_model import UNet
from loader.data_loader import SlicesDataset
from torch.utils.data import DataLoader
import numpy as np
from trainer.inference import UNetInferenceAgent
from utils.utils import Dice3d, Jaccard3d


class UNetExperiment:
  """
  Basic life cycle for segmentation
  """
  def __init__(self, config, split, dataset):
    self.n_epochs = config.n_epochs
    self.split = split
    self._time_start = ""
    self._time_end = ""
    self.epoch = 0 
    self.name = config.name

    #create output folders
    dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
    self.out_dir = os.path.join(config.test_results_dir, dirname)
    os.makedirs(self.out_dir, exist_ok=True)

    #Create data loaders
    self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=0)
    
    self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                                 batch_size = config.batch_size,
                                 shuffle=True,
                                 num_workers=0)
    
    # We will access volumes directly for testing
    self.test_data = dataset[split["test"]]

    # Check whether CUDA is available
    if not torch.cuda.is_available():
      print("WARNING: No CUDA device is found. This may take significantly longer!")
    
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    self.model = UNet(num_classes=3)
    #self.model = UNet(num_classes=3, use_dropout=True)
    self.model.to(self.device)

    #We are using standard cross_entropy loss
    self.loss_function = torch.nn.CrossEntropyLoss()

    #Optimizer
    self.optimizer = optim.Adam(self.model.parameters(),
                                lr = config.learning_rate)
    
    # Scheduler helps us update learning rate automatically 
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min')
    

  def train(self):
    """
    This method is executed once per epoch and takes care of 
    model weight update cycle
    """
    print(f"Training epoch {self.epoch} ...")
    self.model.train()
    loss_list = []
    #Loop over our minibatches
    for i, batch in enumerate(self.train_loader):
      
      batch_imgs = batch["image"]
      batch_segs = batch["seg"]
      data = batch_imgs.to(self.device).type(torch.float)
      target = batch_segs.to(self.device).type(torch.long)

      prediction = self.model(data)

      loss = self.loss_function(prediction, target[:,0,:,:])

      #Adding L2 regularization to the loss
      l2_lambda = 0.001
      l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters() if p.requires_grad == True)
      loss = loss + l2_lambda*l2_norm 
    
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      loss_list.append(loss.item())

      #if (i % 10) == 0:
        # Output to console on every 10th batch
        #print(f"Epoch: {self.epoch} Train loss:{loss/len(self.train_loader)}, {100*(i+1)/len(self.train_loader):.1}% complete") 
    #print("\nTraining comple")
    return loss_list
  
  def validate(self):
    """
    This method runs validation cycle, using same metrics as 
    Train method. Note that model needs to be switched to eval mode 
    and no_grad needs to be called so that gradients do not 
    propagate.
    """
    print(f"Validating epoch {self.epoch}...")

    self.model.eval()
    loss_list = []

    with torch.no_grad():
      for i, batch in enumerate(self.val_loader):
        batch_imgs = batch["image"]
        batch_segs = batch["seg"]
        data = batch_imgs.to(self.device).type(torch.float)
        target = batch_segs.to(self.device).type(torch.long)
        prediction = self.model(data)
        #prediction_softmax = F.softmax(prediction, dim=1)
        #prediction_softmax = prediction
        loss = self.loss_function(prediction, target[:,0,:,:])
        #print(f"Batch {i}. Data shape {data.shape} Loss {loss.item()/len(self.val_loader)}")
        loss_list.append(loss.item())
    
    self.scheduler.step(np.mean(loss_list))
    print(f"Validation complete")
    return loss_list

  
  def save_model_parameters(self):
    """
    Saves model parameters to a file in results directory
    """
    path = os.path.join(self.out_dir, "model.pth")
    torch.save(self.model.state_dict(), path)

  def load_model_parameters(self, path=''):
    """
    Loads model parameters from a supplied path or a results directory
    """
    if not path:
      model_path = os.path.join(self.out_dir, "model.pth")
    else:
      model_path = path
    
    if os.path.exists(model_path):
      self.model.load_state_dict(torch.load(model_path))
    else:
      raise Exception(f"Could not find path {model_path}")

  def run_test(self):
    """
    This runs test cycle on the test dataset.
    Note that process and evaluations are quite different.
    Here we are computing a lot more metrics and returning a 
    dictionary that could later be persisted on a JSON.
    """
    print("Testing...")
    self.model.eval()
    inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

    out_dict = {}
    out_dict["volume_stats"] = []
    dc_list = []
    jc_list = []

    for i, x in enumerate(self.test_data):
      pred_label = inference_agent.single_volume_inference(x["image"])
      segmentation = x["seg"]
      dc = 0.0
      jc = 0.0
      num_classes = 3
      for index in range(num_classes):
        y_true = segmentation
        y_pred = pred_label[:,index,:,:].reshape(segmentation.shape)
        dc += Dice3d(y_true, y_pred)
        jc += Jaccard3d(y_true, y_pred)
      
      dc_list.append(float(dc) / float(num_classes))
      jc_list.append(float(jc) / float(num_classes))

      out_dict["volume_stats"].append({
          "filename":x['filename'],
          "dice":dc,
          "jaccard":jc
      })

      print(f"{x['filename']}; Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")
      print(f"{x['filename']}; Jaccard {jc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

    out_dict["overall"] = {
            "mean_dice": np.mean(dc_list),
            "mean_jaccard": np.mean(jc_list)}

    print("\nTesting complete.")
    return out_dict

  def run(self):
    """
    Kicks off train cycle and writes model parameter file at the end
    """
    self._time_start = time.time()
    print("Experiment started.")
    result = dict()
    result['epoch'] = []
    result['train_loss']=[]
    result['val_loss'] = []
    #Iterate over epochs
    for self.epoch in range(self.n_epochs):
      train_loss_list = self.train()
      val_loss_list = self.validate()
      result['epoch'].append(self.epoch)
      result['train_loss'].append(np.mean(train_loss_list))
      result['val_loss'].append(np.mean(val_loss_list))
      print(f"Epoch: {self.epoch + 1}, Train error: {np.mean(train_loss_list)}, Validation loss: {np.mean(val_loss_list)}")

    
    #Save model for inferencing
    self.save_model_parameters()

    self._time_end = time.time()
    print(f"Run complete. Total tiem:{time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
    return result