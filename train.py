import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, Dataset

# Set up config variables
config = {
    'data_path': './data', # directory path of dataset,
}

# Import MNIST digit dataset
def importMnistDataset(root, train, transform=[]):
  # it will try to download the dataset if dataset is not found under the root directory 
  return torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform) 

# ImportMnistDataset may fail to download because of HTTP error (happens a lot recently).
# We can manually download the datasets with the following comments,
# If data_path value is changed in config, update it here as well.
!wget -nc www.di.ens.fr/~lelarge/MNIST.tar.gz -P ./tmp
!mkdir -p ./data/
!tar -zxvf ./tmp/MNIST.tar.gz -C ./data/

# Test dataset contains 10,000 images
# Train dataset contains 60,000 images
train_set = importMnistDataset(root=config['data_path'], train=True)
test_set = importMnistDataset(root=config['data_path'], train=False)
print(train_set)
print(test_set)
del test_set, train_set

class Noise(object):
    def __init__(self, drop_probability=0):
        self.drop_probability = drop_probability
        
    def __call__(self, tensor):
        n = torch.from_numpy(np.random.choice([0, 1], size=tensor.size(), p=[self.drop_probability, 1-self.drop_probability])) 
        return tensor * n
    
    def __repr__(self):
        return self.__class__.__name__ + '(drop_probability={0})'.format(self.drop_probability)

def generateTransform(drop_probability):
  if drop_probability is not None and drop_probability > 0:
    trans_noise = transforms.Compose([
                              transforms.ToTensor(),
                              Noise(drop_probability)
                              ])
    return trans_noise
  else:
    return transforms.Compose([transforms.ToTensor()])

train_set = importMnistDataset(config['data_path'], True, generateTransform(drop_probability=0.7))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)

train_image_batch, classe_set = iter(train_loader).next()

print(f'train_loader contains {len(train_loader)} batches of data.')
print(f'train_image_batch has shape {train_image_batch.shape},')
print('where 64 is the number of images in a batch, 1 is the number of image channels (1 for grayscale image),\
 28X28 stands for WxH (width and height of a single image).')

def show_gray_digits(image_set, row=2, col=3):
  image_set = image_set.detach().numpy()

  for i in range(row*col):
    plt.subplot(row, col, i+1)
  plt.show()

show_gray_digits(train_image_batch, 2, 3)
print(classe_set[0:6])
del train_image_batch, classe_set, train_set, train_loader


def load_data(path, drop_probability, split_ratio, batch_size):  
  train_set = importMnistDataset(path, True, generateTransform(0))
  train_set = torch.utils.data.Subset(train_set, list(range(1, 800)))
  train_set_noise = importMnistDataset(path, True, generateTransform(drop_probability))
  train_set_noise = torch.utils.data.Subset(train_set_noise, list(range(1, 800)))
  test_set = importMnistDataset(path, False, generateTransform(0))
  test_set_noise = importMnistDataset(path, False, generateTransform(drop_probability))
  train_set = PairDataset(train_set, train_set_noise)
  test_set = PairDataset(test_set, test_set_noise)                             
  train_set, val_set = split_dataSet(train_set, split_ratio)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

  return train_loader, val_loader, test_loader

def split_dataSet(dataset, split_ratio):
  a_size = int(split_ratio * len(dataset))
  b_size = len(dataset) - a_size
  a_set, b_set = random_split(dataset, [a_size, b_size]) 
  return a_set, b_set
  
class PairDataset(Dataset):
    def __init__(self, dataset_origin, dataset_noisy):
        self.dataset_origin = dataset_origin
        self.dataset_noisy = dataset_noisy

    def __getitem__(self, index):
        x1 = self.dataset_origin[index]
        x2 = self.dataset_noisy[index]

        return x1, x2

    def __len__(self):
        return len(self.dataset_origin)


# Load train, validation and test data.
train_loader, val_loader, test_loader = load_data(
    path=config['data_path'], drop_probability=0.7, split_ratio=0.5, batch_size=64)
print(f'train_loader has {len(train_loader)} batches')
print(f'val_loader has {len(val_loader)} batches')
print(f'test_loader has {len(test_loader)} batches')

train_image_batch, train_noise_image_batch = iter(train_loader).next()
val_image_batch, val_noise_image_batch = iter(val_loader).next()
test_image_batch, test_noise_image_batch = iter(test_loader).next()

print('train')
show_gray_digits(train_image_batch[0], row=1, col=3)
show_gray_digits(train_noise_image_batch[0], 1, 3)
print('validation')
show_gray_digits(val_image_batch[0], 1, 3)
show_gray_digits(val_noise_image_batch[0], 1, 3)
print('test')
show_gray_digits(test_image_batch[0], 1, 3)
show_gray_digits(test_noise_image_batch[0], 1, 3)
del train_loader, val_loader, test_loader, train_image_batch, val_image_batch, test_image_batch, train_noise_image_batch, val_noise_image_batch, test_noise_image_batch

class MyNeuralNetRegressor(nn.Module):
    def __init__(self):
        super(MyNeuralNetRegressor, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000) 
        self.fc2 = nn.Linear(1000, 28*28)  
                                        

    def forward(self, x):
        x = x.view(-1, 28*28)           
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))    
        x = x.view(x.shape[0],1,28,28)   
        return x

if torch.cuda.is_available():
    device = torch.device('cuda') #GPU if available (cuda represents GPU)
else:
    device = torch.device('cpu')
  
# Verifying which device is being used GPU vs. CPU 
print(device)

def plot_eval_results(train_report, test_loss):
  train_loss_vals = []
  val_loss_vals = []

  for reports in train_report:
    train_loss_vals.append(reports.get('train_loss'))
    val_loss_vals.append(reports.get('val_loss'))
    
  for epochnum in range(0,len(train_loss_vals)):
    plt.plot(epochnum,test_loss)
    plt.plot(epochnum,val_loss_vals[epochnum])
    plt.plot(epochnum,train_loss_vals[epochnum])
  plt.legend(['test loss','val loss','train loss'])
  plt.title('Report')
  plt.xlabel('#Epochs')
  plt.ylabel('Loss')
  plt.show()
  
def main(config, net, criterion, drop_probability=0.7, epochs=10, batch_size=64, split_ratio=0.5, learning_rate=0.05):
  NeuralNet = net()
  optimizer = optim.Adam(NeuralNet.parameters(), lr=learning_rate)

  train_loader, val_loader, test_loader = load_data(path=config['data_path'],
                drop_probability=drop_probability, split_ratio=split_ratio, batch_size=batch_size)
  print(f'Successfully loaded data')

  NeuralNet, train_report = train(train_loader, val_loader, NeuralNet, epochs, criterion, optimizer, config['model_path'])
  print('Training finished')

  NeuralNet.load_state_dict(torch.load(config['model_path']))

  test_loss = test(test_loader, NeuralNet, criterion)

  print(f'Test loss is {test_loss} for drop_probability={drop_probability}, epochs={epochs}, batch_size={batch_size}, split_ratio={split_ratio}, learning_rate={learning_rate}')

  plot_eval_results(train_report, test_loss)

  return NeuralNet

def train(train_loader, val_loader, model, epochs, loss_function, optimizer, model_path, print_loss=True):
  best_loss = None
  report = []

  for epoch in range(epochs):  
    print(f"epoch {epoch}")

    model.train()
    train_loss = 0.0
    val_loss = 0.0

    report_single={
        train_loss: None,
        val_loss: None
    }

    for train_image_batch, train_noise_image_batch in iter(train_loader):
      model = model.to(device)
      train_noise_image_batch = train_noise_image_batch[0].to(device)
      train_image_batch = train_image_batch[0].to(device)     
      optimizer.zero_grad()
             
      denoised_images = model(train_noise_image_batch)
      loss = loss_function(denoised_images, train_image_batch)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    report_single['train_loss'] = train_loss

    if print_loss == True:
      print(f"train_loss={train_loss}")

    with torch.no_grad():
      for val_image_batch, val_noise_image_batch in iter(val_loader):
        val_image_batch = val_image_batch[0].to(device)
        val_noise_image_batch = val_noise_image_batch[0].to(device)

        denoised_images = model(val_noise_image_batch)

        loss = loss_function(denoised_images, val_image_batch)
          
        val_loss += loss.item()

      val_loss = val_loss/len(val_loader)
      report_single['val_loss'] = val_loss

      if print_loss == True:
        print(f"val_loss={val_loss}") 

      if best_loss is None or (val_loss < best_loss):
        best_loss = val_loss
        torch.save(model.state_dict(), model_path)

    report.append(report_single)

  return model, report

def test(dataloader, model, loss_function):
  test_loss = 0.0
  model.to(device)

  model.eval()

  for image_batch, noise_image_batch in iter(dataloader):
    # Move data to GPU
    image_batch = image_batch[0].to(device)
    noise_image_batch = noise_image_batch[0].to(device)

    denoised_images = model(noise_image_batch)
    loss = loss_function(denoised_images, image_batch)
    test_loss += loss.item()

  test_loss = test_loss/len(dataloader)
 
  return test_loss

config = {
    'data_path': './data', 
    'model_path': './checkpoints/best_model_1.pt' 
}
# create folder checkpoints to save model
!mkdir -p checkpoints

criterion = nn.L1Loss()

model = main(config, net=MyNeuralNetRegressor, criterion=criterion, epochs=50, 
             learning_rate=0.05, drop_probability=0.7, batch_size=64)

def plot_eval_results(train_report, test_loss):
  train_loss_vals = []
  val_loss_vals = []
  test_loss_vals = []

  for reports in train_report:
    train_loss_vals.append(reports.get('train_loss'))
    val_loss_vals.append(reports.get('val_loss'))
    test_loss_vals.append(test_loss) 

  plt.plot(test_loss_vals)
  plt.plot(val_loss_vals)
  plt.plot(train_loss_vals)
  plt.legend(['test loss','val loss','train loss'])
  plt.title('Report')
  plt.xlabel('#Epochs')
  plt.ylabel('Loss')
  plt.show()

criterion = nn.L1Loss()
model = main(config, criterion=criterion, net=MyNeuralNetRegressor, epochs=50, drop_probability=0.7, learning_rate=0.05, batch_size=64)

"""There are two layers. The first layer has input dimension 28x28 = 784, and 1000 neurons. Each of these neurons have a single bias term and weight 784, so the parameters in the first layer can be calculated as (784+1)x1000 = 7850000 parameters. The second layer has input dimension 1000, and 784 neurons. Again, each of these neurons have a single bias term, this time with weight 1000, so the paramters in the second layer can be calculated as (1000+1)x784 = 784784.
Adding the parameters in both layers, we get 785000 + 784784 = 1569784 parameters.
"""

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
  
        self.num_input_channels = 1 # number of input image channel, 
                                # 1 for grayscale images, 3 for color images
        self.num_feature_maps = 10 # number of feature maps
        self.num_output_channels = 1 # number of output image channel
            
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1

        self.conv1 = nn.Conv2d(self.num_input_channels,self.num_feature_maps,self.kernel_size,stride=self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.num_feature_maps,self.num_output_channels,self.kernel_size,stride=self.stride, padding=self.padding)
      

    def forward(self, x):
        
        x = torch.relu(self.conv1(x)) 
        x = torch.relu(self.conv2(x))
        x = x.view(x.shape[0],1,28,28)

        return x

model = main(config, MyCNN, criterion=nn.L1Loss(), drop_probability=0.7, learning_rate=0.05, epochs=50, batch_size=64)

"""layer 1:

1 input ch, 10 output ch, filter size = 3.

There are 1x10 = 10 kernels, with 3x3=9 weights + 1 bias, so there are 100 parameters in this layer.

layer 2:

10 input ch, 1 output ch, fiter size = 3.

There are 10x1 = 10 kernels, with 3x3=9 weights + 1 bias for the entire layer, so there are 9*10 + 1 = 91 parameters in this layer.


Total is 100 + 91 = 191
"""

train_loader, val_loader, test_loader = load_data(
    path=config['data_path'], drop_probability=0.7, split_ratio=0.5, batch_size=64)

train_image_batch, train_noise_image_batch = iter(train_loader).next()
val_image_batch, val_noise_image_batch = iter(val_loader).next()
test_image_batch, test_noise_image_batch = iter(test_loader).next()

train_image_batch, train_noise_image_batch = iter(train_loader).next()
val_image_batch, val_noise_image_batch = iter(val_loader).next()
test_image_batch, test_noise_image_batch = iter(test_loader).next()

net = MyNeuralNetRegressor()
#net = net.to(device)
denoised_images = net(val_noise_image_batch[0])
#show_gray_digits(val_noise_image_batch[0],2,5)
show_gray_digits(denoised_images,2,5)

net2 = MyCNN()
denoised_images2 = net2(val_noise_image_batch[0])
show_gray_digits(denoised_images2,2,5)

class MySuperNN(nn.Module):
    def __init__(self):
        super(MySuperNN, self).__init__()
  
        self.num_input_channels = 1 # number of input image channel, 
                                # 1 for grayscale images, 3 for color images
        self.num_feature_maps = 10 # number of feature maps
        self.num_output_channels = 1 # number of output image channel
            
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1

        self.conv1 = nn.Conv2d(self.num_input_channels,self.num_feature_maps,self.kernel_size,stride=self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.num_feature_maps,self.num_output_channels,self.kernel_size,stride=self.stride, padding=self.padding)
      


    def forward(self, x):
        
        x = torch.relu(self.conv1(x)) 
        x = torch.relu(self.conv2(x))
        x = x.view(x.shape[0],1,28,28)

        return x

model = main(config, MySuperNN, criterion=nn.L1Loss(), drop_probability=0.7, learning_rate=0.01, epochs=60, batch_size=64)

model = main(config, MyCNN, criterion=nn.L1Loss(), drop_probability=0.7, learning_rate=0.05, epochs=20, batch_size=64)