# Starter code for Part 1 of the Small Data Solutions Project
# 

# Set up image data for train and test
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Set up Transforms (train, val, and test)

# <<<YOUR CODE HERE>>>
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

# <<<YOUR CODE HERE>>>
# hint, create a variable that contains the class_names. You can get them from the ImageFolder
data_dir = os.path.join(os.getcwd(), "imagedata-50")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = num_workers)
                for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
train_loader = data_loaders['train']
val_loader = data_loaders['val']
test_loader = data_loaders['test']
class_names = image_datasets['test'].classes

# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#< <<YOUR CODE HERE>>>
# 1. Get trained model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16(weights='DEFAULT').to(device)

# 2. Freeze layers so they won't all be trained again with our data
for param in model.parameters():
    param.requires_grad = False

# 3. Replace top layer classifier with a classifer for our 3 categories
model.classifier[6].out_features = 3
# print(model.classifier)

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

# <<<YOUR CODE HERE>>>
num_epochs = 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
    trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
    test_model(test_loader, trained_model, class_names)


if __name__ == '__main__':
    main()
    print("done")