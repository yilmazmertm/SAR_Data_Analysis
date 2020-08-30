########################################################################################################
########################################################################################################

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader,random_split # önemli kütüphanelerin import edilmesi
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torchvision import models
%matplotlib inline

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU kontrolü

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]) # Resimin ImageNet Standartlarında normalize edilmesi

########################################################################################################
########################################################################################################

data = datasets.ImageFolder('./sar_data/', transform = transform) # dataset oluşturulması

np.random.seed(42)

train_size = int(0.70 * len(data)) # istenilen oranda datasetin train/test şeklinde bölünmesi
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])
print(f"Dataset has {len(data)} data points.")
print(f"Train Dataset has {len(train_dataset)} data points")
print(f"Test Dataset has {len(test_dataset)} data points.") 

########################################################################################################
########################################################################################################

###HYPERPARAMETERS###
batch_size = 16
num_epochs = 10
learning_rate = 0.0003
###HYPERPARAMETERS###

########################################################################################################
########################################################################################################

test_dataset[325]
alongside_cout = building_cout = road_cout = vegetation_cout = water_cout = 0
train_alongside_cout = train_building_cout = train_road_cout = train_vegetation_cout = train_water_cout = 0
for x in range(len(test_dataset)):
    a, b = test_dataset[x]
    if b == 0:
        alongside_cout +=1
    if b == 1:
        building_cout+=1
    if b == 2:
        road_cout+=1
    if b == 3:
        vegetation_cout+=1
    if b == 4:
        water_cout+=1

for x in range(len(train_dataset)):
    a, b = train_dataset[x]
    if b == 0:
        train_alongside_cout +=1
    if b == 1:
        train_building_cout+=1
    if b == 2:
        train_road_cout+=1
    if b == 3:
        train_vegetation_cout+=1
    if b == 4:
        train_water_cout+=1

print(f"Alongside number of samples in train set : {train_alongside_cout}") # hangi kategoride kaç veri var
print(f"Alongside number of samples in test set : {alongside_cout}")
print(f"Building number of samples in train set : {train_building_cout}")
print(f"Building number of samples in test set : {building_cout}")
print(f"Road number of samples in train set : {train_road_cout}")
print(f"Road number of samples in test set : {road_cout}")
print(f"Vegetation number of samples in train set : {train_vegetation_cout}")
print(f"Vegetation number of samples in test set : {vegetation_cout}")
print(f"Water number of samples in train set : {train_water_cout}")
print(f"Water number of samples in test set : {water_cout}")

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)

test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= False)

########################################################################################################
########################################################################################################

# DEFİNE THE MODEL
model = models.densenet121(pretrained= True)

for param in model.parameters():
    param.require_grad = False
    
fc = nn.Sequential(
    nn.Linear(1024,460),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(460, 5)
)

model.classifier = fc

model.to(device)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

########################################################################################################
########################################################################################################

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): ###TRAİNİNG_LOOP
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 70 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



arr_pred = np.empty((0,len(test_dataset)), int)

arr_label = np.empty((0,len(test_dataset)), int)

with torch.no_grad(): 
    correct = 0
    total = 0
    for images, labels in test_loader:  #PREDİCTİON LOOP
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred = predicted.cpu().numpy()
        lb = labels.cpu().numpy()
        arr_pred = np.append(arr_pred, pred)
        arr_label = np.append(arr_label, lb)
        
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


########################################################################################################
########################################################################################################

from sklearn import metrics
from cm_helper import plot_confusion_matrix
cm = metrics.confusion_matrix(arr_label, arr_pred)
plot_confusion_matrix(cm, target_names= ["alongside", "building", "road" ,"vegetation", "water"], title='Confusion matrix' , normalize= False)


print(f"The Accuracy : { 100 * metrics.accuracy_score(arr_label, arr_pred)}")
print(f"The Precision : {100 * metrics.precision_score(arr_label, arr_pred, average= 'macro')}")
print(f"The Recall : {100 * metrics.recall_score(arr_label, arr_pred, average= 'macro')}")
print(f"The F1 Score : {100 *metrics.f1_score(arr_label, arr_pred, average = 'macro')}")
