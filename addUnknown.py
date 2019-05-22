import numpy as np
import torch
from torch import nn
from torch import optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Process our image
def process_input(datadir):
    # Load Image
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data = datasets.ImageFolder(datadir,transform=transformations)
    testloader = torch.utils.data.DataLoader(test_data)
    return testloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = process_input('data/test_final')
label = {'accentObject': 0, 'armChair': 1, 'bed': 2, 'blanket': 3, 'bookcase': 4, 'cabinet': 5, 'chair': 6, 'chandelier': 7, 'clock': 8, 'coatRack': 9, 'coffeeTable': 10, 'console': 11, 'desk': 12, 'diningChair': 13, 'diningTable': 14, 'dishWasher': 15, 'dresser': 16, 'endTable': 17, 'floorLamp': 18, 'fridge': 19, 'kitchenAppliance': 20, 'kitchenWare': 21, 'microwave': 22, 'nightStand': 23, 'officeChair': 24, 'otherLighting': 25, 'ottoman': 26, 'oven': 27, 'pillow': 28, 'rug': 29, 'sofa': 30, 'tableLamp': 31, 'tvStand': 32, 'wallArt': 33, 'Unknown':34}
model = torch.load('model3.pth')
model.eval()
CM = np.zeros((35,35))

for inputs, labels in test_data:
    inputs, labels = inputs.to(device), labels.to(device)
    labels = labels.cpu().data.numpy()[0]
    logps = model.forward(inputs)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    top_class = top_class.cpu().data.numpy()[0][0]
    #print(top_class)
    select_p, top3_class = ps.topk(3, dim=1)
    #equals = top_class == labels.view(*top_class.shape)
    #print(select_p.cpu().data.numpy()[0][0])
    #if top_p.cpu() < 0.8 or select_p.cpu().data.numpy()[0][0] - select_p.cpu().data.numpy()[0][1] < 0.2:
    if np.var(select_p.cpu().data.numpy()) < 0.06:
        top_class = 34
    #print(labels)
    CM[labels][top_class] += 1

np.savetxt('confusion_matrix_result_unknown_var7.txt', CM, fmt='%d')

