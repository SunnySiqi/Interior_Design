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
label = {'accentObject': 0, 'armChair': 1, 'bed': 2, 'blanket': 3, 'bookcase': 4, 'cabinet': 5, 'chair': 6, 'chandelier': 7, 'clock': 8, 'coatRack': 9, 'coffeeTable': 10, 'console': 11, 'desk': 12, 'diningChair': 13, 'diningTable': 14, 'dishWasher': 15, 'dresser': 16, 'endTable': 17, 'floorLamp': 18, 'fridge': 19, 'kitchenAppliance': 20, 'kitchenWare': 21, 'microwave': 22, 'nightStand': 23, 'officeChair': 24, 'otherLighting': 25, 'ottoman': 26, 'oven': 27, 'pillow': 28, 'rug': 29, 'sofa': 30, 'tableLamp': 31, 'tvStand': 32, 'wallArt': 33}
model = torch.load('model3.pth')
model.eval()
#CM = np.zeros((34,34))
probs = np.zeros((1837,3))
probs_right = np.zeros((13721,3))
count = 0
count_error = 0
for inputs, labels in test_data:

    inputs, labels = inputs.to(device), labels.to(device)
    logps = model.forward(inputs)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    select_p, top3_class = ps.topk(3, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    if not equals:
        #print(select_p.cpu().data.numpy())
        probs[count_error] = select_p.cpu().data.numpy()
        count_error += 1
    else:
        probs_right[count] = select_p.cpu().data.numpy()
        count += 1
    #CM += confusion_matrix(labels.cpu(), top_class.cpu(), np.arange(34))

#np.savetxt('confusion_matrix_result.txt', CM, fmt='%d')
#Axes3D.scatter(xs=probs[:,0], ys=probs[:,1], zs=probs[:,2], zdir='z', s=20, c=None)

# ax = plt.axes(projection='3d')
#
#
# #ax.scatter(xs = probs_right[:,0], ys = probs_right[:,1], zs =probs_right[:,2], c= 'green', linewidth=0.5)
# ax.scatter(xs = probs[:,0], ys = probs[:,1], zs =probs[:,2], c= probs[:,2], linewidth=0.5)
# plt.savefig("error_only_confidence_top3")
# plt.show()
print(np.var(probs))
print(np.var(probs_right))