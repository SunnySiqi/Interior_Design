import pandas
import numpy as np
data = np.loadtxt('confusion_matrix_result_unknown_var.txt')
label = {'accentObject': 0, 'armChair': 1, 'bed': 2, 'blanket': 3, 'bookcase': 4, 'cabinet': 5, 'chair': 6, 'chandelier': 7, 'clock': 8, 'coatRack': 9, 'coffeeTable': 10, 'console': 11, 'desk': 12, 'diningChair': 13, 'diningTable': 14, 'dishWasher': 15, 'dresser': 16, 'endTable': 17, 'floorLamp': 18, 'fridge': 19, 'kitchenAppliance': 20, 'kitchenWare': 21, 'microwave': 22, 'nightStand': 23, 'officeChair': 24, 'otherLighting': 25, 'ottoman': 26, 'oven': 27, 'pillow': 28, 'rug': 29, 'sofa': 30, 'tableLamp': 31, 'tvStand': 32, 'wallArt': 33, 'Unknown': 34}
labels = [key for key in label]
output = pandas.DataFrame(data, labels, labels)
output.to_csv('results_unknown_var.csv')