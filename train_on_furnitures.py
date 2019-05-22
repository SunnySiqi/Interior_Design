import numpy as np
import torch
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms, models

data_dir = 'data/train'
def load_split_train_test(datadir, valid_size = .2):
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(datadir,
                    transform=transformations)
    test_data = datasets.ImageFolder(datadir,
                    transform=transformations)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=512)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=512)
    return trainloader, testloader

# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))

    # Get the dimensions of the new image size
    width, height = img.size

    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img / 255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis, :]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


# Using our model to predict the label
def predict(image, model):
    input = image.to(device)
    logps = model.forward(input)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    return top_p, top_class



trainloader, testloader = load_split_train_test(data_dir, .2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(trainloader.dataset.class_to_idx)

#for inputs, labels in trainloader:
    #print(labels)

model = models.resnet50(pretrained=True)
#print(model)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 34),
                         nn.LogSoftmax(dim=1))
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0005)
model.to(device)

epochs = 24
steps = 0
running_loss = 0
print_every = 5
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            # running_corrects = 0
            model.eval()
            count = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    # running_corrects += torch.sum(top_class == labels.data)
                    count += 1
            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")
                  # f"Test accuracy: {running_corrects.double() / len(testloader):.3f}")
            running_loss = 0
            model.train()


torch.save(model,'model3.pth')

# model = torch.load('model.pth')
# model.eval()


# Process Image
image = process_image("C:/Users/Siqi Wang/Desktop/scene/2Dimg/pillow.jpg")
# Give image to model to predict output
top_prob, top_class = predict(image, model)
print("The model is ", top_prob, "% certain that the image has a predicted class of ", top_class)
