import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from PIL import Image
import os

from PIL import Image
import os

root_dir = "PetImages"
bad_files = []

for class_name in ["Cat", "Dog"]:
    class_dir = os.path.join(root_dir, class_name)

    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            bad_files.append(path)

print(f"Found {len(bad_files)} corrupted files.")

for path in bad_files:
    print(f"Removing: {path}")
    os.remove(path)

print("Cleanup finished.")

# importing dataset and model

transform = transforms.Compose([transforms.Resize((128, 128)),# resize pictures to [3,128,128]
                                transforms.ToTensor()]) # change from numpy array to tensor also 0-255 to 0-1


dataset = datasets.ImageFolder(root='PetImages', transform=transform) # the dataset is organized in a way that each class has its own folder, so we can use ImageFolder to load it easily
print(f"Number of samples in dataset: {len(dataset)}") # so dataset basicaly has fast accses to images and labels, we can use dataset[i] to get the i-th image and its label

print(f"Class names: {dataset.classes}")
print(f"Class to index mapping: {dataset.class_to_idx}")


print(dataset[0][0].shape) # check the shape of the first image tensor
print(f'random image is a cat' if dataset[0][1]==0 else 'random image is a dog') # check the label of the first image tensor

# plotting some images from the dataset

loader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(loader))

def imshow(img, filename="sample_batch.png"):
    npimg = img.numpy()# matlib expects images in numpy format, so we convert the tensor to numpy array
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))# the images are in the format [3,128,128], but matplotlib expects them in the format [128,128,3], so we use np.transpose to change the order of dimensions
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight")
    print(f"Image saved as {filename}")

grid = torchvision.utils.make_grid(images)#takes 4 images and arranges them in a grid format as one image, so we can visualize them together
imshow(grid)

print("Numeric labels:", labels) 
print("Class names:", [dataset.classes[label] for label in labels])


cat_indecies=[i for i,(_, label) in enumerate(dataset.samples) if label==0]
dog_indecies=[i for i,(_, label) in enumerate(dataset.samples) if label==1]

print(f"Number of cat images: {len(cat_indecies)}")
print(f"Number of dog images: {len(dog_indecies)}")

random.seed(42)
torch.manual_seed(42)

random.shuffle(cat_indecies)
random.shuffle(dog_indecies)

cat_split = int(0.8 * len(cat_indecies))
dog_split = int(0.8 * len(dog_indecies))

cat_indecies_train = cat_indecies[:cat_split]
cat_indecies_val = cat_indecies[cat_split:]

dog_indecies_train = dog_indecies[:dog_split]
dog_indecies_val = dog_indecies[dog_split:]

print(f'first 10 cat indecies for training: {cat_indecies_train[:10]}')
print(f'first 10 dog indecies for training: {dog_indecies_train[:10]}')


train_indecies = cat_indecies_train + dog_indecies_train
val_indecies = cat_indecies_val + dog_indecies_val




from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indecies) # train indecies = [17, 203, 999, ...]  train_dataset[0] == dataset[train_indecies[0]] 
val_dataset = Subset(dataset, val_indecies)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # from those indecies we create a dataloader for training, we set shuffle to True to make sure that the order of the images is different in each epoch, so the model doesn't learn the order of the images
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1) # input channels = 3 (RGB), output channels = 16, kernel size = 3x3, stride = 1, padding = 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1) # input channels = 16, output channels = 32, kernel size = 3x3, stride = 1, padding = 1
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2) # output layer with 2 neurons for binary classification

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) # conv1 -> relu -> pool
        x=self.pool(F.relu(self.conv2(x))) # conv2 -> relu -> pool
        x=torch.flatten(x,1) # flatten the output of the convolutional layers to feed it to the fully connected layers
        x=F.relu(self.fc1(x)) # fc1 -> relu
        x=self.fc2(x)
        return x


net=Net().to(device)


criterion = nn.CrossEntropyLoss() # loss function for multi-class classification, it combines log softmax and negative log likelihood loss in one function
optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # Adam optimizer with learning

images,labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

outputs = net(images)
print(f"Output shape: {outputs.shape}") # should be [batch_size, 2] because we have 2 classes
print(f"Output: {outputs}") # the output is the raw scores (logits) for each class, we will apply softmax to get the probabilities


num_epochs = 10

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    

    avg_train_loss = running_loss / len(train_loader)
    
    net.eval()
    val_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for images, labels in val_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=net(images)
            loss= criterion(outputs,labels)

            val_loss+=loss.item()

            _,predicted=torch.max(outputs,1)
            total += labels.size(0)
            correct+=(predicted == labels).sum().item()
    
    avg_val_loss= val_loss/len(val_loader)
    val_accuracy = 100* correct/total
    
    
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
        )
print("Finished training")

torch.save(net.state_dict(), "cats_vs_dogs_cnn.pth")
print("Model saved as cats_vs_dogs_cnn.pth")


        