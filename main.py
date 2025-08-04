from torchvision.datasets import ImageFolder
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader
import torchvision.models as model
import torch.nn as nn
import torch.optim as optim
import torch, torchvision



#resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#DataLoader for dataset
train_dataset = ImageFolder("ImageFolder", transform=transform)
val_dataset   = ImageFolder("ImageFolder", transform=transform)
train_loader = DataLoader(
    train_dataset,batch_size = 32, shuffle = True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size = 32
)

#Initialize model, loss function, optimizer
model = model.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10) #Assuming 10 classes

criterion = nn.CrossEntropyLoss()
Optimizer = optim.Adam(model.parameters(), lr= 0.001)


for epoch in range(10): #Number of epochs
    model.train()  #set model to training mode
    for images, labels in train_loader:
        outputs = model(images) #forward pass
        loss = criterion(outputs, labels)#calculate loss
        Optimizer.zero_grad() #zero the gradients
        loss.backward()#backward pass
        Optimizer.step()