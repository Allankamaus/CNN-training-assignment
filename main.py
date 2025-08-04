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

