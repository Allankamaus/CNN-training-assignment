from torchvision.datasets import ImageFolder
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader
import torchvision.models as models
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
train_dataset = ImageFolder("ImageFolder/Training", transform=transform)
val_dataset   = ImageFolder("ImageFolder/Validation", transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=32
)


def train_and_validate():
    # Initialize model, loss function, optimizer
    model = models.resnet50(weights=True)
    model.fc = nn.Linear(model.fc.in_features, 2) # 2 classes: men, women
    criterion = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

    # Validation loop: output predictions for each image
    model.eval()
    class_names = train_dataset.classes  # ['men', 'women']
    print("\nValidation Results:")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(images.size(0)):
                batch_start = batch_idx * val_loader.batch_size
                img_path, _ = val_dataset.samples[batch_start + i]
                print(f"{img_path}: Predicted - {class_names[preds[i]]}")

if __name__ == "__main__":
    train_and_validate()

