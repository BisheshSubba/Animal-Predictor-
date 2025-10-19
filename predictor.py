import kagglehub
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image
# from google.colab import files

path = kagglehub.dataset_download("alessiocorrado99/animals10")
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
             "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep","ragno":"spider","scoiattolo": "squirrel",}
image_path = path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def find_image_root(base_path):
    for root, dirs, files in os.walk(base_path):
        if any(os.path.isdir(os.path.join(root, d)) for d in dirs):
            if len(dirs) > 1 and all(not d.startswith('.') for d in dirs):
                return root
    return base_path

image_root = find_image_root(image_path)
print("Detected image folder:", image_root)

def is_valid_image(path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False
all_image_paths = []
for root, _, files in os.walk(image_root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_image_paths.append(os.path.join(root, f))

valid_images = [p for p in all_image_paths if is_valid_image(p)]
invalid_images = set(all_image_paths) - set(valid_images)

print(f"📸 Total images found: {len(all_image_paths)}")
print(f"✅ Valid images: {len(valid_images)}")
print(f"❌ Corrupted images: {len(invalid_images)}")
full_dataset = datasets.ImageFolder(root=image_root, transform=None, is_valid_file=is_valid_image)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices =random_split(range(len(full_dataset)), [train_size, val_size])

train_dataset = datasets.ImageFolder(root=image_root, transform=train_transform, is_valid_file=is_valid_image)
val_dataset = datasets.ImageFolder(root=image_root, transform=val_transform, is_valid_file=is_valid_image)

train_dataset = Subset(train_dataset, train_indices.indices)
val_dataset = Subset(val_dataset, val_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)   # ✅ fixed
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)

epoches = 15

for epoch in range(epoches):
    model.train()
    running_loss =0.0
    for images, labels in train_loader:
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss/ len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct/total

    print(f"Epoch [{epoch + 1}/{epoches}] "
          f"Train Loss: {avg_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}%")

print("Training Complete")

save_path = "animals_resnet34.pth"
torch.save(model.state_dict(),save_path)

def predict_image(image_path):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = val_transform
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

    classes = full_dataset.classes
    predicted_class_it = classes[class_idx]

    predicted_class_en = translate.get(predicted_class_it, predicted_class_it)

    print(f"🖼️ Predicted: {predicted_class_en.capitalize()} ({predicted_class_it})")



uploaded = files.upload()

if uploaded:
    image_path = list(uploaded.keys())[0]
    print(f"Selected image: {image_path}")
    predict_image(image_path)
else:
    print("No image selected.")
