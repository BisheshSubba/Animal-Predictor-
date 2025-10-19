import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

model = models.resnet34(weights = None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model.state_dict(torch.load("", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])    

def predicted_image(image_file):
    image = Image.open(image_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _,predicted = torch.max(output, 1)
    return predicted.item()

root = tk.Tk
root.title("Animal Classifier")
root.geometry("400x600")
root.configure(bg="#ffe599")
label_result = tk.Label(root, text="", font=("Arial", 18, "bold"), bg="#f0f0f0")
label_result.pack(pady=20)

label_image = tk.Label(root, bg="#f0f0f0")
label_image.pack()
 
def upload_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    label_image.configure(image=img_tk)
    label_image.image = img_tk

    result = predicted_image(file_path)
    label_result.configure(text=f"Prediction: {result}")

btn_upload = tk.Button(root, text="Upload Image", command=upload_predict,
                       font=("Arial", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
btn_upload.pack(pady=20)

root.mainloop()
