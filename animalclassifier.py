import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import requests
from io import BytesIO

model = models.resnet34(weights = None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model_path= r"C:\Users\swastik limbu\Desktop\leethoni\animalclassification\animals_resnet34.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])    
classes = [
    "dog", "horse", "elephant", "butterfly", 
    "chicken", "cat", "cow", "sheep", "spider", "squirrel"
]

animal_images = {
    "dog": "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg",
    "horse": "https://cdn.pixabay.com/photo/2016/11/14/04/36/boy-1822559_1280.jpg",
    "elephant": "https://cdn.pixabay.com/photo/2017/09/25/13/14/elephant-2785144_1280.jpg",
    "butterfly": "https://cdn.pixabay.com/photo/2017/08/18/17/04/butterfly-2656265_1280.jpg",
    "chicken": "https://cdn.pixabay.com/photo/2017/06/09/15/55/hen-2387694_1280.jpg",
    "cat": "https://cdn.pixabay.com/photo/2017/07/25/01/22/cat-2536662_1280.jpg",
    "cow": "https://cdn.pixabay.com/photo/2016/11/17/12/52/cow-1831129_1280.jpg",
    "sheep": "https://cdn.pixabay.com/photo/2016/11/21/13/19/lamb-1845839_1280.jpg",
    "spider": "https://cdn.pixabay.com/photo/2014/11/17/20/19/spider-535361_1280.jpg",
    "squirrel": "https://cdn.pixabay.com/photo/2016/03/22/09/45/squirrel-1272013_1280.jpg"
}

def predicted_image(image_file):
    image = Image.open(image_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    predicted_class = classes[predicted.item()]
    confidence_percent = confidence.item() * 100
    
    return predicted_class, confidence_percent

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img_data = response.content
        return Image.open(BytesIO(img_data))
    except:
        return Image.new('RGB', (200, 200), color='gray')

class AnimalClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐾 Animal Classifier")
        self.root.geometry("500x700")
        self.root.configure(bg="#f0f8ff")
        self.root.resizable(True, True)
        
        # Create main frame
        main_frame = tk.Frame(root, bg="#f0f8ff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="🐾 Animal Classifier", 
                              font=("Arial", 24, "bold"), bg="#f0f8ff", fg="#2c3e50")
        title_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(main_frame, 
                             text="Upload an image to identify the animal!\nSupported: Dog, Cat, Horse, Elephant, Butterfly, Chicken, Cow, Sheep, Spider, Squirrel",
                             font=("Arial", 12), bg="#f0f8ff", fg="#34495e", wraplength=400)
        desc_label.pack(pady=10)
        
        # Upload button
        self.btn_upload = tk.Button(main_frame, text="📁 Upload Image", 
                                   command=self.upload_predict,
                                   font=("Arial", 14, "bold"), 
                                   bg="#3498db", fg="white", 
                                   padx=20, pady=10,
                                   cursor="hand2")
        self.btn_upload.pack(pady=20)
        
        # Image display
        self.image_frame = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.label_image = tk.Label(self.image_frame, bg="#ecf0f1")
        self.label_image.pack(pady=10)
        
        # Result display
        self.result_frame = tk.Frame(main_frame, bg="#f0f8ff")
        self.result_frame.pack(pady=10, fill=tk.X)
        
        self.label_result = tk.Label(self.result_frame, text="", 
                                    font=("Arial", 16, "bold"), 
                                    bg="#f0f8ff", fg="#27ae60")
        self.label_result.pack()
        
        self.label_confidence = tk.Label(self.result_frame, text="", 
                                       font=("Arial", 12), 
                                       bg="#f0f8ff", fg="#7f8c8d")
        self.label_confidence.pack()
    
    
    def upload_predict(self):
        file_path = filedialog.askopenfilename(
            title="Select an animal image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if not file_path:
            return
        
        # Show loading state
        self.btn_upload.config(text="🔍 Analyzing...", state=tk.DISABLED)
        self.root.update()
        
        try:
            # Display uploaded image
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.label_image.configure(image=img_tk)
            self.label_image.image = img_tk
            
            # Get prediction
            result, confidence = predicted_image(file_path)
            
            # Display results
            self.label_result.configure(text=f"Prediction: {result.capitalize()}")
            self.label_confidence.configure(text=f"Confidence: {confidence:.1f}%")
            
            # Color code based on confidence
            if confidence > 80:
                self.label_confidence.configure(fg="#27ae60")
            elif confidence > 60:
                self.label_confidence.configure(fg="#f39c12")
            else:
                self.label_confidence.configure(fg="#e74c3c")
                
        except Exception as e:
            self.label_result.configure(text="Error processing image")
            self.label_confidence.configure(text="Please try another image")
            print(f"Error: {e}")
        
        finally:
            # Reset button
            self.btn_upload.config(text="📁 Upload Image", state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalClassifierApp(root)
    root.mainloop()