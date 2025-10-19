<h1 align="center">🐾 Animal Image Classifier using ResNet34 (PyTorch)</h1>

<p align="center">
A deep learning project that classifies 10 animal species using a pretrained <b>ResNet34</b> model in <b>PyTorch</b>.  
Trained on the <a href="https://www.kaggle.com/datasets/alessiocorrado99/animals10">Animals-10 Kaggle dataset</a> using Google Colab (T4 GPU),  
and deployed locally with a <b>Tkinter GUI</b> for real-time image predictions.
</p>

<hr>

<h3>📚 Features</h3>
<ul>
  <li>Transfer learning with pretrained ResNet34</li>
  <li>Trained on 10 animal classes from Kaggle Animals10</li>
  <li>Detects and skips corrupted images</li>
  <li>Interactive Tkinter interface for local predictions</li>
</ul>

<h3>🐾 Animal Classes</h3>
<p>Dog, Cat, Horse, Elephant, Butterfly, Chicken, Cow, Sheep, Spider, Squirrel</p>

<h3>⚙️ Tech Stack</h3>
<ul>
  <li><b>Language:</b> Python</li>
  <li><b>Framework:</b> PyTorch</li>
  <li><b>Model:</b> ResNet34 (ImageNet pretrained)</li>
  <li><b>Training:</b> Google Colab (T4 GPU)</li>
  <li><b>Interface:</b> Tkinter GUI</li>
</ul>

<h3>🚀 Run the Project</h3>

<pre>
# Clone repo
git clone https://github.com/your-username/animal-classifier-resnet34.git
cd animal-classifier-resnet34

# Install dependencies
pip install torch torchvision Pillow kagglehub

# (Optional) Train the model
python train_model.py

# Run GUI for prediction
python predict_gui.py
</pre>

<h3>📦 Model Output</h3>
<pre>🖼️ Predicted: Cat (gatto)</pre>

