# covid19-xray-classifier

# COVID-19 X-ray Classifier 🩻  
End-to-end Deep Learning system for COVID-19 detection from chest X-ray images using **ResNet50**, with a **Flask backend** and **React frontend**.

---

## 📌 Overview
This project implements a complete machine learning pipeline to classify chest X-ray images as **COVID-19 positive or negative**. It includes data preprocessing, multi-stage model training, evaluation, and deployment via a web application.

The project demonstrates how deep learning models can be integrated into **scalable, production-ready systems**.

---

## 🧠 Key Features
- Transfer learning using **ResNet50**
- Two-stage training strategy (feature extraction → fine-tuning)
- Model evaluation with saved metrics
- REST API using **Flask**
- Interactive **React** frontend for predictions
- Clean project structure suitable for real-world ML systems

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- TensorFlow / Keras
- ResNet50 (pretrained on ImageNet)
- NumPy, Pandas

### Backend
- Flask
- REST APIs

### Frontend
- React
- HTML, CSS, JavaScript

- ## 📂 Project Structure
covid19-xray-classifier/
│
├── backend/ # Flask backend
├── frontend/ # React frontend
│
├── data/
│ ├── train/
│ └── val/
│
├── model/
│ ├── best_model.h5
│ ├── best_model_stage2.h5
│
├── train_stage1.py # Initial training (frozen backbone)
├── train_stage2.py # Fine-tuning ResNet layers
├── evaluate.py # Model evaluation
├── predict_one.py # Single image prediction
├── Split_dataset.py # Dataset splitting utility
├── results_test.txt # Evaluation results
│
├── app.py # Flask API
├── flask_app.py # Flask app entry
├── README.md
├── .gitignore

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
Windows

bash
Copy code
venv\Scripts\activate
Mac / Linux

bash
Copy code
source venv/bin/activate
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
If requirements.txt is not present, install manually:

bash
Copy code
pip install tensorflow flask numpy pandas pillow
🧪 How to Run the Project
Step 1: Prepare Dataset
Ensure the dataset is structured as:

css
Copy code
data/
 ├── train/
 │   ├── covid/
 │   └── normal/
 └── val/
     ├── covid/
     └── normal/
(Optional) Automatically split dataset:

bash
Copy code
python Split_dataset.py
Step 2: Train the Model
Stage 1 – Feature Extraction
bash
Copy code
python train_stage1.py
Stage 2 – Fine-Tuning
bash
Copy code
python train_stage2.py
Trained models are saved in:

Copy code
model/
Step 3: Evaluate the Model
bash
Copy code
python evaluate.py
Evaluation results are written to:

Copy code
results_test.txt
🧪 How to Test
Test with a Single Image
bash
Copy code
python predict_one.py --image path/to/xray_image.png
The script outputs:

Predicted class (COVID / Normal)

Confidence score

🌐 Web Application
Run Flask Backend
bash
Copy code
python app.py
Backend runs at:

arduino
Copy code
http://localhost:5000
Run React Frontend
bash
Copy code
cd frontend
npm install
npm start
Frontend runs at:

arduino
Copy code
http://localhost:3000
The frontend communicates with the Flask API to perform real-time predictions.

⚠️ Notes
Large datasets and trained model files are excluded from version control.

This project is intended for educational and research purposes only and should not be used for medical diagnosis.

🚀 Future Enhancements
Add confusion matrix and ROC curves

Model explainability using Grad-CAM

Dockerize backend and frontend

Cloud deployment (AWS / GCP)

Multi-class classification support

👩‍💻 Author
Rishitha
MS Computer Science, Texas A&M University
Industry Experience: SAP, KPMG
Interests: Software Engineering, AI, Machine Learning, Scalable Systems

---

## 📂 Project Structure
