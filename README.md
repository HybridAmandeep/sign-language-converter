
````md
🤟 Sign Language Converter

A real-time **Sign Language to Text & Speech Converter** built using Python, MediaPipe, TensorFlow, Flask, and OpenCV.

The application detects hand gestures through a webcam, predicts alphabet signs using a trained Machine Learning model, converts them into text, and can also speak the generated sentence aloud.

Supports both:

- 🖥️ Desktop Application
- 🌐 Mobile-Friendly Web Dashboard


📸 Preview

Desktop Application
![Desktop UI](docs/desktop_screenshot.png)

Web Dashboard
![Web Dashboard](docs/web_dashboard.png)


✨ Features

- 🎥 Real-time webcam hand tracking
- ✋ Hand landmark detection using MediaPipe
- 🤖 TensorFlow/Keras gesture classification model
- 🗣️ Offline Text-to-Speech support
- 📝 Sentence builder from predicted letters
- 🌐 Responsive Flask web interface
- 📱 Mobile browser support over Wi-Fi
- 🎨 Modern glassmorphism UI
- ⚡ Fast real-time predictions

---

🛠️ Tech Stack

| Category | Technology |
|----------|-------------|
| Language | Python 3.10+ |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | TensorFlow, Keras |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Text-to-Speech | pyttsx3 |

---

📂 Project Structure

```bash
sign-language-converter/
│
├── app.py
├── collect_data.py
├── preprocess.py
├── train_model.py
├── requirements.txt
│
├── data/
├── model/
├── docs/
│
├── web/
│   ├── server.py
│   ├── static/
│   └── templates/
│
└── README.md
````

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/HybridAmandeep/sign-language-converter.git
cd sign-language-converter
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Model Training

## Collect Dataset

```bash
python collect_data.py
```

Follow the on-screen instructions to capture gesture samples.

---

## Preprocess Dataset

```bash
python preprocess.py
```

---

## Train Model

```bash
python train_model.py
```

This generates:

* `gesture_model.h5`
* `gesture_model.tflite`

inside the `model/` directory.

---

# ▶️ Running the Application

## Desktop Version

```bash
python app.py
```

Features:

* Real-time prediction
* Webcam interface
* Sentence builder
* Voice output

---

## Web Version

```bash
python web/server.py
```

Open in browser:

```bash
http://localhost:5000
```

For mobile access on same Wi-Fi:

```bash
http://YOUR_LOCAL_IP:5000
```

Example:

```bash
http://192.168.1.105:5000
```

---

# ⚙️ Configuration

## Confidence Threshold

Modify:

```python
CONFIDENCE_THRESHOLD
```

inside:

* `app.py`
* `web/server.py`

---

## Webcam Selection

Change:

```python
cv2.VideoCapture(0)
```

to:

```python
cv2.VideoCapture(1)
```

if using external webcam.

---

# 📱 Mobile Support

The Flask dashboard is fully responsive and works on:

* Android browsers
* iPhone browsers
* Tablets
* Desktop browsers

Ensure both devices are connected to the same Wi-Fi network.

---

# 🤝 Contributing

Contributions are welcome.

## Steps

1. Fork the repository
2. Create your feature branch

```bash
git checkout -b feature/my-feature
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push branch

```bash
git push origin feature/my-feature
```

5. Open Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgements

* MediaPipe
* TensorFlow
* OpenCV
* Flask
* pyttsx3

---

# 👨‍💻 Developer

## Amandeep Kumar

* IoT & AI Developer
* Android & ML Enthusiast
* Open Source Contributor

GitHub:
[HybridAmandeep GitHub Profile](https://github.com/HybridAmandeep?utm_source=chatgpt.com)

---

# ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork the project
* 🛠️ Contribute improvements

---

> “Bridging communication gaps through AI-powered sign language recognition.” ✨

```
```
