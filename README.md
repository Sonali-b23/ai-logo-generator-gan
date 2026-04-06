# 🎨 AI Logo Generator using GANs

## 🚀 Overview

This project builds an **AI-powered logo generator** that creates startup logos from input text (company names) using **Generative Adversarial Networks (GANs)**.

The system evolves from a basic GAN to a **Conditional GAN (cGAN)** that generates logos conditioned on textual input, enabling customized and meaningful logo creation.

---

## 🧠 Key Features

* 🧬 Generate logos using GANs
* 🔤 Text-to-image generation (company name → logo)
* 🎯 Conditional GAN (cGAN) implementation
* 🖼️ Real-time logo generation
* 🌐 Web interface using Streamlit

---

## 🏗️ Project Structure

```
ai-logo-generator-gan/
│
├── data/                   # Dataset (raw & processed)
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Core source code
│   ├── config.py           # Configurations & hyperparameters
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # Generator & Discriminator
│   ├── training/           # Training scripts
│   ├── inference/          # Logo generation
│   └── embeddings/         # Text encoding
│
├── app/                    # Streamlit web app
├── outputs/                # Generated images & checkpoints
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Python**
* **PyTorch / TensorFlow** (for GANs)
* **NumPy, OpenCV, PIL**
* **Streamlit** (for UI)

---

## 🧩 How It Works

1. Input a **company name**
2. Convert text → embedding
3. Combine embedding with random noise
4. Generator creates a logo image
5. Discriminator evaluates authenticity
6. Model improves through adversarial training

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-logo-generator-gan.git
cd ai-logo-generator-gan
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Model

```bash
python src/training/train.py
```

### Generate Logos

```bash
python src/inference/generate_logo.py
```

### Run Web App

```bash
streamlit run app/app.py
```

---

## 📊 Future Improvements

* 🔥 Use CLIP for better text-image alignment
* 🎨 Style and color control (e.g., "minimal", "luxury")
* 🧠 Improve GAN stability (WGAN, DCGAN)
* ☁️ Deploy on cloud (AWS / Hugging Face Spaces)

---

## ⚠️ Challenges

* GAN training instability
* Mode collapse
* Dataset quality and bias

---

## 📌 Applications

* Startup branding
* Design automation
* Creative AI tools

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* GAN research papers
* Open-source datasets
* Deep learning community

---

## ⭐ Support

If you like this project, please ⭐ the repository!
