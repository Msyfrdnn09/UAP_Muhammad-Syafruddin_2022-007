# 🗑️ GARBAGE CLASSIFICATION DASHBOARD
### UAP Pembelajaran Mesin - Muhammad Syafruddin (2022-007)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Daftar Isi
1. [Deskripsi Proyek](#-deskripsi-proyek)
2. [Dataset dan Preprocessing](#-dataset-dan-preprocessing)
3. [Model yang Digunakan](#-model-yang-digunakan)
4. [Hasil Evaluasi dan Analisis](#-hasil-evaluasi-dan-analisis)
5. [Panduan Instalasi](#-panduan-instalasi)
6. [Cara Menjalankan Sistem](#-cara-menjalankan-sistem)
7. [Struktur Proyek](#-struktur-proyek)
8. [Screenshot Aplikasi](#-screenshot-aplikasi)
9. [Kontributor](#-kontributor)

---

## 🎯 Deskripsi Proyek

Project **Garbage Classification Dashboard** adalah sistem klasifikasi gambar sampah berbasis web yang menggunakan teknik Deep Learning untuk mengidentifikasi dan mengklasifikasikan jenis sampah secara otomatis. Sistem ini dikembangkan untuk membantu proses pemilahan sampah yang lebih efisien dan akurat.

### Tujuan Proyek:
- Mengimplementasikan model Deep Learning untuk klasifikasi gambar sampah
- Membandingkan performa tiga arsitektur model yang berbeda
- Membangun dashboard interaktif untuk prediksi real-time
- Menyediakan visualisasi hasil evaluasi model

### Fitur Utama:
- ✅ Upload dan prediksi gambar sampah
- ✅ Perbandingan performa tiga model berbeda
- ✅ Visualisasi confusion matrix dan classification report
- ✅ Dashboard interaktif dengan Streamlit
- ✅ Real-time prediction dengan confidence score

---

## 📊 Dataset dan Preprocessing

### Sumber Dataset
- **Nama Dataset**: Garbage Classification Dataset
- **Platform**: Kaggle / Google Colab
- **Total Training Data**: ~1,400 images (estimated)
- **Total Test Data**: 157 images
- **Jumlah Kelas**: 6 kategori

### Kategori Sampah:

| Kategori | Deskripsi | Test Samples | % Dataset |
|----------|-----------|--------------|-----------|
| **Cardboard** | Kotak karton, kemasan kardus | 15 | 9.6% |
| **Glass** | Botol kaca, pecahan kaca | 28 | 17.8% |
| **Metal** | Kaleng, besi, aluminium | 25 | 15.9% |
| **Paper** | Kertas, koran, majalah | 44 | 28.0% |
| **Plastic** | Botol plastik, kemasan plastik | 35 | 22.3% |
| **Trash** | Sampah organik, campuran | 10 | 6.4% |
| **TOTAL** | | **157** | **100%** |

**Observasi:**
- ⚠️ **Class imbalance**: Paper (44) vs Trash (10) = 4.4x difference
- ⚠️ **Small test set**: 157 samples total
- ⚠️ Impact: Model bias toward majority classes (paper, plastic, glass)

### Preprocessing Data

#### 1. **Image Augmentation**
Untuk meningkatkan variasi data training dan mengurangi overfitting:
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

#### 2. **Image Resizing**
- **Ukuran Input**: 224x224 pixels (sesuai standar ImageNet)
- **Format**: RGB (3 channels)

#### 3. **Normalisasi**
- Pixel values dinormalisasi ke range [0, 1] dengan membagi 255
- Membantu mempercepat konvergensi model

#### 4. **Split Data**
- **Training Set**: 70% (untuk training model)
- **Validation Set**: 15% (untuk validasi selama training)
- **Test Set**: 15% (untuk evaluasi final)

---

## 🤖 Model yang Digunakan

Proyek ini mengimplementasikan dan membandingkan tiga arsitektur Deep Learning:

### 1. **Custom CNN (Convolutional Neural Network)**

#### Arsitektur:
```
Model: Custom CNN
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Conv2D-1                    (None, 222, 222, 32)      896       
MaxPooling2D-1              (None, 111, 111, 32)      0         
Conv2D-2                    (None, 109, 109, 64)      18,496    
MaxPooling2D-2              (None, 54, 54, 64)        0         
Conv2D-3                    (None, 52, 52, 128)       73,856    
MaxPooling2D-3              (None, 26, 26, 128)       0         
Flatten                     (None, 86528)             0         
Dense-1                     (None, 128)               11,075,712
Dropout                     (None, 128)               0         
Dense-2 (Output)            (None, 6)                 774       
=================================================================
Total params: 11,169,734
Trainable params: 11,169,734
```

#### Karakteristik:
- **Keunggulan**: 
  - Model ringan dan cepat untuk training
  - Cocok untuk dataset kecil-menengah
  - Mudah dikustomisasi sesuai kebutuhan
- **Kelemahan**:
  - Akurasi mungkin lebih rendah dari pre-trained model
  - Membutuhkan lebih banyak data untuk hasil optimal

#### Hyperparameters:
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50
- **Loss Function**: Categorical Crossentropy

---

### 2. **MobileNetV2 (Transfer Learning)**

#### Arsitektur:
MobileNetV2 adalah model yang dioptimalkan untuk perangkat mobile dan embedded systems.

```
Model: MobileNetV2 Transfer Learning
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
MobileNetV2 Base (frozen)   (None, 7, 7, 1280)        2,257,984 
GlobalAveragePooling2D      (None, 1280)              0         
Dense-1                     (None, 128)               163,968   
Dropout                     (None, 128)               0         
Dense-2 (Output)            (None, 6)                 774       
=================================================================
Total params: 2,422,726
Trainable params: 164,742
Non-trainable params: 2,257,984
```

#### Karakteristik:
- **Keunggulan**:
  - Model ringan dan cepat (~14 MB)
  - **BEST PERFORMER pada project ini (63% accuracy)**
  - Cocok untuk deployment di web/mobile
  - Inference time cepat
  - Efisien dalam penggunaan memori
  - Generalisasi baik dengan frozen base
- **Kelemahan**:
  - Masih perlu improvement untuk akurasi >70%
  - Beberapa kelas sulit dibedakan

#### Konfigurasi:
```python
base_mobilenet = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_mobilenet.trainable = False  # Freeze base

# Custom top layers
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(6, activation="softmax")(x)
```

- **Optimizer**: Adam dengan learning rate 0.001
- **Batch Size**: 32
- **Epochs**: 10
- **Training Time**: ~35 menit

---

### 3. **ResNet50 (Transfer Learning)**

#### Arsitektur:
ResNet50 menggunakan residual connections untuk mengatasi vanishing gradient problem.

```
Model: ResNet50 Transfer Learning
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
ResNet50 Base (frozen)      (None, 7, 7, 2048)        23,587,712
GlobalAveragePooling2D      (None, 2048)              0         
Dense-1                     (None, 128)               262,272   
Dense-2 (Output)            (None, 6)                 774       
=================================================================
Total params: 23,850,758
Trainable params: 263,046
Non-trainable params: 23,587,712
```

#### Karakteristik:
- **Keunggulan**:
  - Arsitektur powerful dengan residual connections
  - Pre-trained pada ImageNet
  - 50 layers deep architecture
  - Potensi tinggi untuk image classification
  
- **Kelemahan** (pada project ini):
  - **SEVERE OVERFITTING** (29.94% validation accuracy)
  - Model terlalu kompleks untuk dataset size ini
  - Training accuracy tinggi tapi validation rendah
  - Inference time lambat (54 detik)
  - File size besar (~90 MB)

#### Konfigurasi:
```python
base_resnet = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_resnet.trainable = False  # Freeze base

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(6, activation="softmax")(x)
```

- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 10

#### Analisis Masalah ResNet50:
1. **Dataset terlalu kecil** untuk model sebesar ini
2. **Domain shift**: ImageNet (general objects) vs Garbage (specific domain)
3. **Frozen layers** mungkin terlalu rigid
4. **Solusi potensial**:
   - Unfreeze beberapa top layers
   - Fine-tune dengan learning rate kecil
   - Tambah data augmentation
   - Gunakan dataset lebih besar

---

## 📈 Hasil Evaluasi dan Analisis

### Metrik Evaluasi yang Digunakan:
1. **Accuracy**: Persentase prediksi yang benar
2. **Precision**: Ketepatan prediksi positif
3. **Recall**: Kemampuan mendeteksi kelas positif
4. **F1-Score**: Harmonic mean dari precision dan recall

### Perbandingan Performa Model

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| CNN Base | 24.84% | 29.73% | 24.84% | 25.19% | ~152 min (10 epochs) | ~23s |
| MobileNetV2 | **63.06%** | **64.94%** | **63.06%** | **60.91%** | ~35 min | ~27s |
| ResNet50 | 29.94% | 27.07% | 29.94% | 21.29% | ~40 min | ~54s |

*Note: Waktu training dan inference diukur pada Google Colab dengan GPU T4*

### Analisis Perbandingan:

#### 🏆 **MobileNetV2 - Best Overall Performance**
**Kelebihan:**
- Akurasi tertinggi (63.06%)
- Performa terbaik untuk semua metrik
- Model relatif ringan (~14 MB)
- Inference time cepat (27 detik untuk 157 samples)
- Cocok untuk production dengan balance speed dan accuracy

**Kekurangan:**
- Memerlukan fine-tuning lebih lanjut untuk akurasi >70%
- Beberapa kelas masih sulit dibedakan (carton vs paper)

**Kapan menggunakan:**
- **REKOMENDASI UTAMA untuk production**
- Balance terbaik antara akurasi dan kecepatan
- Deployment di web app atau mobile
- Resource komputasi moderate

**Hasil Detail (Classification Report):**
```
              precision    recall  f1-score   support
      carton       0.38      0.33      0.36        15
       glass       0.62      0.71      0.67        28
       metal       0.60      0.84      0.70        25
       paper       0.71      0.80      0.75        44
     plastic       0.77      0.29      0.42        35
       trash       0.53      0.80      0.64        10

    accuracy                           0.63       157
```

---

#### ⚠️ **ResNet50 - Transfer Learning Challenge**
**Kelebihan:**
- Arsitektur powerful untuk image classification
- Pre-trained pada ImageNet
- Potensi tinggi jika di-fine-tune dengan benar

**Kekurangan:**
- Akurasi rendah (29.94%) - **OVERFITTING DETECTED**
- Training accuracy tinggi tapi validation rendah
- Model terlalu kompleks untuk dataset ini
- Inference time lambat (54 detik)

**Analisis Masalah:**
- Model terlalu dalam untuk dataset yang relatif kecil
- Frozen layers mungkin tidak cocok untuk garbage domain
- Perlu fine-tuning seluruh layer atau unfreeze beberapa layer
- Data augmentation perlu ditingkatkan

**Hasil Detail (Classification Report):**
```
              precision    recall  f1-score   support
      carton       0.00      0.00      0.00        15
       glass       0.00      0.00      0.00        28
       metal       0.21      0.28      0.24        25
       paper       0.32      0.80      0.46        44
     plastic       0.60      0.09      0.15        35
       trash       0.22      0.20      0.21        10

    accuracy                           0.30       157
```

---

#### 🎯 **CNN Base - Baseline Model**
**Kelebihan:**
- Arsitektur sederhana dan cepat training
- Good for learning dan baseline comparison
- Fully customizable

**Kekurangan:**
- Akurasi paling rendah (24.84%) - **SEVERE OVERFITTING**
- Training accuracy: 89.42% vs Validation: 24.84%
- Gap terlalu besar menunjukkan model menghafal, bukan belajar
- Model terlalu sederhana untuk kompleksitas data

**Analisis Masalah:**
- Training accuracy naik terus tapi validation turun → classic overfitting
- Perlu regularization (Dropout, BatchNorm, L2)
- Dataset mungkin terlalu kecil untuk train from scratch
- Perlu data augmentation lebih agresif

**Hasil Detail (Classification Report):**
```
              precision    recall  f1-score   support
      carton       0.00      0.00      0.00        15
       glass       0.38      0.11      0.17        28
       metal       0.21      0.36      0.26        25
       paper       0.45      0.34      0.39        44
     plastic       0.29      0.29      0.29        35
       trash       0.06      0.20      0.10        10

    accuracy                           0.25       157
```

**Training History:**
- Epoch 1: Train Acc: 39.31%, Val Acc: 26.11%
- Epoch 10: Train Acc: 89.42%, Val Acc: 24.84%
- **Gap semakin besar = overfitting semakin parah**

---

### Confusion Matrix Analysis

#### MobileNetV2 (Best Performance)
```
Actual    →  carton  glass  metal  paper  plastic  trash
carton         5      1      2      3       2       2
glass          1     20      3      2       1       1
metal          0      1     21      2       1       0
paper          1      1      3     35       3       1
plastic        4      3      4      6      10       8
trash          0      0      0      2       0       8
```

**Insight dari Confusion Matrix:**
- **Kelas terbaik**: Glass (71% recall), Metal (84% recall), Paper (80% recall)
- **Kelas bermasalah**: 
  - **Plastic** (29% recall) - sering salah klasifikasi ke kelas lain
  - **Carton** (33% recall) - confused dengan paper dan plastic
  - **Trash** (80% recall tapi precision rendah)
- **Confusion patterns**:
  - Carton ↔ Paper: Material serupa (kertas)
  - Plastic → Multiple classes: Variasi plastic sangat beragam
  - Metal cukup distinctive (akurasi tinggi)

**Rekomendasi Improvement:**
1. Tambah data untuk kelas Plastic dan Carton
2. Feature engineering untuk membedakan carton vs paper
3. Augmentation lebih agresif untuk plastic variations
4. Consider ensemble methods

---

### Training History Analysis

#### CNN Base - Severe Overfitting
```
Epoch    Train Acc    Val Acc    Train Loss    Val Loss
1        39.31%       26.11%     1.64          2.61
5        79.78%       25.48%     0.57          2.48
10       89.42%       24.84%     0.31          2.95
```
**Problem**: Gap semakin membesar → model menghafal training data

#### MobileNetV2 - Good Generalization  
```
Validation Accuracy: 63.06%
Training stabilized after epoch 7-10
Minimal overfitting observed
```
**Success**: Model belajar pattern yang general

#### ResNet50 - Failed Transfer Learning
```
Validation Accuracy: 29.94%
Model tidak belajar dengan baik
Frozen layers terlalu rigid untuk domain garbage
```
**Problem**: Pre-trained ImageNet features tidak compatible

---

### Rekomendasi Deployment

#### ✅ Untuk Production: **PILIH MobileNetV2**

**Alasan:**
- ✅ Akurasi terbaik (63.06%) - 2x lipat dari model lain
- ✅ Model ringan (~14 MB) - cocok untuk web deployment
- ✅ Inference cepat (27s untuk 157 images = ~170ms per image)
- ✅ Generalisasi baik - tidak overfitting
- ✅ Balance optimal antara speed dan accuracy

**Use Cases:**
- **Web Dashboard** (Streamlit) ✅ **Recommended**
- Mobile app (dengan TensorFlow Lite)
- Real-time garbage sorting system
- Edge devices dengan resource terbatas

**Optimizations untuk Production:**
1. Convert ke TensorFlow Lite untuk mobile
2. Quantization untuk reduce model size
3. Batch prediction untuk multiple images
4. Caching untuk frequent predictions

---

#### 🔧 Untuk Improvement: **Fine-tune MobileNetV2**

**Next Steps:**
1. **Unfreeze top layers** (last 20-30 layers)
   ```python
   base_mobilenet.trainable = True
   for layer in base_mobilenet.layers[:-30]:
       layer.trainable = False
   ```

2. **Train lebih lama** dengan learning rate kecil (1e-5)
   - Potential: 70-80% accuracy

3. **Data augmentation lebih agresif**
   ```python
   rotation_range=40,
   zoom_range=0.3,
   brightness_range=[0.5, 1.5]
   ```

4. **Ensemble methods**
   - Combine predictions dari multiple augmentations
   - Voting mechanism

---

#### ❌ Tidak Direkomendasikan

**CNN Base:**
- ❌ Severe overfitting (89% train vs 25% val)
- ❌ Tidak cocok untuk production
- ✅ Hanya untuk learning purposes

**ResNet50:**
- ❌ Underfitting dengan current setup
- ❌ Perlu major architecture changes
- ❌ Terlalu berat untuk benefit yang didapat
- ⚠️ Bisa dicoba lagi dengan:
  - Unfreeze more layers
  - Different pre-processing
  - More training data

---

## 🛠️ Panduan Instalasi

### Prerequisites
Pastikan Anda sudah menginstall:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Git

### Langkah Instalasi

#### 1. Clone Repository
```bash
git clone https://github.com/Msyfrdnn09/UAP_Muhammad-Syafruddin_2022-007.git
cd UAP_Muhammad-Syafruddin_2022-007
```

#### 2. Buat Virtual Environment (Recommended)
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Isi requirements.txt:**
```
streamlit==1.28.0
tensorflow==2.13.0
keras==2.13.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
pillow==10.0.0
opencv-python==4.8.0
```

#### 4. Download Pre-trained Models
Model-model yang sudah di-training tersedia di folder `models/`:
- `custom_cnn_model.h5`
- `vgg16_model.h5`
- `mobilenet_model.h5`

*Note: Jika model belum tersedia, jalankan notebook training terlebih dahulu*

---

## 🚀 Cara Menjalankan Sistem

### 1. Menjalankan Jupyter Notebook (Training Model)

Jika Anda ingin training ulang model atau melihat proses training:

```bash
# Install Jupyter jika belum
pip install jupyter

# Jalankan Jupyter Notebook
jupyter notebook

# Buka file UAP.ipynb di browser
```

**Isi Notebook:**
- Data Loading dan Exploration
- Data Preprocessing dan Augmentation
- Training ketiga model
- Evaluasi dan Comparison
- Save trained models

---

### 2. Menjalankan Dashboard Streamlit

#### Cara 1: Dari Command Line
```bash
# Pastikan virtual environment sudah aktif
streamlit run app.py
```

#### Cara 2: Dengan Python
```bash
python -m streamlit run app.py
```

Dashboard akan terbuka otomatis di browser pada:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

### 3. Menggunakan Dashboard

#### A. Halaman Utama (Home)
- Overview proyek
- Informasi dataset
- Statistik model

#### B. Halaman Prediction
1. **Upload Gambar**
   - Klik "Browse files" atau drag & drop
   - Format: JPG, JPEG, PNG
   - Max size: 200MB

2. **Pilih Model**
   - Custom CNN
   - VGG16
   - MobileNetV2

3. **Klik "Predict"**
   - Hasil prediksi akan muncul
   - Confidence score per kelas
   - Visualization

#### C. Halaman Model Comparison
- Perbandingan metrik ketiga model
- Confusion matrix
- Classification report
- Training history

#### D. Halaman About
- Informasi pembuat
- Deskripsi proyek
- Dokumentasi

---

### 4. Testing

#### Unit Testing
```bash
# Jalankan test
python -m pytest tests/

# Dengan coverage
pytest --cov=src tests/
```

#### Manual Testing
1. Test dengan sample images di folder `test_images/`
2. Coba berbagai jenis sampah
3. Verifikasi prediksi dengan ground truth

---

## 📁 Struktur Proyek

```
UAP_Muhammad-Syafruddin_2022-007/
│
├── 📂 data/
│   ├── 📂 garbage_classification/    # Dataset utama
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── 📂 test_images/               # Sample images untuk testing
│
├── 📂 models/
│   ├── cnn_base_model.keras          # CNN Base Model (24.84% acc)
│   ├── mobilenet_model.keras         # MobileNetV2 (63.06% acc) ⭐ BEST
│   ├── resnet50_model.keras          # ResNet50 (29.94% acc)
│   └── 📂 training_history/          # Training logs dan plots
│       ├── cnn_history.png
│       ├── mobilenet_history.png
│       └── resnet_history.png
│
├── 📂 notebooks/
│   └── UAP.ipynb                     # Main training notebook
│       ├── Data Loading & EDA
│       ├── Data Preprocessing & Augmentation
│       ├── Model Training (3 models)
│       ├── Evaluation & Comparison
│       └── Visualization
│
├── 📂 streamlit_app/
│   ├── app.py                        # Main Streamlit application
│   ├── 📂 pages/
│   │   ├── 1_🏠_Home.py              # Home page
│   │   ├── 2_🔮_Prediction.py        # Image upload & prediction
│   │   ├── 3_📊_Model_Comparison.py  # Compare 3 models
│   │   └── 4_ℹ️_About.py             # About project
│   └── 📂 utils/
│       ├── model_loader.py           # Load trained models
│       ├── image_processor.py        # Image preprocessing
│       └── visualization.py          # Plot confusion matrix, etc.
│
├── 📂 src/                           # Source code modules
│   ├── __init__.py
│   ├── config.py                     # Configuration & constants
│   ├── data_loader.py                # Dataset loading utilities
│   ├── preprocessing.py              # Image preprocessing functions
│   └── evaluator.py                  # Model evaluation functions
│
├── 📂 static/
│   ├── 📂 images/                    # Images for README
│   │   ├── banner.png
│   │   ├── screenshot_home.png
│   │   ├── screenshot_prediction.png
│   │   ├── confusion_matrix_mobilenet.png
│   │   └── model_comparison.png
│   └── 📂 css/                       # Custom styling
│       └── style.css
│
├── 📂 results/                       # Evaluation results
│   ├── classification_reports.json   # All models metrics
│   ├── confusion_matrices.png        # Confusion matrix plots
│   └── comparison_table.csv          # Model comparison table
│
├── 📂 tests/                         # Unit tests (optional)
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Documentation (file ini)
├── 📄 .gitignore                     # Git ignore rules
├── 📄 LICENSE                        # MIT License
└── 📄 setup.py                       # Package setup (optional)
```

### File Descriptions:

**Core Files:**
- `UAP.ipynb` - Jupyter notebook berisi seluruh proses training dan evaluasi
- `app.py` - Main entry point untuk Streamlit dashboard
- `requirements.txt` - Semua dependencies yang dibutuhkan

**Model Files:**
- `cnn_base_model.keras` - Custom CNN (baseline)
- `mobilenet_model.keras` - **PRODUCTION MODEL** ⭐
- `resnet50_model.keras` - ResNet50 (experimental)

**Dataset Structure:**
```
data/garbage_classification/
├── cardboard/  (15 test images)
├── glass/      (28 test images)
├── metal/      (25 test images)
├── paper/      (44 test images)
├── plastic/    (35 test images)
└── trash/      (10 test images)
Total: 157 test images
```

---

## 📸 Screenshot Aplikasi

### 1. Home Page
<img width="1913" height="917" alt="image" src="https://github.com/user-attachments/assets/025a196a-d867-4211-9d42-f67ca4d04ab9" />

### 2. Prediction Page
MobileNetV2
<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/f14fb90c-34bf-42a0-b437-713246832452" />
CNN base
<img width="1915" height="914" alt="image" src="https://github.com/user-attachments/assets/5ff889cd-6c94-486e-a74f-0cf9759b84b4" />
ResNet50
<img width="1911" height="910" alt="image" src="https://github.com/user-attachments/assets/99e68932-bc7a-470e-8f36-b93c1a5dec13" />



### 3. Model Comparison
<img width="981" height="374" alt="image" src="https://github.com/user-attachments/assets/089117e6-a51f-48a4-ba70-eb73a18cd5fb" />
<img width="990" height="374" alt="image" src="https://github.com/user-attachments/assets/ab78d6da-1df7-48de-a691-0dacd367e19f" />
<img width="990" height="374" alt="image" src="https://github.com/user-attachments/assets/85001dca-1d08-486b-a9bd-331729833c5a" />
*Perbandingan performa ketiga model*

### 4. Confusion Matrix
<img width="508" height="547" alt="image" src="https://github.com/user-attachments/assets/e91442b5-a625-4261-81cf-24a1e89c1eef" />

---

## 🔧 Troubleshooting

### Problem: ModuleNotFoundError
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Problem: CUDA Out of Memory
**Solution:**
- Reduce batch size
- Use CPU instead: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

### Problem: Streamlit Port Already in Use
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### Problem: Model File Not Found
**Solution:**
- Pastikan model sudah di-download/training
- Check path di `config.py`

---

## 🚀 Future Improvements

### Priority 1: Meningkatkan Akurasi (Target 75%+)

#### A. Fine-tuning MobileNetV2
- [ ] Unfreeze top 30 layers dari MobileNetV2
- [ ] Train dengan learning rate sangat kecil (1e-5 atau 1e-6)
- [ ] Train 20-30 epochs lagi dengan early stopping
- [ ] **Expected improvement**: +10-15% accuracy

#### B. Data Augmentation Enhancement
```python
ImageDataGenerator(
    rotation_range=40,           # lebih agresif
    zoom_range=0.3,
    brightness_range=[0.5, 1.5], # variasi lighting
    channel_shift_range=0.2,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,           # tambahan
    fill_mode='reflect'
)
```

#### C. Handle Class Imbalance
- Dataset saat ini: 
  - Paper: 44 samples (banyak)
  - Trash: 10 samples (sedikit)
- **Solution**: Class weighting atau oversampling
```python
class_weight = {
    0: 2.0,  # carton (15 samples)
    1: 1.0,  # glass (28 samples)
    2: 1.2,  # metal (25 samples)
    3: 0.8,  # paper (44 samples)
    4: 1.0,  # plastic (35 samples)
    5: 3.0   # trash (10 samples)
}
```

---

### Priority 2: Optimization

#### A. Model Optimization
- [ ] Convert MobileNetV2 ke TensorFlow Lite
- [ ] Quantization (INT8) untuk reduce size 75%
- [ ] Pruning untuk faster inference
- [ ] **Expected**: Model size 3-4 MB, inference <50ms

#### B. Inference Speed
- [ ] Batch prediction untuk multiple images
- [ ] Model caching di production
- [ ] GPU acceleration di server
- [ ] **Target**: <100ms per prediction

---

### Priority 3: Feature Enhancement

#### A. Dashboard Features
- [ ] Batch upload (multiple images at once)
- [ ] Export predictions to CSV/Excel
- [ ] Confidence threshold filtering
- [ ] History of predictions with database
- [ ] User authentication system

#### B. Advanced Analytics
- [ ] Accumulated statistics (harian/mingguan)
- [ ] Real-time monitoring dashboard
- [ ] A/B testing framework untuk model comparison
- [ ] Model versioning system

---

### Priority 4: Deployment

#### A. Cloud Deployment
- [ ] Deploy ke **Streamlit Cloud** (gratis, mudah)
- [ ] Alternative: Heroku, Railway, Render
- [ ] Setup CI/CD dengan GitHub Actions
- [ ] Auto-deployment on push to main

#### B. API Development
```python
# FastAPI endpoint
@app.post("/predict")
async def predict(file: UploadFile):
    image = preprocess(file)
    prediction = model.predict(image)
    return {
        "class": class_names[prediction],
        "confidence": float(confidence),
        "all_probabilities": probs.tolist()
    }
```

#### C. Mobile App
- [ ] Flutter/React Native app
- [ ] TFLite integration untuk offline prediction
- [ ] Camera integration untuk real-time capture
- [ ] Upload ke cloud untuk accuracy tracking

---

### Priority 5: Dataset Expansion

#### Current Issues:
- ❌ Dataset kecil (157 test samples total)
- ❌ Class imbalance (trash: 10, paper: 44)
- ❌ Limited variety per class

#### Action Plan:
1. **Collect more data**
   - Target: 1000+ images per class
   - Sources: Public datasets, web scraping, manual collection
   
2. **Synthetic data generation**
   - GAN-based augmentation
   - Style transfer
   - Mix-up techniques

3. **Active learning**
   - Identify misclassified samples
   - Add similar samples to training
   - Iterative improvement

---

### Priority 6: Experiment Tracking

#### Setup MLOps Tools
- [ ] **Weights & Biases** untuk experiment tracking
- [ ] **DVC** untuk dataset version control
- [ ] **MLflow** untuk model registry
- [ ] Automated testing pipeline

#### Example W&B Integration:
```python
import wandb

wandb.init(project="garbage-classification")

history = model.fit(
    train_gen,
    callbacks=[WandbCallback()]
)
```

---

### Timeline & Roadmap

| Phase | Tasks | Timeline | Priority |
|-------|-------|----------|----------|
| **Phase 1** | Fine-tune MobileNetV2, Fix class imbalance | Week 1-2 | 🔴 High |
| **Phase 2** | Deploy to Streamlit Cloud, Add batch prediction | Week 3 | 🔴 High |
| **Phase 3** | Build REST API, Model optimization | Week 4-5 | 🟡 Medium |
| **Phase 4** | Mobile app development, Dataset expansion | Week 6-8 | 🟢 Low |
| **Phase 5** | MLOps setup, A/B testing | Week 9-10 | 🟢 Low |

---

### Research Opportunities

#### A. Advanced Architectures
- [ ] **EfficientNet** family (B0-B7)
- [ ] **Vision Transformer** (ViT) for comparison
- [ ] **ConvNeXt** - modern CNN architecture
- [ ] **Ensemble methods** - combine multiple models

#### B. Novel Techniques
- [ ] **Contrastive learning** (SimCLR, MoCo)
- [ ] **Self-supervised pre-training** on garbage domain
- [ ] **Few-shot learning** untuk new waste categories
- [ ] **Meta-learning** untuk fast adaptation

#### C. Multi-modal Learning
- [ ] Combine image + metadata (size, weight, texture)
- [ ] Use CLIP for zero-shot classification
- [ ] Add text descriptions for better context

---

### Success Metrics

**Short-term (1-2 bulan):**
- ✅ Accuracy > 75% on test set
- ✅ Inference time < 100ms
- ✅ Web app deployed and accessible
- ✅ 100+ successful predictions logged

**Mid-term (3-6 bulan):**
- ✅ Accuracy > 85%
- ✅ Mobile app released
- ✅ API serving 1000+ requests/day
- ✅ Dataset expanded to 5000+ images

**Long-term (6-12 bulan):**
- ✅ Accuracy > 90%
- ✅ Real-time video classification
- ✅ Deployed in actual waste sorting facility
- ✅ Published research paper/technical blog

---

## 📚 Referensi

1. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv preprint arXiv:1409.1556*.

2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520.

3. Keras Documentation: https://keras.io/
4. TensorFlow Documentation: https://www.tensorflow.org/
5. Streamlit Documentation: https://docs.streamlit.io/

---

## 👨‍💻 Kontributor

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Msyfrdnn09.png" width="100px;" alt=""/>
      <br />
      <sub><b>Muhammad Syafruddin</b></sub>
      <br />
      <sub>NIM: 2022-007</sub>
    </td>
  </tr>
</table>

### Informasi Kontak:
- **Email**: [msyfrdnn09@webmail.umm.ac.id]
- **GitHub**: [@Msyfrdnn09](https://github.com/Msyfrdnn09)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Muhammad Syafruddin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- Dataset dari **[Kaggel (https://www.kaggle.com/datasets/manonstr/tipe-webscraping?resource=download-directory)]**
- Inspirasi arsitektur dari paper-paper terkait
- Komunitas Stack Overflow dan GitHub
- Universitas **[Universitas Muhammadiyah Malang]**

---

## 📞 Support

Jika Anda menemukan bug atau memiliki saran, silakan:
1. Buka [Issue](https://github.com/Msyfrdnn09/UAP_Muhammad-Syafruddin_2022-007/issues)
2. Atau hubungi via email

---

<div align="center">
  
### ⭐Ujian Akhir Praktikum ! ⭐

**Made with ❤️ by Muhammad Syafruddin**


[⬆ Back to Top](#-garbage-classification-dashboard)

</div>
