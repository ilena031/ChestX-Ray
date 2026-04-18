
# Optimalisasi Klasifikasi X-Ray Menggunakan Medical Stable Diffusion dan Dual Feature Aggregation

> **KCV Lab Selection Project — Institut Teknologi Sepuluh Nopember, 2026**

---

## Overview

Proyek ini mengklasifikasikan citra chest X-ray ke dalam **6 kelas** menggunakan pipeline hybrid yang memanfaatkan **frozen Stable Diffusion U-Net** sebagai feature extractor, diikuti oleh modul **Dual Feature Aggregation (DFATB + FAFN + Differential Denoising)** dan **MLP classification head**.

### Kelas Target
| # | Kelas | Deskripsi |
|---|-------|-----------|
| 0 | Atelectasis | Kolaps paru parsial |
| 1 | Effusion | Cairan di rongga pleura |
| 2 | Infiltration | Substansi abnormal di jaringan paru |
| 3 | No Finding | Normal / tidak ada temuan klinis |
| 4 | Nodule | Massa kecil di paru |
| 5 | Pneumothorax | Udara di rongga pleura |

---

##  End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Chest X-Ray (512×512)                │
│                        + Text Prompt ("A chest X-ray")             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 1: LATENT ENCODING (Frozen VAE)                            │
│  ─────────────────────────────────────                            │
│  • Image → VAE Encoder → latent z ∈ ℝ^(4×64×64)                 │
│  • Gaussian noise injection at timestep t=10                      │
│  • noisy_latent = scheduler.add_noise(z, noise, t)               │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 2: U-NET FEATURE EXTRACTION (Frozen SD v1.4 + LoRA)       │
│  ─────────────────────────────────────────────────────────        │
│  • Single forward pass through frozen U-Net                       │
│  • BlockFeatureCollector: hooks on up_blocks → 4D feature maps   │
│    [B, C, H, W] for each resolution level                        │
│  • AttentionCollector: custom cross-attention processor captures  │
│    spatial attention maps from text-image alignment               │
└──────────┬────────────────────────────────┬───────────────────────┘
           │ Feature Maps                   │ Attention Maps
           ▼                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 3: DUAL FEATURE AGGREGATION (FA Module — Trainable)        │
│  ─────────────────────────────────────────────────────────        │
│                                                                   │
│  ┌─────────────┐   ┌──────────┐   ┌──────────────────────┐       │
│  │   DFATB      │ → │  FAFN    │ → │ Differential         │       │
│  │ (Dual-Focus  │   │ (Feed-   │   │ Denoising            │       │
│  │  Attention   │   │  forward │   │ (attention-guided     │       │
│  │  Block)      │   │  Gating) │   │  noise suppression)  │       │
│  └─────────────┘   └──────────┘   └──────────┬───────────┘       │
│  • SpatialAttn                                │                   │
│  • ChannelAttn                    ┌───────────▼───────────┐       │
│  • BatchNorm                      │  GAP + Concat         │       │
│                                   │  → BottleneckProj     │       │
│                                   │  → z ∈ ℝ^128          │       │
│                                   └───────────────────────┘       │
└───────────────────────────────┬───────────────────────────────────┘
                                │ z (128-dim)
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 4: MLP CLASSIFICATION HEAD                                 │
│  ────────────────────────────────                                 │
│  LayerNorm(128) → Linear(128, 512) → GELU → Dropout(0.3)        │
│  → Linear(512, 6) → softmax → prediction                        │
└───────────────────────────────────────────────────────────────────┘
```

---

##  Experimental Scenarios

Pipeline dievaluasi dalam **4 skenario** yang merupakan kombinasi dari dua dimensi:

| Scenario | Dataset | Augmentation | Deskripsi |
|----------|---------|-------------|-----------|
| **1** | Balanced (~416/kelas) | ❌ None | Baseline ideal — distribusi seimbang, tanpa augmentasi |
| **2** | Balanced (~416/kelas) | ✅ FSA | Uji apakah FSA meningkatkan generalisasi pada data seimbang |
| **3** | Imbalanced (10:1 skew) | ❌ None | Simulasi kondisi medis nyata — class weights menangani skew |
| **4** | Imbalanced (10:1 skew) | ✅ FSA | Worst-case scenario — imbalance + augmentasi |

Setiap skenario dijalankan untuk **4 feature extractor** yang dibandingkan:

| Feature Extractor | Tipe | Deskripsi |
|-------------------|------|-----------|
| **MedSD (FE+FA)** | Generative Prior | Medical SD v1.4 + LoRA + Dual Feature Aggregation *(proposed)* |
| **DINOv2** | Vision Transformer | Self-supervised ViT dari Meta AI |
| **ConvNeXtV2** | CNN | Modern ConvNet dari Facebook/Meta |
| **MaxViT** | Hybrid CNN+ViT | Multi-axis attention dari Google |

### Feature Space Augmentation (FSA) — 3-Stage Pipeline

Diterapkan secara **berurutan** di feature space saat training (Scenario 2 & 4):

1. **Feature Space SMOTE** — Oversample kelas minoritas via interpolasi k-NN di ruang fitur
2. **Gaussian Noise Injection** — Perturbasi `N(0, 0.01²)` ke seluruh batch (asli + sintetis)
3. **Mixup** — Interpolasi konveks antar sampel `λ ~ Beta(0.2, 0.2)` → soft label

---

##  Project Structure

```
ChestX-Ray/
├── app/                          # Next.js frontend
│   ├── page.tsx                  #   Homepage / overview
│   ├── methodology/page.tsx      #   Pipeline & scenario descriptions
│   ├── results/page.tsx          #   Tables, charts, confusion matrices
│   ├── model/page.tsx            #   Architecture documentation
│   ├── inference/page.tsx        #   Live demo (upload X-ray → predict)
│   ├── team/page.tsx             #   Research team
│   ├── references/page.tsx       #   Bibliography
│   ├── globals.css               #   Design system
│   └── layout.tsx                #   Root layout + metadata
│
├── backend/                      # FastAPI inference server
│   ├── app.py                    #   Server entry + /health, /warmup, /predict
│   ├── inference.py              #   MedicalXRayPipeline orchestrator
│   ├── models.py                 #   FA module, MLP head, hooks, utilities
│   ├── colab_run.py              #   Google Colab deployment script (pyngrok)
│   ├── Dockerfile                #   Backend container
│   └── requirements.txt          #   Python dependencies
│
├── research/                     # Pipeline research files & trained weights
│   ├── fp-medical-banun.py       #   Medical SD feature extraction pipeline
│   ├── fp-mlp-classifier.py      #   MLP training + FSA + evaluation
│   ├── Medical X-ray Stable Diffusion_feature_map_extractor.py
│   │                             #   Core SD feature map extraction logic
│   ├── fa_best_med_balanced_1.pt #   FA encoder weights (trained)
│   └── best_overall_weights.pt   #   MLP classifier weights
│
├── components/                   # Shared React components
│   ├── Navbar.tsx
│   ├── Footer.tsx
│   └── SectionHeader.tsx
│
├── public/                       # Static assets
│   ├── charts/                   #   Result visualizations (barchart, radar, CM)
│   └── team/                     #   Team member photos
│
├── docker-compose.yml            # Full-stack orchestration
├── Dockerfile.frontend           # Frontend container
├── .env.local                    # API URL config (local/ngrok)
├── .gitignore
└── README.md                     # ← You are here
```

---

##  Research Files Reference

| File | Deskripsi |
|------|-----------|
| **`fp-medical-banun.py`** | Pipeline utama untuk ekstraksi fitur menggunakan Medical Stable Diffusion (SD v1.4 + LoRA). Menjalankan forward pass melalui frozen U-Net, mengumpulkan feature maps via `BlockFeatureCollector`, dan melatih modul Feature Aggregation (DFATB + FAFN + DiffDenoising). Output: file `.pt` checkpoint FA. |
| **`fp-mlp-classifier.py`** | Training dan evaluasi MLP classification head. Membaca CSV fitur yang diekstrak, menjalankan 4 skenario (balanced/imbalanced × ±FSA) untuk setiap feature extractor. Includes: 3-stage FSA (FS-SMOTE, Gaussian Noise, Mixup), class weighting, dan comprehensive evaluation (Acc, F1-macro, AUC-OvR). |
| **`Medical X-ray Stable Diffusion_feature_map_extractor.py`** | Core extraction logic — preprocessing gambar, VAE encoding, noise injection at timestep, U-Net forward pass dengan hook attachment, dan attention map collection. Digunakan oleh `fp-medical-banun.py`. |
| **`fa_best_med_balanced_1.pt`** | Trained Feature Aggregation model checkpoint (~315 MB). Berisi: `fa_model_state`, hyperparameters (channels, z_dim), dan metadata. |
| **`best_overall_weights.pt`** | MLP classifier weights (~274 KB). State dict berisi `net.0` (LayerNorm), `net.1` (Linear 128→512), `net.4` (Linear 512→6). |

---

##  Deployment

### Local Development
```bash
# Frontend
npm install && npm run dev       # http://localhost:3000

# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Google Colab (GPU)
1. Upload `backend/` dan `research/` ke Colab
2. Copy isi `backend/colab_run.py` ke cell Colab
3. Set `NGROK_AUTH_TOKEN` dari [ngrok dashboard](https://dashboard.ngrok.com)
4. Jalankan cell — salin URL ngrok ke `.env.local`
5. Restart `npm run dev`

### Docker
```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## 👥 Team

| Nama | Role | Kontribusi |
|------|------|------------|
| Syahribanun | Lead Researcher | Research design · pipeline development · FE+FA implementation · writeup |
| Ahmad Naufal Farras | Researcher | Classification model development · Feature Extraction module implementation · Model & web deployment |

**Institution:** Departemen Teknik Informatika, Lab Komputasi Cerdas dan Visi (KCV), ITS Surabaya
