# CAPTCHA Solver

A complete, end‑to‑end framework for building and deploying a **Transformer‑based** text CAPTCHA solver (Microsoft TrOCR).

## Contents
- [Project Overview](#project-overview)
- [File‑by‑File Guide](#file-by-file-guide)
- [Installation](#installation)
- [Running the Prediction Server](#running-the-prediction-server)
- [Chrome Extension](#chrome-extension)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Labeling Real‑World CAPTCHAs](#labeling-real-world-captchas)
- [Typical Workflows](#typical-workflows)
- [License](#license)

## Project Overview

The repository addresses accessibility challenges posed by text‑based CAPTCHAs:

* **Data generation** – create synthetic, highly‑distorted images.  
* **Distributed fine‑tuning** – adapt TrOCR with multi‑GPU training.  
* **Serving** – lightweight Flask API delivering top‑3 predictions.  
* **User interface** – a Chrome extension (packaged separately) that lets users draw a rectangle around any CAPTCHA and auto‑fills the best guess.

## File‑by‑File Guide

### `generator.py`
Generates synthetic CAPTCHA images with:
* gradient backgrounds,
* per‑character rotation, overlap, drop shadows,
* random noise (lines, dots, arcs).

**Usage**

```bash
python generator.py          # creates 15 000 images → ./generated_captchas
```

Edit the call to `generate_captcha_batch()` at the bottom of the file to change the amount or visual parameters.

### `data_split.py`
Shuffles the images in **`generated_captchas/`** and creates a `dataset/` directory with **train / val / test** subfolders.  
Each split receives a `labels.txt` containing `filename text`.

**Usage**

```bash
python data_split.py         # 80 / 10 / 10 split
```

### `label_captachs.py`
Uses an already‑trained TrOCR model (expected in **`./trocr-finetuned-captcha`**) to auto‑label raw PNG images.
Designed for labelling new raw data with partially trained model when including more data :

1. Drop images into **`./saved_captchas/`**.  
2. Run the script; correctly predicted images are copied to **`./labeled_captchas/`**, renamed to the predicted text.

**Usage**

```bash
python label_captachs.py
```

### `train_ddp.py`
Distributed training script using **PyTorch DDP**.  
Key features: AMP, checkpointing, static‑graph optimizations, early stopping.

*Hyper‑parameters (`num_epochs`, `batch_size`, learning‑rate, etc.) are set at the top of the file – edit them directly.*

**Usage**

```bash
python train_ddp.py          # spawns one process per visible GPU
```

Checkpoints and the final model are saved in **`./trocr-finetuned-captcha/`**.

### `test.py`
Evaluates a trained model on the **test** split prepared by `data_split.py`.  
Prints four accuracy metrics (top‑1 / any‑match × case‑sensitive / insensitive).

**Usage**

```bash
python test.py
```

### `server.py`
Lightweight Flask app exposing a single endpoint:

```
POST /solve
Form-data:  image=<PNG/JPEG>
Returns:  { "captcha_image": "<base64>", "predictions": [ { "prediction": "n4Bc", "confidence": 0.82 }, … ] }
```

**Usage**

```bash
python server.py             # binds to 0.0.0.0:5000
```

The Chrome extension talks to this endpoint.

### Model Folder – `trocr-finetuned-captcha/`
Created automatically by `train_ddp.py`. Holds the Hugging Face encoder‑decoder weights and tokenizer.

## Chrome Extension

The extension offers a **point‑and‑click** UI for solving CAPTCHAs in the browser.

### Installation

1. Unzip **`extension.zip`** so you have an `extension/` folder.  
2. Open Chrome and navigate to `chrome://extensions`.  
3. Enable **Developer mode** (toggle in top‑right).  
4. Click **Load unpacked** and select the `extension/` directory.  

### Usage

1. **Start the prediction server** locally:  
   ```bash
   python server.py
   ```  
2. Navigate to any page containing a text CAPTCHA.  
3. Click the extension icon; the cursor turns into a crosshair.  
4. Drag a rectangle around the CAPTCHA image.  
5. A popup lists up to three predictions with confidence scores.  
6. Click a prediction to copy it to the clipboard (and auto‑fill if the input field is focused).  

The extension makes a one‑off HTTP request; no images or predictions are stored.

## Installation

```bash
git clone https://github.com/<your-handle>/captcha-solver.git
cd captcha-solver
pip install -r requirements.txt
```

It is recommanded to follow instructions from [PyTorch](pytorch.org) to install `torch` `torchvision` `torchaudio` prior to run the commands above for best compatibility. 

Note that this project was built and tested on Linux and Nvidia GPUs with CUDA support only and may not work on other systems. 

## Running the Prediction Server

```bash
python server.py
```

The API will listen on **`http://0.0.0.0:5000/solve`** (modifiable inside the script).

## Training Pipeline

```bash
python generator.py          # 15k synthetic images
python data_split.py         # make train/val/test
python train_ddp.py          # fine‑tune (multi‑GPU if available)
```

## Evaluation

```bash
python test.py
```

## Labeling Real‑World CAPTCHAs

```bash
# 1. Place samples in ./saved_captchas
python label_captachs.py
# ➜ labeled images → ./labeled_captchas
```

Manually adjust the incorrect labels, then merge the labeled set back into `dataset/train/` and run `train_ddp.py` again for semi‑supervised fine‑tuning. 

## Typical Workflows

### Quick Inference Only

1. Train a model into **`./trocr-finetuned-captcha`**.  
2. `pip install -r requirements.txt`  
3. `python server.py`  
4. Load the Chrome extension (unzip `extension.zip` → `chrome://extensions` → **Load unpacked**).

### Full Re‑Training

```bash
python generator.py
python data_split.py
python train_ddp.py
```

## License

Released under the **MIT License**.  
TrOCR © Microsoft; redistributed under the terms of its original license.