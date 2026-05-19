# Plant Disease Classifier

A web app that classifies plant leaf diseases using **CLIP embeddings** and a small **MLP classifier**. No deep model training required — runs on CPU.

---

## Why CLIP instead of a traditional classifier?

A traditional CNN classifier learns visual features from scratch on domain-specific data, requiring thousands of labeled images and GPU training time. CLIP (Contrastive Language–Image Pretraining) is pretrained on 400M image–text pairs and already understands rich visual concepts. We exploit this by:

1. Running images through CLIP's frozen vision encoder to get a 768-d embedding
2. Training only a **small MLP head** on top of those embeddings (10 classes × 500 images, minutes on CPU)

The result is a model that generalises with minimal data and zero GPU training.

---

## Model limitations

This is a portfolio prototype — results should be interpreted with caution:

- **Supported crops only:** Apple, Corn (maize), Grape, Potato, Tomato. Other plants will still produce a prediction — it will just be wrong.
- **Leaf images only:** The model was trained on close-up, isolated leaf scans (PlantVillage dataset). Field photos, fruit images, or full-plant shots are out of distribution.
- **Known Corn bias:** The model has a tendency to predict Corn (maize) classes for ambiguous inputs. Low-confidence Corn predictions should be treated with extra scepticism.
- **Not fine-tuned:** The CLIP backbone is frozen. Only the MLP head is trained on plant data.

---

## Architecture

```
[Browser]
    │  multipart/form-data POST /predict
    ▼
[FastAPI + SlowAPI rate limiter]
    │
    ▼
[Image Preprocessing]  ←  PIL resize + CLIP transform
    │
    ▼
[CLIP ViT-L/14 Encoder]  →  768-d L2-normalised embedding
    │           │
    ▼           ▼
[MLP Classifier]    [FAISS IndexFlatIP]
(best of GridSearchCV) (cosine nearest neighbours)
    │           │
    ▼           ▼
 {predicted_class,   [similar image paths]
  confidence,
  top_k}
    │
    ▼
[HTML + Tailwind UI]
  • sidebar with clickable sample images (pics/)
  • predicted label + confidence bar
  • top-k probability bars
  • nearest-neighbour image gallery
```

---

## Supported Classes (10)

| Class | Type |
|---|---|
| Apple — Apple scab | Disease |
| Corn — Common rust | Disease |
| Corn — Healthy | Healthy |
| Grape — Black rot | Disease |
| Potato — Early blight | Disease |
| Potato — Late blight | Disease |
| Potato — Healthy | Healthy |
| Tomato — Early blight | Disease |
| Tomato — Late blight | Disease |
| Tomato — Healthy | Healthy |

---

## Quick Start (local)

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cpu` so PyTorch installs correctly without a separate command. If you have a GPU, replace `cpu` in that line with your CUDA version (e.g. `cu121`) before installing.

### 2. Prepare data and models

Downloads ~5 000 PlantVillage images from HuggingFace, trains the MLP classifier with hyperparameter search, and builds the FAISS index. Run once:

```bash
python scripts/prepare_data.py
```

Expected output:
```
Loading CLIP ViT-L-14...
Downloading BrandonFors/Plant-Diseases-PlantVillage-Dataset...
Collecting up to 500 images/class (5000 total)...
Split — train: 4000, test: 1000
Balanced train set: XXXX samples
Running GridSearchCV over MLP hyperparameters (3-fold CV)...
Best params:   {'alpha': ..., 'hidden_layer_sizes': ...}
Best CV acc:   0.XXX
Test accuracy: 0.XXX
Building FAISS index on gallery embeddings...
Done. Models saved to models/
```

### 3. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser. The sidebar on the left shows sample images from `pics/` — click any to run an instant prediction.

---

## VPS Deployment (Ubuntu 20+)

Any VPS with **2 GB RAM** and Ubuntu 20.04+ is sufficient.

### 1. Clone and prepare

```bash
git clone https://github.com/YOUR_USERNAME/plant-classifier-app.git
cd plant-classifier-app
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py
```

### 2. Run the app

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

To keep it running after logout, use a systemd service or `tmux`/`screen`. For systemd, create `/etc/systemd/system/plant-classifier.service`:

```ini
[Unit]
Description=Plant Classifier
After=network.target

[Service]
User=YOUR_USER
WorkingDirectory=/path/to/plant-classifier-app
ExecStart=/path/to/plant-classifier-app/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now plant-classifier
```

### 3. Nginx reverse proxy

Create `/etc/nginx/sites-available/plant-classifier` and fill in `server_name` with your domain:

```nginx
server {
    listen 80;
    server_name your-domain.com;          # <-- change this

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 30s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/plant-classifier /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 4. TLS with Let's Encrypt (optional)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Technical Notes

### CLIP Embeddings

CLIP's ViT-L/14 encoder maps any image to a 768-dimensional vector. Images of the same visual concept cluster together in this space regardless of lighting, background, or zoom level. ViT-L/14 was chosen over the smaller ViT-B/32 (512-d) for better separation of visually similar disease classes.

### MLP Classifier

A small MLP trained on frozen CLIP features. `scripts/prepare_data.py` runs a `GridSearchCV` over hidden layer sizes `[(256,), (256,128), (512,256)]` and regularisation strengths `[1e-4, 1e-3, 1e-2]` with 3-fold cross-validation, then evaluates the best configuration on a held-out 20% test split. The training set is oversampled to equalise class sizes before fitting, which reduces the known bias toward majority classes.

### FAISS Retrieval

`IndexFlatIP` performs exact inner-product search over L2-normalised vectors, equivalent to cosine similarity. The 5 nearest neighbours retrieved for each query provide visual explanations: "your image looks most like these examples." The retrieval index is built on the first 100 gallery images per class; the remaining 400 per class are used only for classifier training.

### Sample Sidebar

The sidebar is populated dynamically at page load by calling `GET /pics-list`, which reads the `pics/` directory. To change the sample images, drop or remove files in `pics/` and reload the page — no code changes needed.

---

## Project Structure

```
plant-classifier-app/
├── app/
│   ├── main.py          FastAPI app, /predict and /pics-list endpoints, rate limiting
│   ├── model.py         CLIP ViT-L/14 wrapper (encode_image / encode_text)
│   ├── classifier.py    MLP inference
│   ├── retrieval.py     FAISS nearest-neighbour search
│   └── utils.py         Image loading helpers
├── frontend/
│   ├── index.html       Single-page UI with sidebar and warning banner
│   ├── app.js           Fetch + render logic, dynamic sidebar from pics/
│   └── style.css        Minimal overrides (Tailwind via CDN)
├── static/
│   └── sample_images/   Gallery images (generated by prepare_data.py)
├── pics/                Sample images shown in the sidebar (user-managed)
├── data/
│   ├── labels.txt       Selected class names (reference, not read by app)
│   └── prompts.json     Text prompts per class (reference, not read by app)
├── models/              Artefacts (generated by prepare_data.py)
├── scripts/
│   └── prepare_data.py  One-time data download + model training
└── requirements.txt
```

---

## Rate Limiting

The `/predict` endpoint is limited to **10 requests per minute** per IP via SlowAPI. Adjust in `app/main.py`:

```python
@limiter.limit("10/minute")
```
