# Stable Audio 3 — Local Web UI

Generate music and sound effects locally using Stability AI's Stable Audio 3 models. No API costs, no usage limits, runs entirely on your machine.

---

## Before You Start

You need three things installed before anything else:

- **Python 3.10 or 3.11** — download from https://python.org. During installation check "Add Python to PATH"
- **Git** — download from https://git-scm.com/download/win
- **A Hugging Face account** — free at https://huggingface.co

---

## Step 1 — Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token** → select **User Access Token (classic)** → Role: **Read** → Generate
3. Copy the token somewhere safe

Then accept the license (click "Agree and access repository" while logged in):

- https://huggingface.co/stabilityai/stable-audio-3-small-music
- https://huggingface.co/stabilityai/stable-audio-3-small-sfx
- https://huggingface.co/stabilityai/stable-audio-3-medium

---

## Step 2 — Create and Activate a Virtual Environment

**Windows:**

```
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**

```
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line. Keep this active for all steps below.

---

## Step 3 — Install PyTorch

Choose ONE depending on your setup:

**NVIDIA GPU (recommended for speed):**

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (no NVIDIA GPU):**

```
pip install torch torchaudio
```

---

## Step 4 — Install All Dependencies

Run these in order:

```
pip install setuptools --upgrade
pip install pandas
pip install -r requirements.txt
pip install "stable-audio-tools @ git+https://github.com/Stability-AI/stable-audio-tools.git" --no-deps --ignore-requires-python
pip install "k-diffusion @ git+https://github.com/crowsonkb/k-diffusion.git" --no-deps
```

If you installed the NVIDIA GPU version of PyTorch, also run this:

```
pip uninstall torchvision -y
```

---

## Step 5 — Apply Compatibility Patch

There is a bug in stable-audio-tools that causes a crash on newer NumPy versions. Run this entire block to fix it:

**Windows:**

```
python -c "import pathlib; p = pathlib.Path('venv/Lib/site-packages/stable_audio_tools/inference/generation.py'); txt = p.read_text(); txt = txt.replace('np.random.randint(0, 2**32 - 1)', 'np.random.randint(0, 2**31 - 1)'); p.write_text(txt); print('Patched OK')"
```

**Mac / Linux:**

```
python -c "import pathlib; p = pathlib.Path('venv/lib/python3.11/site-packages/stable_audio_tools/inference/generation.py'); txt = p.read_text(); txt = txt.replace('np.random.randint(0, 2**32 - 1)', 'np.random.randint(0, 2**31 - 1)'); p.write_text(txt); print('Patched OK')"
```

It should print `Patched OK`. If it says the file was not found, check the Python version in the path (e.g. `python3.10` instead of `python3.11`).

---

## Step 6 — Add Your Hugging Face Token

In the project folder, find the file called `.env.example`. Copy it and rename the copy to `.env`. Open `.env` in any text editor and replace:

```
HF_TOKEN=hf_your_token_here
```

with your actual token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

Save the file.

---

## Step 7 — Run the App

Make sure your virtual environment is still active (you see `(venv)` in the terminal). Then:

**GPU mode:**

```
python app.py
```

Open your browser at: **http://localhost:5000**

**CPU mode:**

```
python cpu.py
```

Open your browser at: **http://localhost:5001**

The first time you generate with each model it will download the weights automatically. This takes a few minutes depending on your connection. After that everything loads from local disk instantly.

---

## Models

| Model       | Parameters | Max Duration | Best For                            |
| ----------- | ---------- | ------------ | ----------------------------------- |
| Small Music | 459M       | 2 minutes    | Music tracks and compositions       |
| Small SFX   | 459M       | 2 minutes    | Sound effects and foley             |
| Medium      | 1.4B       | 6 min 20 sec | Higher quality music, longer tracks |

**GPU recommendation:** Small and Small SFX run fine on CPU. Medium is usable on CPU but slow — recommend 8GB VRAM GPU for Medium.

---

## Generation Tips

- Start with **8 inference steps** for speed. Go up to **32** for highest quality
- Be specific and descriptive for SFX prompts
- All generated files are saved as WAV in the `outputs/` folder

---

## Troubleshooting

**`No module named 'k_diffusion'`**

```
pip install "k-diffusion @ git+https://github.com/crowsonkb/k-diffusion.git" --no-deps
```

**`high is out of bounds for int32`** — run the patch in Step 5 again.

**`operator torchvision::nms does not exist`**

```
pip uninstall torchvision -y
```

**`TorchCodec is required`**

```
pip install soundfile
```

**Medium model crashes on GPU** — Close other GPU apps (especially Ollama), increase Windows virtual memory to 16384 MB max (`Win+R` → `sysdm.cpl` → Advanced → Performance → Virtual Memory), then restart and try again.

**Generation is slow** — You're likely on CPU mode. Confirm CUDA is working:

```
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If it prints `False`, reinstall PyTorch using the CUDA command in Step 3.
