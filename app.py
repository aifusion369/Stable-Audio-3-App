import os
import uuid
import threading
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

jobs = {}
_model_cache = {}
_model_lock = threading.Lock()

MODEL_IDS = {
    "small-music": "stabilityai/stable-audio-3-small-music",
    "small-sfx":   "stabilityai/stable-audio-3-small-sfx",
    "medium":      "stabilityai/stable-audio-3-medium",
}

MODEL_MAX_DURATION = {
    "small-music": 120,
    "small-sfx":   120,
    "medium":      380,
}


def get_model(model_key, device):
    import torch
    from stable_audio_tools import get_pretrained_model
    from huggingface_hub import login

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set. Copy .env.example to .env and add your token.")

    hf_id = MODEL_IDS[model_key]

    with _model_lock:
        if model_key in _model_cache:
            return _model_cache[model_key]

        for k in list(_model_cache.keys()):
            del _model_cache[k]
            if device == "cuda":
                torch.cuda.empty_cache()

        login(token=HF_TOKEN, add_to_git_credential=False)

        model, model_config = get_pretrained_model(hf_id)

        if device == "cuda":
            model = model.to(torch.float16)
            model = model.to(device)
        else:
            model = model.to(device)

        _model_cache[model_key] = (model, model_config)
        return model, model_config


def run_generation(job_id, prompt, duration, steps, cfg_scale, model_key):
    try:
        import torch
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        jobs[job_id]["device"] = device
        jobs[job_id]["status"] = "loading_model"

        model, model_config = get_model(model_key, device)

        sample_rate = model_config["sample_rate"]
        model_sample_size = model_config["sample_size"]

        duration = min(duration, MODEL_MAX_DURATION[model_key])
        calculated_size = int(duration * sample_rate)
        sample_size = min(calculated_size, model_sample_size, 2_147_483_647)

        conditioning = [{"prompt": prompt, "seconds_total": duration}]
        jobs[job_id]["status"] = "generating"

        output = generate_diffusion_cond_inpaint(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type="pingpong",
            device=device
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = (
            output.to(torch.float32)
                  .div(torch.max(torch.abs(output)))
                  .clamp(-1, 1)
                  .cpu()
        )

        filename = f"{job_id}.wav"
        filepath = os.path.join(OUTPUTS_DIR, filename)
        sf.write(filepath, output.numpy().T, sample_rate, subtype="PCM_16")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["filename"] = filename
        jobs[job_id]["model_key"] = model_key

    except Exception as e:
        import traceback
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt    = data.get("prompt", "").strip()
    duration  = int(data.get("duration", 30))
    steps     = int(data.get("steps", 8))
    cfg_scale = float(data.get("cfg_scale", 1.0))
    model_key = data.get("model", "small-music")

    if model_key not in MODEL_IDS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "filename": None, "error": None, "model_key": model_key, "device": None}

    thread = threading.Thread(
        target=run_generation,
        args=(job_id, prompt, duration, steps, cfg_scale, model_key),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/history")
def history():
    files = []
    for f in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
        if f.endswith(".wav"):
            path = os.path.join(OUTPUTS_DIR, f)
            files.append({
                "filename": f,
                "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2)
            })
    return jsonify(files)


if __name__ == "__main__":
    if not HF_TOKEN:
        print("\n ERROR: HF_TOKEN not found.")
        print(" Copy .env.example to .env and add your Hugging Face token.\n")
        exit(1)
    print(f"\n GPU mode — open http://localhost:5000\n")
    app.run(debug=True, port=5000)
