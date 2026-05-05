"""
=============================================================================
app.py – Flask Web GUI for Thyroid Nodule Detection
=============================================================================
"""

import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

from inference import predict_single

app = Flask(__name__)
CORS(app)


def file_to_bgr(file_storage):
    """Convert uploaded file to BGR numpy array."""
    img_bytes = file_storage.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    bgr       = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


def ndarray_to_b64(arr, is_rgb=False):
    """Encode numpy image to base64 PNG."""
    if not is_rgb:
        pil = Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        pil = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ─────────────────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>ThyroSegment – Thyroid Nodule Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
    <style>
        :root {
            --bg: #0f1419;
            --bg2: #1a202c;
            --accent: #00d4ff;
            --accent2: #00ff9d;
            --warn: #ff6b35;
            --text: #e8f4fd;
            --muted: #7a9bb5;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            padding: 24px 32px;
            border-bottom: 1px solid #2a3f5f;
            background: linear-gradient(90deg, var(--bg) 0%, var(--bg2) 50%, var(--bg) 100%);
        }
        
        .logo {
            font-family: 'Syne', sans-serif;
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        main {
            flex: 1;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 0;
            overflow: hidden;
        }
        
        .left-panel {
            background: var(--bg2);
            border-right: 1px solid #2a3f5f;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 24px;
        }
        
        .section {
            margin-bottom: 24px;
        }
        
        .section-title {
            font-family: 'Syne', sans-serif;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2px;
            color: var(--muted);
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        
        .upload-zone {
            border: 2px dashed #2a3f5f;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: rgba(0, 212, 255, 0.02);
        }
        
        .upload-zone:hover {
            border-color: var(--accent);
            background: rgba(0, 212, 255, 0.06);
        }
        
        .upload-zone input {
            display: none;
        }
        
        .upload-icon { font-size: 36px; margin-bottom: 12px; }
        .upload-text { font-size: 12px; color: var(--muted); line-height: 1.6; }
        .upload-text strong { color: var(--text); }
        
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            color: var(--bg);
            font-family: 'Syne', sans-serif;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            margin-top: 12px;
        }
        
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,212,255,0.3); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        
        .metric-card {
            background: rgba(0, 212, 255, 0.08);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-label { font-size: 11px; color: var(--muted); }
        .metric-value { font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700; color: var(--accent); }
        
        .verdict {
            background: rgba(255, 56, 96, 0.08);
            border: 1px solid rgba(255, 56, 96, 0.3);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            margin-top: 12px;
            display: none;
        }
        
        .verdict.nodule { display: block; }
        .verdict-icon { font-size: 32px; margin-bottom: 8px; }
        .verdict-title { font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 700; color: #ff6b7a; margin-bottom: 6px; }
        .verdict-sub { font-size: 11px; color: var(--muted); }
        
        .verdict.clear {
            background: rgba(0, 255, 157, 0.06);
            border-color: rgba(0, 255, 157, 0.3);
        }
        
        .verdict.clear .verdict-title { color: var(--accent2); }
        
        .right-panel {
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background: var(--bg);
        }
        
        .tabs {
            display: flex;
            gap: 0;
            border-bottom: 1px solid #2a3f5f;
            background: var(--bg2);
            padding: 0 24px;
        }
        
        .tab-btn {
            padding: 14px 20px;
            background: none;
            border: none;
            color: var(--muted);
            font-family: 'Syne', sans-serif;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            text-transform: uppercase;
        }
        
        .tab-btn:hover { color: var(--text); }
        .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
        
        .canvas-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        
        .image-card {
            background: var(--bg2);
            border: 1px solid #2a3f5f;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .image-card-title {
            padding: 10px;
            font-size: 10px;
            color: var(--muted);
            border-bottom: 1px solid #2a3f5f;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .image-card img, .image-card canvas {
            width: 100%;
            display: block;
        }
        
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            gap: 16px;
            color: var(--muted);
            text-align: center;
        }
        
        .empty-icon { font-size: 64px; opacity: 0.3; }
        .empty-title { font-size: 18px; color: var(--text); opacity: 0.4; }
    </style>
</head>
<body>

<header>
    <div class="logo">🔬 ThyroSegment</div>
</header>

<main>
    <!-- LEFT PANEL -->
    <div class="left-panel">
        <div class="section">
            <div class="section-title">Input Image</div>
            <div class="upload-zone" id="uploadZone">
                <input type="file" id="fileInput" accept="image/*"/>
                <div class="upload-icon">⚕️</div>
                <div class="upload-text">
                    <strong>Drop image here</strong><br>or click to browse
                </div>
            </div>
        </div>
        
        <div class="section">
            <button id="analyzeBtn" disabled onclick="analyzeImage()">
                ⚡ Analyse Image
            </button>
        </div>
        
        <div class="section" id="resultsSection" style="display:none;">
            <div class="section-title">Results</div>
            <div class="metric-card">
                <div class="metric-label">Nodule Detected</div>
                <div class="metric-value" id="nodulesYN">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Coverage %</div>
                <div class="metric-value" id="areaPct">—</div>
            </div>
            <div class="verdict" id="verdict">
                <div class="verdict-icon" id="verdictIcon"></div>
                <div class="verdict-title" id="verdictTitle"></div>
                <div class="verdict-sub" id="verdictSub"></div>
            </div>
        </div>
    </div>
    
    <!-- RIGHT PANEL -->
    <div class="right-panel">
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('results')">Results</button>
            <button class="tab-btn" onclick="switchTab('methods')">Methods</button>
            <button class="tab-btn" onclick="switchTab('steps')">Steps</button>
        </div>
        
        <div class="canvas-area">
            <!-- Results Tab -->
            <div class="tab-content active" id="tab-results">
                <div class="empty-state" id="emptyState">
                    <div class="empty-icon">🔬</div>
                    <div class="empty-title">Ready to Analyse</div>
                    <p style="font-size:12px;color:var(--muted);">Upload a thyroid ultrasound image to begin</p>
                </div>
                <div id="resultsContent" style="display:none;">
                    <div class="image-grid" id="resultsGrid"></div>
                </div>
            </div>
            
            <!-- Methods Tab -->
            <div class="tab-content" id="tab-methods">
                <div class="empty-state" id="emptyMethods">
                    <div class="empty-icon">🎯</div>
                    <div class="empty-title">No Analysis Yet</div>
                </div>
                <div id="methodsContent" style="display:none;">
                    <div class="image-grid" id="methodsGrid"></div>
                </div>
            </div>
            
            <!-- Steps Tab -->
            <div class="tab-content" id="tab-steps">
                <div class="empty-state" id="emptySteps">
                    <div class="empty-icon">✨</div>
                    <div class="empty-title">No Enhancement Yet</div>
                </div>
                <div id="stepsContent" style="display:none;">
                    <div class="image-grid" id="stepsGrid"></div>
                </div>
            </div>
        </div>
    </div>
</main>

<script>
let uploadedFile = null;

// Upload handling
document.getElementById('uploadZone').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', (e) => {
    uploadedFile = e.target.files[0];
    if (uploadedFile) {
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('uploadZone').querySelector('.upload-text').innerHTML =
            `<strong>${uploadedFile.name}</strong><br><span style="font-size:10px;color:var(--accent)">✓ Ready</span>`;
    }
});

function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    event.target.classList.add('active');
}

function analyzeImage() {
    if (!uploadedFile) return;
    
    const formData = new FormData();
    formData.append('image', uploadedFile);
    
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('analyzeBtn').textContent = '⏳ Processing...';
    
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        displayResults(data);
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').textContent = '⚡ Analyse Image';
    })
    .catch(err => {
        console.error(err);
        alert('Error: ' + err);
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').textContent = '⚡ Analyse Image';
    });
}

function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('resultsContent').style.display = 'block';
    document.getElementById('emptyMethods').style.display = 'none';
    document.getElementById('methodsContent').style.display = 'block';
    document.getElementById('emptySteps').style.display = 'none';
    document.getElementById('stepsContent').style.display = 'block';
    
    // Update metrics
    document.getElementById('nodulesYN').textContent = data.nodule_detected ? '✅ YES' : '❌ NO';
    document.getElementById('areaPct').textContent = data.nodule_area_pct.toFixed(1) + '%';
    
    // Verdict
    const v = document.getElementById('verdict');
    v.className = 'verdict ' + (data.nodule_detected ? 'nodule' : 'clear');
    document.getElementById('verdictIcon').textContent = data.nodule_detected ? '⚠️' : '✅';
    document.getElementById('verdictTitle').textContent = data.nodule_detected ? 'NODULE DETECTED' : 'NO NODULE';
    document.getElementById('verdictSub').textContent = data.nodule_detected 
        ? `Coverage: ${data.nodule_area_pct.toFixed(1)}%`
        : 'Normal appearance';
    
    // Results images
    const grid = document.getElementById('resultsGrid');
    grid.innerHTML = '';
    const images = [
        {name: 'Original', src: data.original},
        {name: 'Enhanced', src: data.enhanced},
        {name: 'Prediction', src: data.overlay},
        {name: 'Saliency', src: data.saliency},
    ];
    images.forEach(img => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.innerHTML = `<div class="image-card-title">${img.name}</div><img src="data:image/png;base64,${img.src}"/>`;
        grid.appendChild(card);
    });
    
    // Methods images
    const methodsGrid = document.getElementById('methodsGrid');
    methodsGrid.innerHTML = '';
    Object.entries(data.methods).forEach(([name, src]) => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.innerHTML = `<div class="image-card-title">${name}</div><img src="data:image/png;base64,${src}"/>`;
        methodsGrid.appendChild(card);
    });
    
    // Steps images
    const stepsGrid = document.getElementById('stepsGrid');
    stepsGrid.innerHTML = '';
    if (data.steps) {
        Object.entries(data.steps).forEach(([name, src]) => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `<div class="image-card-title">${name}</div><img src="data:image/png;base64,${src}"/>`;
            stepsGrid.appendChild(card);
        });
    }
}
</script>

</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    bgr_img  = file_to_bgr(img_file)

    # Run inference
    result = predict_single(bgr_img)

    # Prepare response
    response = {
    "nodule_detected": bool(result["nodule_detected"]),
    "nodule_area_pct": float(result["nodule_area_pct"]),
    "original": ndarray_to_b64(result["original"]),
    "enhanced": ndarray_to_b64(result["enhanced"]),
    "overlay": ndarray_to_b64(result["overlay"], is_rgb=True),
    "saliency": ndarray_to_b64((result["saliency"]*255).astype(np.uint8)),
    "methods": {
        name: ndarray_to_b64(mask)
        for name, mask in result["method_masks"].items()
    },
    "steps": {  # ← Add this block
        name: ndarray_to_b64((im if isinstance(im, np.ndarray) and im.dtype == np.uint8 
                              else (im*255).astype(np.uint8)))
        for name, im in result["steps"]
    }
}

    return jsonify(response)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🔬 ThyroSegment - Thyroid Nodule Detection GUI")
    print("  Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
