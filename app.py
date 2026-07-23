"""
SAM Demo — single-file Flask app.

Two endpoints, mirroring SAM's encode-once / prompt-many design:
  POST /embed    -> runs the heavy image encoder once, caches the embedding
  POST /segment  -> runs the light prompt encoder + mask decoder per click

Run:  python app.py
Then: http://localhost:5000
"""

import base64
import io
import os
import uuid
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, Response
from PIL import Image

# --- your modules ---
from model.image_encoder import ImageEncoderViT
from model.prompt_encoder import PromptEncoder
from model.transformer import TwoWayTransformer
from model.mask_decoder import MaskDecoder
from model.sam import SAM

# ---------------------------------------------------------------- config
CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LONG_SIDE = 1024
MAX_CACHE = 8  # embeddings held in memory

app = Flask(__name__)

# embedding cache: id -> (embedding, original_size, resized_size, scale)
CACHE: Dict[str, Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], float]] = {}
CACHE_ORDER = []


# ---------------------------------------------------------------- model
def build_sam() -> SAM:
    image_encoder = ImageEncoderViT(
        img_size=1024, patch_size=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        out_chans=256, qkv_bias=True,
        use_abs_pos=True, use_rel_pos=True,
        window_size=14, global_attn_indexes=(2, 5, 8, 11),
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048
        ),
    )
    model = SAM(image_encoder, prompt_encoder, mask_decoder)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


print(f"Loading SAM on {DEVICE} ...")
SAM = build_sam()
print("Ready.")


# ---------------------------------------------------------------- helpers
def preprocess_image(pil: Image.Image):
    """Resize so the long side is 1024. Returns tensor + geometry."""
    img = np.array(pil.convert("RGB"))
    oh, ow = img.shape[:2]
    scale = LONG_SIDE / max(oh, ow)
    nh, nw = int(oh * scale + 0.5), int(ow * scale + 0.5)

    t = torch.as_tensor(img).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, (nh, nw), mode="bilinear", align_corners=False)[0]
    return t.to(DEVICE), (oh, ow), (nh, nw), scale


def cache_put(key, value):
    CACHE[key] = value
    CACHE_ORDER.append(key)
    while len(CACHE_ORDER) > MAX_CACHE:
        CACHE.pop(CACHE_ORDER.pop(0), None)


def mask_to_png_b64(mask: np.ndarray, rgba=(255, 106, 61, 150)) -> str:
    """Boolean mask -> transparent PNG data URI."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[mask] = rgba
    buf = io.BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- routes
@app.post("/embed")
def embed():
    """Heavy step. Runs the ViT image encoder once and caches the result."""
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded."), 400

    pil = Image.open(file.stream)
    img_t, orig_size, resized_size, scale = preprocess_image(pil)

    with torch.no_grad():
        batched = SAM.preprocess(img_t)[None]          # (1,3,1024,1024)
        embedding = SAM.image_encoder(batched)          # (1,256,64,64)

    key = uuid.uuid4().hex
    cache_put(key, (embedding, orig_size, resized_size, scale))

    return jsonify(
        embedding_id=key,
        original_size=orig_size,
        resized_size=resized_size,
    )


@app.post("/segment")
def segment():
    """Light step. Prompt encoder + mask decoder against a cached embedding."""
    data = request.get_json(force=True)
    key = data.get("embedding_id")
    entry = CACHE.get(key)
    if entry is None:
        return jsonify(error="Embedding expired. Upload the image again."), 404

    embedding, orig_size, resized_size, scale = entry
    multimask = bool(data.get("multimask", True))

    points = data.get("points") or []       # [{x,y,label}] in ORIGINAL coords
    box = data.get("box")                    # [x0,y0,x1,y1] in ORIGINAL coords

    coords = labels = None
    if points:
        coords = torch.tensor(
            [[[p["x"] * scale, p["y"] * scale] for p in points]],
            dtype=torch.float, device=DEVICE,
        )
        labels = torch.tensor(
            [[int(p.get("label", 1)) for p in points]], device=DEVICE
        )

    boxes = None
    if box:
        boxes = torch.tensor([box], dtype=torch.float, device=DEVICE) * scale

    with torch.no_grad():
        sparse, dense = SAM.prompt_encoder(
            points=(coords, labels) if coords is not None else None,
            boxes=boxes,
            masks=None,
        )
        low_res, iou_pred = SAM.mask_decoder(
            image_embeddings=embedding,
            image_pe=SAM.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
        )
        masks = SAM.postprocess_masks(low_res, resized_size, orig_size)
        masks = (masks > SAM.mask_threshold)[0].cpu().numpy()

    scores = iou_pred[0].tolist()
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    return jsonify(
        masks=[
            {
                "png": mask_to_png_b64(masks[i]),
                "score": round(scores[i], 3),
                "area": int(masks[i].sum()),
            }
            for i in order
        ]
    )


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


# ---------------------------------------------------------------- frontend
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segment Anything — local</title>
<style>
  :root{
    --ink:#12100e; --paper:#e8e4dc; --line:#c3bcae;
    --hot:#ff5c26; --cool:#1f6feb; --muted:#6c6558;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--paper); color:var(--ink);
    font:15px/1.5 ui-monospace,"SF Mono",Menlo,monospace;
    -webkit-font-smoothing:antialiased;
  }
  header{
    padding:22px 26px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
  }
  h1{
    margin:0; font-size:19px; font-weight:700; letter-spacing:-.02em;
  }
  .sub{color:var(--muted); font-size:13px}
  main{
    display:grid; grid-template-columns: 1fr 300px;
    gap:0; min-height:calc(100vh - 66px);
  }
  @media (max-width:860px){ main{grid-template-columns:1fr} }
  #stage{
    padding:26px; display:flex; align-items:flex-start; justify-content:center;
  }
  #frame{position:relative; line-height:0; max-width:100%}
  #frame img{max-width:100%; height:auto; display:block; cursor:crosshair}
  #overlay{position:absolute; inset:0; pointer-events:none}
  #overlay img{position:absolute; inset:0; width:100%; height:100%}
  .dot{
    position:absolute; width:13px; height:13px; margin:-7px 0 0 -7px;
    border-radius:50%; border:2px solid #fff;
    box-shadow:0 0 0 1px rgba(0,0,0,.4);
  }
  .dot.pos{background:var(--hot)} .dot.neg{background:var(--cool)}
  aside{
    border-left:1px solid var(--line); padding:26px 22px;
    display:flex; flex-direction:column; gap:22px;
  }
  @media (max-width:860px){ aside{border-left:0; border-top:1px solid var(--line)} }
  .lbl{
    font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--muted); margin-bottom:9px;
  }
  button, .filebtn{
    font:inherit; font-size:13px; padding:9px 13px; cursor:pointer;
    background:transparent; border:1px solid var(--ink); border-radius:2px;
    transition:background .12s, color .12s;
  }
  button:hover, .filebtn:hover{background:var(--ink); color:var(--paper)}
  button:disabled{opacity:.35; cursor:default}
  button:disabled:hover{background:transparent; color:var(--ink)}
  .row{display:flex; gap:8px; flex-wrap:wrap}
  .seg{display:flex; border:1px solid var(--ink); border-radius:2px; overflow:hidden}
  .seg button{border:0; border-radius:0; flex:1}
  .seg button[aria-pressed="true"]{background:var(--ink); color:var(--paper)}
  input[type=file]{display:none}
  .card{
    border:1px solid var(--line); padding:11px 12px; cursor:pointer;
    display:flex; justify-content:space-between; align-items:center; gap:10px;
    background:#efece6;
  }
  .card[aria-selected="true"]{border-color:var(--ink); background:#fff}
  .card .n{font-weight:700}
  .card .m{color:var(--muted); font-size:12px}
  .status{font-size:12px; color:var(--muted); min-height:18px}
  .hint{font-size:12px; color:var(--muted); line-height:1.65}
  kbd{
    border:1px solid var(--line); border-bottom-width:2px; border-radius:3px;
    padding:1px 5px; font-size:11px; background:#efece6;
  }
</style>
</head>
<body>

<header>
  <h1>Segment Anything</h1>
  <span class="sub">ViT-B · built from the paper </span>
</header>

<main>
  <section id="stage">
    <div id="frame">
      <img id="photo" alt="" hidden>
      <div id="overlay"></div>
      <p id="empty" class="hint">Load a photo to begin.</p>
    </div>
  </section>

  <aside>
    <div>
      <div class="lbl">Image</div>
      <label class="filebtn">Choose photo
        <input type="file" id="file" accept="image/*">
      </label>
      <p class="status" id="status"></p>
    </div>

    <div>
      <div class="lbl">Click mode</div>
      <div class="seg">
        <button id="mPos" aria-pressed="true">Include</button>
        <button id="mNeg" aria-pressed="false">Exclude</button>
      </div>
      <p class="hint" style="margin:9px 0 0">
        Include adds to the object. Exclude carves away.
      </p>
    </div>

    <div>
      <div class="lbl">Results</div>
      <div id="cards"></div>
    </div>

    <div class="row">
      <button id="undo">Undo click</button>
      <button id="reset">Clear</button>
    </div>

    <p class="hint">
      One <kbd>/embed</kbd> call per photo (the slow part).
      Every click is a <kbd>/segment</kbd> call against the cached embedding.
    </p>
  </aside>
</main>

<script>
const $ = s => document.querySelector(s);
const photo = $('#photo'), overlay = $('#overlay'), cards = $('#cards');
let embId = null, points = [], label = 1, masks = [], picked = 0, natW = 0, natH = 0;

function setStatus(t){ $('#status').textContent = t || ''; }

$('#mPos').onclick = () => setMode(1);
$('#mNeg').onclick = () => setMode(0);
function setMode(v){
  label = v;
  $('#mPos').setAttribute('aria-pressed', v === 1);
  $('#mNeg').setAttribute('aria-pressed', v === 0);
}

$('#file').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  reset();
  photo.src = URL.createObjectURL(f);
  photo.hidden = false; $('#empty').hidden = true;
  await new Promise(r => photo.onload = r);
  natW = photo.naturalWidth; natH = photo.naturalHeight;

  setStatus('Encoding image…');
  const fd = new FormData(); fd.append('image', f);
  const res = await fetch('/embed', { method:'POST', body: fd }).then(r => r.json());
  if (res.error) { setStatus(res.error); return; }
  embId = res.embedding_id;
  setStatus(`Encoded ${res.original_size[1]}×${res.original_size[0]}. Click the image.`);
};

photo.onclick = ev => {
  if (!embId) return;
  const r = photo.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width  * natW;
  const y = (ev.clientY - r.top)  / r.height * natH;
  points.push({ x, y, label });
  drawDots();
  run();
};

async function run(){
  if (!points.length) { masks = []; render(); return; }
  setStatus('Segmenting…');
  const res = await fetch('/segment', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ embedding_id: embId, points, multimask: points.length === 1 })
  }).then(r => r.json());
  if (res.error) { setStatus(res.error); embId = null; return; }
  masks = res.masks; picked = 0;
  setStatus(`${masks.length} candidate${masks.length>1?'s':''}.`);
  render();
}

function render(){
  [...overlay.querySelectorAll('img')].forEach(n => n.remove());
  if (masks[picked]) {
    const im = new Image(); im.src = masks[picked].png; overlay.prepend(im);
  }
  cards.innerHTML = '';
  masks.forEach((m, i) => {
    const d = document.createElement('div');
    d.className = 'card';
    d.setAttribute('aria-selected', i === picked);
    d.innerHTML = `<span class="n">${(m.score*100).toFixed(1)}%</span>
                   <span class="m">${m.area.toLocaleString()} px</span>`;
    d.onclick = () => { picked = i; render(); };
    cards.appendChild(d);
  });
}

function drawDots(){
  [...overlay.querySelectorAll('.dot')].forEach(n => n.remove());
  points.forEach(p => {
    const d = document.createElement('div');
    d.className = 'dot ' + (p.label ? 'pos' : 'neg');
    d.style.left = (p.x / natW * 100) + '%';
    d.style.top  = (p.y / natH * 100) + '%';
    overlay.appendChild(d);
  });
}

$('#undo').onclick = () => { points.pop(); drawDots(); run(); };
$('#reset').onclick = () => { points = []; drawDots(); masks = []; render(); setStatus(embId ? 'Cleared.' : ''); };
function reset(){ embId = null; points = []; masks = []; overlay.innerHTML = ''; cards.innerHTML = ''; }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))