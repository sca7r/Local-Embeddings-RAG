"""
export_reranker.py
------------------
Run this ONCE before starting the server to export and quantize the
CrossEncoder to ONNX INT8 format.

Usage:
    python export_reranker.py

Output:  ./reranker_onnx/   (set RERANKER_ONNX_DIR env var to change the path)

Requirements:
    pip install optimum[onnxruntime]
"""

import os
import shutil
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

MODEL_ID  = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SAVE_DIR  = os.getenv("RERANKER_ONNX_DIR", "./reranker_onnx")

print(f"Exporting {MODEL_ID} → {SAVE_DIR} ...")

# Fix: wipe any stale files from previous partial runs.
# Multiple .onnx files in the same dir confuse ORTQuantizer.
if os.path.isdir(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
    print(f"  ✓ Cleared stale export at {SAVE_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Save the tokenizer explicitly — this is what the app loads at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(SAVE_DIR)
print("  ✓ Tokenizer saved")

# Step 2: Export model weights to ONNX (fp32)
ort_model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
ort_model.save_pretrained(SAVE_DIR)
print("  ✓ FP32 ONNX export complete")

# Step 3: INT8 dynamic quantization
# Fix: specify file_name explicitly so ORTQuantizer doesn't get confused
#      if there are multiple .onnx files present.
quantizer = ORTQuantizer.from_pretrained(SAVE_DIR, file_name="model.onnx")
qconfig   = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
quantizer.quantize(save_dir=SAVE_DIR, quantization_config=qconfig)
print("  ✓ INT8 quantization complete")

# Step 4: Remove the intermediate FP32 model — app only needs the quantized one.
# The quantized model is saved as model_quantized.onnx by optimum.
fp32_path = os.path.join(SAVE_DIR, "model.onnx")
if os.path.exists(fp32_path):
    os.remove(fp32_path)
    print("  ✓ Removed intermediate FP32 model.onnx (keeping model_quantized.onnx only)")

# Verify
files = os.listdir(SAVE_DIR)
print(f"\nFiles in {SAVE_DIR}:")
for f in sorted(files):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    print(f"  {f:40s}  {size // 1024:>6} KB")

required = ["tokenizer_config.json", "model_quantized.onnx"]
missing  = [f for f in required if not os.path.exists(os.path.join(SAVE_DIR, f))]
if missing:
    print(f"\n⚠ Missing expected files: {missing} — something went wrong.")
else:
    print("\n✓ Export complete. Start the server with:")
    print("  uvicorn src.app:app --host 0.0.0.0 --port 8000")