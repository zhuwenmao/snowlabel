"""Convert local sam3.1_multiplex.pt -> HF Sam3Model format.

Adapts the official transformers convert_sam3_to_hf logic for the *multiplex*
checkpoint: keeps only `detector.*` weights (drops `tracker.*` video keys and
the multiplex-only interactive/propagation conv stacks), applies the official
key mapping + QKV split, loads into Sam3Model (strict=False), and saves an HF
snapshot + processor to weights/sam3_hf/.
"""
import gc
import os
import sys

import torch
from transformers import AutoConfig, Sam3Model, Sam3Processor

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from convert_sam3_to_hf import convert_old_keys_to_new_keys, split_qkv  # noqa: E402

W = os.path.join(SCRIPTS, "..", "weights", "facebooksam3.1")
CKPT = os.path.join(W, "sam3.1_multiplex.pt")
OUT = os.path.join(SCRIPTS, "..", "weights", "sam3_hf")

print("Loading multiplex checkpoint...")
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
det = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
del sd
gc.collect()
print(f"  detector tensors: {len(det)}")

print("Mapping keys...")
keymap = convert_old_keys_to_new_keys(list(det.keys()))
new_sd = {}
for ok, v in det.items():
    nk = keymap.get(ok, ok)
    if nk == "vision_encoder.backbone.embeddings.position_embeddings":
        new_sd[nk] = v[:, 1:, :]  # strip cls token position
    else:
        new_sd[nk] = v
del det
gc.collect()

print("Splitting QKV...")
new_sd = split_qkv(new_sd)
if "text_encoder.text_projection.weight" in new_sd:
    new_sd["text_encoder.text_projection.weight"] = new_sd["text_encoder.text_projection.weight"].T

print("Building Sam3Model and loading weights...")
config = AutoConfig.from_pretrained(W)  # sam3_video config; Sam3Model extracts detector
model = Sam3Model(config)
missing, unexpected = model.load_state_dict(new_sd, strict=False)
print(f"  missing={len(missing)} unexpected={len(unexpected)}")
for k in missing:
    print("   MISSING:", k)
del new_sd
gc.collect()

print(f"Saving HF model to {OUT} ...")
model.save_pretrained(OUT)
del model
gc.collect()

print("Saving processor (from local tokenizer/processor files)...")
proc = Sam3Processor.from_pretrained(W)
proc.save_pretrained(OUT)

print("DONE. Converted SAM3 (image detector) saved to:", OUT)
