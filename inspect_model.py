# inspect_model.py
import torch, sys
from pathlib import Path

MODEL = Path("models/best_model.pt")
print("Model path:", MODEL, "exists:", MODEL.exists())
if not MODEL.exists():
    sys.exit(1)

# Try ultralytics load test (safe)
print("\\n1) Try ultralytics.YOLO (if installed):")
try:
    from ultralytics import YOLO
    try:
        y = YOLO(str(MODEL))
        print("-> ultralytics.YOLO loaded successfully. Model type:", type(y))
    except Exception as e:
        print("-> ultralytics import OK but failed to load model:", e)
except Exception as e:
    print("-> ultralytics not available or import failed:", e)

# Try torch.load
print("\\n2) Try torch.load:")
try:
    obj = torch.load(str(MODEL), map_location="cpu")
    print("-> torch.load succeeded. Type:", type(obj))
    if isinstance(obj, dict):
        print("-> Keys:", list(obj.keys())[:20])
        # heuristic: state dict if keys contain 'state_dict' or start with 'module.' or typical layer names
        keys = list(obj.keys())
        if 'state_dict' in keys:
            print("-> Looks like a checkpoint dict (has 'state_dict').")
        else:
            # check if it *is* a state_dict (string keys like 'conv1.weight')
            sample_keys = keys[:5]
            if all(isinstance(k, str) and ('.' in k or 'conv' in k or 'fc' in k) for k in sample_keys):
                print("-> Looks like a state_dict (keys look like layer names).")
    else:
        # not a dict -> could be a saved nn.Module object
        print("-> Not a dict. It might be a full torch.nn.Module object (saved with torch.save(model)).")
except Exception as e:
    print("-> torch.load failed:", e)