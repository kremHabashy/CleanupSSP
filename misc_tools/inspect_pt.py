#!/usr/bin/env python3
import sys, torch

for path in sys.argv[1:]:
    print(f"\n== {path}")
    obj = torch.load(path, map_location="cpu")
    print("type:", type(obj))

    if isinstance(obj, torch.nn.Module):
        print("module:", obj.__class__.__name__)
        total = sum(p.numel() for p in obj.parameters())
        print(f"params: {total:,}")
        for n,p in list(obj.named_parameters())[:5]:
            print(f"  {n}: {tuple(p.shape)}")

    elif isinstance(obj, dict):
        print("dict keys:", list(obj.keys())[:20])
        if "state_dict" in obj:
            sd = obj["state_dict"]
            print(f"state_dict has {len(sd)} tensors")
            for n,t in list(sd.items())[:5]:
                print(f"  {n}: {tuple(t.shape)}")
    else:
        print("unrecognized content")
