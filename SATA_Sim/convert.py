import torch
import sys
import os

def convert_pth_tar(src_path, dst_path=None):
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + '_state.pth'
    ckpt = torch.load(src_path, map_location='cpu')
    # Try common keys
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        else:
            # maybe it's already a state_dict
            # verify values look like tensors
            first = next(iter(ckpt.values()))
            if hasattr(first, 'shape') or hasattr(first, 'dtype'):
                sd = ckpt
            else:
                raise RuntimeError("Unrecognized .pth.tar contents. Keys: " + ", ".join(ckpt.keys()))
    else:
        raise RuntimeError("Unsupported checkpoint format")

    torch.save(sd, dst_path)
    print("Wrote state_dict to", dst_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_pth_tar.py path/to/checkpoint.pth.tar [out_path.pth]")
        sys.exit(1)
    convert_pth_tar(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
