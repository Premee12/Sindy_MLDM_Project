import h5py
import numpy as np
import os

def save_h5(path, u, x, t, attrs=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("u", data=np.asarray(u))
        f.create_dataset("x", data=np.asarray(x))
        f.create_dataset("t", data=np.asarray(t))
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v

def load_h5(path):
    with h5py.File(path, "r") as f:
        u = f["u"][:]
        x = f["x"][:]
        t = f["t"][:]
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    return u, x, t, attrs

def make_tag(L, N, dt, kind, noise_std, seed, stride=1):
    return (f"L{int(L)}_N{int(N)}_dt{dt:.3f}_s{stride}_"
            f"{kind}_noise{noise_std:.3f}_seed{seed}")

