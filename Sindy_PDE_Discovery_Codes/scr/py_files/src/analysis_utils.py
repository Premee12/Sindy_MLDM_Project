import numpy as np
import matplotlib.pyplot as plt
import os

def spacetime(u, x, t, out_path=None, title="Space-time"):
    plt.figure(figsize=(10,3))
    plt.imshow(u.T, cmap="RdBu", aspect="auto", origin="lower",
               extent=(t[0], t[-1], x[0], x[-1]))
    plt.colorbar(label="u")
    plt.xlabel("time"); plt.ylabel("space"); plt.title(title)
    plt.tight_layout()
    if out_path: os.makedirs(os.path.dirname(out_path), exist_ok=True); plt.savefig(out_path, dpi=180)
    plt.close()

def energy_curve(u, t, out_path=None, title="Average u^2"):
    E = (u**2).mean(axis=1)
    plt.figure()
    plt.plot(t, E)
    plt.xlabel("time"); plt.ylabel("average u^2"); plt.title(title)
    plt.tight_layout()
    if out_path: os.makedirs(os.path.dirname(out_path), exist_ok=True); plt.savefig(out_path, dpi=180)
    plt.close()
    return float(E.mean())

def change_curve(u, t, out_path=None, title="Avg |Δu| per step"):
    ch = np.mean(np.abs(np.diff(u, axis=0)), axis=1)
    plt.figure()
    plt.plot(t[1:], ch)
    plt.xlabel("time"); plt.ylabel("avg |Δu| per step"); plt.title(title)
    plt.tight_layout()
    if out_path: os.makedirs(os.path.dirname(out_path), exist_ok=True); plt.savefig(out_path, dpi=180)
    plt.close()
    return float(ch.mean())

def spectrum_plot(u, x, out_path=None, title="Time-avg spatial spectrum"):
    Uhat = np.fft.rfft(u, axis=1)
    E_k = (np.abs(Uhat)**2).mean(axis=0)
    dx = x[1]-x[0]
    k = 2*np.pi*np.fft.rfftfreq(x.size, d=dx)
    plt.figure()
    plt.loglog(k[1:], E_k[1:])
    plt.xlabel("|k|"); plt.ylabel("<|Û|^2>"); plt.title(title)
    plt.tight_layout()
    if out_path: os.makedirs(os.path.dirname(out_path), exist_ok=True); plt.savefig(out_path, dpi=180)
    plt.close()

def analyze_to_pngs(u, x, t, tag, figs_dir="figs"):
    out1 = os.path.join(figs_dir, f"{tag}_spacetime.png")
    out2 = os.path.join(figs_dir, f"{tag}_energy.png")
    out3 = os.path.join(figs_dir, f"{tag}_change.png")
    out4 = os.path.join(figs_dir, f"{tag}_spectrum.png")
    spacetime(u, x, t, out1, title=f"{tag}: space-time")
    e_mean = energy_curve(u, t, out2, title=f"{tag}: energy")
    c_mean = change_curve(u, t, out3, title=f"{tag}: change per frame")
    spectrum_plot(u, x, out4, title=f"{tag}: spectrum")
    return {"energy_mean": e_mean, "change_mean": c_mean,
            "spacetime_png": out1, "energy_png": out2,
            "change_png": out3, "spectrum_png": out4}
