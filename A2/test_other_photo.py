# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow", "scikit-learn"]
# ///
"""
test_other_photo.py -- does my Imputer also work on a DIFFERENT piece of Austria?

It downloads an aerial photograph of Krems an der Donau (the same open service and the same
zoom level as the Wachau photograph, 0.79 m per pixel), trains my Imputer on the Wachau
patches exactly as A2.py does, and measures the error on random patches of the new photo.

Put it next to A2.py and get_wachau_photo.py and run:

    uv run test_other_photo.py

This is only my own check. It is NOT handed in. A2.py is not changed by it.
Datenquelle: basemap.at (CC BY 4.0)
"""
import io
import sys

import numpy as np
from PIL import Image

import A2                        # my Imputer, labels_of, error_of, build_patches
import get_wachau_photo as g     # the tile download helpers of the course

Image.MAX_IMAGE_PIXELS = None
OTHER = A2.HERE / "krems_ortho.jpg"

# Krems an der Donau: old town, the Danube, Stein and Mautern. East of the Wachau box.
BOX = dict(west=15.570, south=48.395, east=15.630, north=48.420)


def download():
    """Download the Krems photograph from basemap.at once, tile by tile."""
    if OTHER.exists():
        print(f"{OTHER.name} is already here. Not downloading again.")
        return
    g.BOX = BOX                  # geometry() reads the box from the module
    meta = g.geometry()
    nx, ny = meta["nx"], meta["ny"]
    print(f"Downloading {nx * ny} tiles of Krems ({meta['width']} x {meta['height']} px) ...")
    img = Image.new("RGB", (meta["width"], meta["height"]))
    for i in range(nx):
        for j in range(ny):
            url = g.ORTHO_URL.format(z=g.ZOOM, x=meta["x0"] + i, y=meta["y0"] + j)
            img.paste(Image.open(io.BytesIO(g.fetch(url))), (i * g.TILE, j * g.TILE))
        print(f"  column {i + 1}/{nx}", end="\r", flush=True)
    print()
    img.save(OTHER, quality=88)
    print(f"  wrote {OTHER.name}")


def random_patches(photo, h, w, m, seed=1):
    """m patches of h x w at random positions of `photo`: shape (m, h, w, 3)."""
    H, W, _ = photo.shape
    ch, cw = h // 2, w // 2
    rng = np.random.default_rng(seed)
    rows = rng.integers(ch, H - ch, m)
    cols = rng.integers(cw, W - cw, m)
    dr, dc = np.meshgrid(np.arange(-ch, ch + 1), np.arange(-cw, cw + 1), indexing="ij")
    return photo[rows[:, None, None] + dr, cols[:, None, None] + dc].astype(np.float32)


if __name__ == "__main__":
    download()
    other = np.asarray(Image.open(OTHER).convert("RGB"))

    # train exactly as A2.py does: on the western Wachau patches
    _, train_patches, val_patches = A2.build_patches()
    imputer = A2.Imputer().train(train_patches)
    baseline = train_patches.reshape(-1, 3).mean(axis=0)      # the mean Wachau colour

    print(f"\nTrained on Wachau (west). Error = MSE, lower is better.")
    print(f"Wachau east, as in A2.py:  my imputer E_v = {A2.error_of(imputer, val_patches):.1f}")
    print(f"\nKrems, 2,000 random patches per size:")
    print(f"{'patch size':<12}{'baseline':>10}{'my imputer':>12}")
    for h, w in [(15, 15), (7, 7), (5, 5), (3, 3), (9, 21), (21, 21)]:
        P = random_patches(other, h, w, 2000)
        base = float(np.mean((A2.labels_of(P) - baseline) ** 2))
        mine = A2.error_of(imputer, P)
        print(f"{f'{h} x {w}':<12}{base:>10.1f}{mine:>12.1f}")
    print("\nDatenquelle: basemap.at (CC BY 4.0)")