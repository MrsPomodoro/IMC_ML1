# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
# ///
"""
A1 -- Data points, features and labels from one photograph
Concepts: data point, features, label, loss; baseline; spatial train/test split
Starts: Session 1 (Wed 30.09) + Session 2 (Thu 01.10)   |   Due: Wed 07.10.2026, 23:59

WRITE YOUR PROGRAM IN THIS FILE. You get the full task in A1.pdf. Attach this
file to the A1 Aufgabe in Teams as you go, together with A1_whytrail.txt
(two files, no ZIP, Abgeben by the deadline).

GETTING PYTHON TO RUN -- once, on your own laptop, about 15 minutes
  1. Install uv. It installs Python for you and runs your files.
       Windows, in PowerShell:
         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
       macOS / Linux, in Terminal:
         curl -LsSf https://astral.sh/uv/install.sh | sh
     Close and reopen the terminal, then check:      uv --version
     "command not found" / "not recognized"? That is normal and expected: paste
     the whole error into the LLM with "I am on Windows/Mac, I just installed uv
     and I get this, how do I fix it?", try its suggestion once, then ask me.
  2. Install VS Code (code.visualstudio.com) and its Python extension
     (left sidebar -> Extensions -> search "Python" -> Install).
  3. Run this file from its folder:                  uv run A1.py
     Need a library? One command adds it to the header at the top of this file:
                                  uv add --script A1.py numpy pillow scikit-learn
     The first run takes a few seconds: uv downloads Python and the libraries.
  Stuck for more than 15 minutes? Do not keep fighting it. Open
  colab.research.google.com in the browser and work there for now; we fix your
  setup together in the session. Nobody is held up by an install in this course.
  .py files, not notebooks: a notebook hides the order its cells ran in, and a
  hidden run order is exactly where data leakage hides.

DATA
Both data files are downloaded once by the attached script (2-4 min, open data):
  uv run get_wachau_photo.py
  wachau_ortho.jpg   the photograph from basemap.at (9216x4608, 0.79 m per pixel,
                     7.3 x 3.7 km: Weissenkirchen, Rossatz, Duernstein, Loiben)
  wachau_labels.png  the INVEKOS parcels on the same grid: 0 = no parcel,
                     1 = vineyard, 2 = other crop
It is a big picture: 41,472 patches of 32 px. Open it with
  Image.MAX_IMAGE_PIXELS = None   (Pillow refuses large images by default)
and compute a few features per patch instead of keeping all 3072 of its raw numbers.

SCORING
  This file is the artifact: 25 of the 100 points = it runs (15) + a defensible
  result (10). The other 75 are in A1_whytrail.txt (why-trail 45, hand
  computation 30). Using an LLM to write this file is expected; what is graded
  is the checks you ran, written down in the why-trail.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None       # the photograph is 42 megapixels; Pillow refuses it by default
HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / name
    if not path.exists():
        sys.exit(f"{name} not found next to A1.py: run  uv run get_wachau_photo.py  once.")
    return Image.open(path)


# The photograph as a 3-d array on the pixel grid: photo[row, column, channel].
# Rows run north to south, columns west to east, channel = red, green, blue (0..255).
photo = np.asarray(load("wachau_ortho.jpg").convert("RGB"))     # shape (4608, 9216, 3), uint8

# The parcel labels on the SAME pixel grid: labels[row, column] belongs to photo[row, column]
# and is 0 (no parcel), 1 (vineyard) or 2 (other crop).
labels = np.asarray(load("wachau_labels.png"))                   # shape (4608, 9216), uint8

assert photo.shape[:2] == labels.shape, "photo and labels must share one pixel grid"
H, W = labels.shape
print(f"photo  {photo.shape} {photo.dtype}: {H} rows x {W} columns x 3 channels, 0.79 m per pixel")
print(f"labels {labels.shape} {labels.dtype}: values {np.unique(labels).tolist()}")

# A histogram of the label values: how many pixels carry 0, 1 and 2, and their share.
counts = np.bincount(labels.ravel(), minlength=3)          # counts[v] = number of pixels with label v
print("histogram of the labels:")
for value, meaning in ((0, "no parcel"), (1, "vineyard"), (2, "other crop")):
    share = counts[value] / counts.sum()
    print(f"   {value} = {meaning:<11}{counts[value]:>12,} px  {share * 100:5.1f} %  {'#' * round(share * 50)}")

# One pixel, to see the grid: row 2400, column 8600 of both arrays (on the Loibenberg).
r, c = 2400, 8600
print(f"pixel ({r}, {c}): RGB = {photo[r, c].tolist()}, label = {labels[r, c]}")

# Keep the arrays uint8: photo.astype(np.float32) alone needs 500 MB.
# A P x P patch grid is one reshape away, e.g. P = 32 (25 m on the ground):
#   P = 32; grid = photo[: H // P * P, : W // P * P].reshape(H // P, P, W // P, P, 3)
#   grid.mean(axis=(1, 3))  -> one mean colour per patch, shape (144, 288, 3)
# and the same reshape on labels gives you each patch's share of vineyard pixels.


# STEP 2 (A1.pdf): the three candidate data points; print m and d for each
#         (a) one pixel:    m = H * W, d = 3   ->  photo.reshape(-1, 3)

# Expected values, copied from A1.pdf for a check 
M_EXPECTED = 42_467_328
D_EXPECTED = 3

X = photo.reshape(-1, 3)  
m_a, d_a = X.shape
dm, dd = m_a - M_EXPECTED, d_a - D_EXPECTED
print(f"     ")
print(f"    (a) one pixel: ")
print(f"    expected:   m = {M_EXPECTED:,}   d = {D_EXPECTED}")
print(f"    difference: m {dm:+,}   d {dd:+}   -> {'OK' if dm == 0 and dd == 0 else 'NOT OK'}")
print(f"    m = {m_a:,}   d = {d_a:}   ")



#         (b) one patch:    P = 32; cut the grid, compute a few features per patch

P = 32                                   # patch size in pixels (25 m on the ground)
C = photo.shape[-1]                      # number of colour channels (3 for RGB)

n_rows = H // P                          # patches vertically
n_cols = W // P                          # patches horizontally

cropped = photo[:n_rows * P, :n_cols * P]            # shape (n_rows*P, n_cols*P, C)

# Step : split rows and columns into (patch index, pixel inside patch)
grid = cropped.reshape(n_rows, P, n_cols, P, C)      # (patch_row, y, patch_col, x, C)

# Step: put the two patch indices next to each other
grid = grid.transpose(0, 2, 1, 3, 4)                 # (patch_row, patch_col, y, x, C)

# Step: one row per patch, all its pixels listed together
patches = grid.reshape(n_rows * n_cols, P * P, C)    # (n_patches, pixels_per_patch, C)

# Step: features per patch = mean and std of each colour channel
means = patches.mean(axis=1)                         # (n_patches, C)
stds  = patches.std(axis=1)                          # (n_patches, C)

# Step: glue features side by side -> data matrix
X_b = np.hstack([means, stds])                       # (n_patches, 2*C)

m_b, d_b = X_b.shape
print(f"     ")
print(f"    (b) one patch: ")
print(f"    m = {m_b:,}   d = {d_b}")

#         (c) whole image:  m = 1

# STEP 3: the labels. A pixel's label is labels[row, column]. For every patch, the
#         share of its pixels with value 1; the patch is vineyard if the share exceeds
#         a threshold. Print the share of vineyard patches for > 25 %, > 50 %, > 75 %.

# STEP 4: the baseline. For pixels, and for patches at each threshold: print the
#         vineyard share and the accuracy of the constant majority prediction.

# STEPS 5 and 6 are written in A1_whytrail.txt, not here.
