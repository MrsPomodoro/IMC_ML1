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

print(f"     ")
print(f"    STEP 2 ")
print(f"    (a) one pixel: ")

# Expected values, copied from A1.pdf for a check 
M_EXPECTED = 42_467_328
D_EXPECTED = 3

X = photo.reshape(-1, 3)  
m_a, d_a = X.shape
dm, dd = m_a - M_EXPECTED, d_a - D_EXPECTED
print(f"    expected:   m = {M_EXPECTED:,}   d = {D_EXPECTED}")
print(f"    difference: m {dm:+,}   d {dd:+}   -> {'OK' if dm == 0 and dd == 0 else 'NOT OK'}")
print(f"    m = {m_a:,}   d = {d_a:}   ")

#         (b) one patch:    P = 32; cut the grid, compute a few features per patch
print(f"     ")
print(f"    (b) one patch: ")

P = 32                                   # patch size in pixels (25 m on the ground)
C = photo.shape[-1]                      # number of colour channels (3 for RGB)

n_rows = H // P                          # patches vertically
n_cols = W // P                          # patches horizontally
print(f"    patches vertically (n_rows) = {n_rows}  patches horizontally (n_cols) = {n_cols:}   ")

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
#X_b = patches.mean(axis=1)        #features per patch = mean of each colour channel      (n_patches, C)
X_b = np.hstack([means, stds])                       # (n_patches, 2*C)
m_b, d_b = X_b.shape
print(f"    m = {m_b:,}   d = {d_b}")


## CHECKS:
# check if I got the same amount of pixels as in (a) (patches x pixels per patch = m from (a))
print(f"     ")
print(f"    My checks task 2: ")
total = m_b * P * P
print(f"    patch check_1: {m_b:,} x {P*P:,} = {total:,}   "
      f"-> {'NO PIXELS GOT LOSS' if total == M_EXPECTED else 'I LOST SOME PIXELS'}")

# check if both H & W are divisible without a remainder
rest_H, rest_W = H % P, W % P
lost = H * W - (H - rest_H) * (W - rest_W)
status = "OK" if rest_H == 0 and rest_W == 0 else "NOT OK"
print(f"    patch check_2: H / P = {rest_H}   W / P = {rest_W}   lost pixels = {lost:,}   -> {status}")

#         (c) whole image:  m = 1
print(f"     ")
print(f"    (c) whole image: ")
C = photo.shape[-1]                      # number of colour channels (3 for RGB)

# Step 1: count all values in the image
n_values = H * W * C                     # every pixel, every channel

# Step 2: lay every value out in one long row -> one sample
row = photo.reshape(-1)                  # shape (H*W*C,)

# Step 3: make it a data matrix with exactly 1 row
X_c = row.reshape(1, n_values)           # shape (1, H*W*C)

m_c, d_c = X_c.shape                     # m = 1, d = H*W*C

print(f"    m = {m_c:,}   d = {d_c:,}")


# ---- checks task 2 by LLM ----
print(f"     ")
print(f"    ---- checks task 2 by LLM ---- ")
def ok(cond):
    return "OK" if cond else "NOT OK"

print(f"    check same size photo/labels: {photo.shape[:2]} vs {labels.shape} "
      f"-> {ok(photo.shape[:2] == labels.shape)}")
print(f"    check shapes: X {X.shape}   X_b {X_b.shape}   X_c {X_c.shape} -> "
      f"{ok(X.shape == (H*W, C) and X_b.shape == (n_rows*n_cols, 2*C) and X_c.shape == (1, H*W*C))}")
print(f"    check finite X_b -> {ok(np.isfinite(X_b).all())}")
print(f"    check stds >= 0 -> {ok((X_b[:, C:] >= 0).all())}")

# STEP 3: the labels. A pixel's label is labels[row, column]. For every patch, the
#         share of its pixels with value 1; the patch is vineyard if the share exceeds
#         a threshold. Print the share of vineyard patches for > 25 %, > 50 %, > 75 %.
print(f"     ")
print(f"    STEP 3 ")

#  for (a) option - data point is a pixel,  label: 1 = vineyard, 0 and 2 = not vineyard
y_pixel = (labels.reshape(-1) == 1)          # (m_a,) one True / False per pixel, same order as X
print(f"    data point: (a) pixel ")
print(f"    labels: m = {len(y_pixel):,}   vineyard pixels = {y_pixel.mean():.1%}")     # vineyard pixels / all pixels y_pixel.sum() / len(y_pixel)   

#  fro (b)  option - datapoint is patch, label per patch: vineyard if share of pixels with value 1 > threshold
print(f"     ")
print(f"    data point: (b) patch ")
# thresholds for the patch label
threshold_25 = 0.25
threshold_50 = 0.50
threshold_75 = 0.75

# Step: cut labels into the same patches as the photo (same crop, same order)
lab_cropped = labels[:n_rows * P, :n_cols * P]                        # (n_rows*P, n_cols*P)
lab_grid = lab_cropped.reshape(n_rows, P, n_cols, P).transpose(0, 2, 1, 3)
lab_patches = lab_grid.reshape(n_rows * n_cols, P * P)               # (n_patches, P*P)

# Step: share of vineyard pixels (value 1) in every patch; 0 and others count as not vineyard
vine_share = (lab_patches == 1).mean(axis=1)                          # (n_patches,)

# Step: label per threshold and print the share of vineyard patches
for given_treshold in [threshold_25, threshold_50, threshold_75]:
    y_b = (vine_share > given_treshold)                                            # True = vineyard
    print(f"patch labels > {given_treshold:.0%}:  vineyard = {y_b.sum():,} of {y_b.size:,}   share = {y_b.mean():.2%}")


# for (c) option - the datapoint is the whole image: vineyard if share of pixels with value 1 > threshold
print(f"     ")
print(f"    data point: (c) whole img ")
# Step: share of vineyard pixels in the whole image (0 and others count as not vineyard)
img_share = (labels == 1).mean()                                      # one number
# Step: one label per threshold (m = 1, so the share of vineyard data points is 0% or 100%)
for given_treshold in [threshold_25, threshold_50, threshold_75]:
    y_c = np.array([img_share > given_treshold])                                   # shape (1,)
    print(f"(c) image label > {given_treshold:.0%}:  image share = {img_share:.2%}   "
          f"vineyard = {y_c.sum()} of {y_c.size}   share = {y_c.mean():.0%}")

## ChECKS
# check for patches: mean share of all patches = share of vineyard pixels from (a)
print(f"     ")
print(f"    My checks task 3: ")
print(f"    check for patches: ")
# print(share)
mean_share = vine_share.mean()             # average vineyard share over all patches
pixel_share = y_pixel.mean()          # vineyard share over all pixels
status = "OK" if np.isclose(mean_share, pixel_share) else "NOT OK"
print(f"    check: mean share = {mean_share:.1%}   pixels = {pixel_share:.1%}   -> {status}")


print(f"     ")

# ---- checks task 3  by LLM ----
print(f"     ")
print(f"     ---- checks task 3  by LLM ---- ")
print(f"    label values: {np.unique(labels)}")
print(f"    check len X == len y_pixel: {len(X):,} vs {len(y_pixel):,}   "
      f"-> {'OK' if len(X) == len(y_pixel) else 'NOT OK'}")
print(f"    check len X_b == len vine_share: {len(X_b):,} vs {len(vine_share):,}   "
      f"-> {'OK' if len(X_b) == len(vine_share) else 'NOT OK'}")
print(f"    check shares between 0 and 1: "
      f"-> {'OK' if ((vine_share >= 0) & (vine_share <= 1)).all() else 'NOT OK'}")

# STEP 4: the baseline. For pixels, and for patches at each threshold: print the
#         vineyard share and the accuracy of the constant majority prediction.

# STEPS 5 and 6 are written in A1_whytrail.txt, not here.
