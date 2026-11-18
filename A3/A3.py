# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow", "scikit-learn"]
# ///
"""
A3 -- Eleven scikit-learn models on the two S5 tasks
Concepts: model selection by training and validation error; ridge and logistic regression, k nearest neighbours, tree ensembles
Starts: Sessions 5 (Wed 21.10, online) and 6 (Sat 31.10)   |   Due: Thu 19.11.2026, 23:59

WRITE YOUR PROGRAM IN THIS FILE. You get the full task in A3.pdf. This file is
the ONLY thing you hand in: attach it to the A3 Aufgabe in Teams as you go
(one file, no ZIP, Abgeben by the deadline). The manual computation goes in the
comment block at the bottom of this one.

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
  3. Run this file from its folder:                  uv run A3.py
     Need a library? One command adds it to the header at the top of this file:
                                  uv add --script A3.py numpy pillow scikit-learn
     The first run takes a few seconds: uv downloads Python and the libraries.
  Stuck for more than 15 minutes? Stop there. Open
  colab.research.google.com in the browser and work there for now; we fix your
  setup together in the session. Nobody is held up by an install in this course.
  .py files, not notebooks: a notebook does not record the order its cells ran
  in, and data leakage often comes from that unknown order.

DATA
A3.py reads wachau_ortho.jpg and wachau_labels.png from its own folder. A3.zip
(attached) holds A3.py and both files: unpack it and run. The same two files are
in demo_S5.zip (Slides/S5).

SCORING
  This file carries all 100 points: the program is the artifact, 70 = it runs
  (20) + the two tables, complete and right (50); the manual computation in
  the comment block at the bottom is 30.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
# every scikit-learn class the eleven models need, so that nothing has to be imported later
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None       # the photograph is 42 megapixels; Pillow refuses it by default
HERE = Path(__file__).resolve().parent

# ============================================================================================
# THE DATA SETS -- one training set and one validation set, for each of the two tasks
# ============================================================================================
# Session 5, unchanged: patches of 15 x 15 pixels, the western 70 % of the columns train,
# the eastern 30 % validate, one fixed seed. Why west against east and not a random split:
# neighbouring patches overlap, so a random split would put near-copies of training patches
# into the validation set and make every validation error look better than it is. Two parts
# of the valley do not overlap. You may lower M_TRAIN and M_VAL if a model is too slow
# on your laptop (say so in a comment); the lecture used 20,000 and 5,000.
PATCH = 15
C = PATCH // 2                      # the centre row and column
WEST_SHARE = 0.70
M_TRAIN, M_VAL = 20000, 5000


def load(name):
    "One of the two data files, next to this file, as A3.zip or demo_S5.zip put it."
    path = HERE / name
    if not path.exists():
        sys.exit(f"{name} not found next to A3.py: unpack A3.zip and run A3.py inside that "
                 f"folder, or put A3.py into the unpacked demo_S5 folder.")
    return np.asarray(Image.open(path))


photo = load("wachau_ortho.jpg")             # (4608, 9216, 3), values 0..255
mask = load("wachau_labels.png")             # (4608, 9216): 0 no parcel, 1 vineyard, 2 other crop
H, W, _ = photo.shape

split = int(W * WEST_SHARE)                  # the column where the west ends
# A patch centre is a (row, column) pair, drawn uniformly, C = 7 pixels away from the edges.
rng = np.random.default_rng(0)
west = np.c_[rng.integers(C, H - C, M_TRAIN), rng.integers(C, split - C, M_TRAIN)]
east = np.c_[rng.integers(C, H - C, M_VAL), rng.integers(split + C, W - C, M_VAL)]
dr, dc = np.meshgrid(np.arange(-C, PATCH - C), np.arange(-C, PATCH - C), indexing="ij")   # the 15 x 15 offsets

def patches_at(image, centres):
    "The 15 x 15 patch of `image` around every centre: (m, 15, 15, ...)."
    # numpy indexing with two integer arrays picks one pixel per (row, col) pair. The centres
    # are shaped (m, 1, 1), the offsets (15, 15); they broadcast to (m, 15, 15), so all m
    # patches are cut in one line. It works on the photograph (..., 3) and on the mask alike.
    return image[centres[:, 0][:, None, None] + dr, centres[:, 1][:, None, None] + dc]

train_patches = patches_at(photo, west).astype(np.float32)     # (M_TRAIN, 15, 15, 3)
val_patches = patches_at(photo, east).astype(np.float32)       # (M_VAL, 15, 15, 3)

# ---- task 1, regression: impute the centre pixel ------------------------------------------
# label y: the centre pixel (R, G, B); features X: the other 224 pixels, flattened, d = 672.
# In reading order the centre sits at position 7 * 15 + 7 = 112; np.delete takes it out, so
# that the label is never among the features -- a model that sees it just copies it.
centre = C * PATCH + C
y_train, y_val = train_patches[:, C, C, :], val_patches[:, C, C, :]         # train_patches has a shape (20000, 15, 15, 3): [:, C, C, :] = all patches,
X_train = np.delete(train_patches.reshape(M_TRAIN, PATCH * PATCH, 3), centre, axis=1).reshape(M_TRAIN, -1)  #delete the centre pixel from the feature - 112 pixel 
X_val = np.delete(val_patches.reshape(M_VAL, PATCH * PATCH, 3), centre, axis=1).reshape(M_VAL, -1)

# ---- task 2, classification: vineyard or not ----------------------------------------------
# label c: 1 if more than half of the patch lies in a vineyard parcel (A1's rule). The mask
# is cut into patches exactly like the photograph; (== 1).mean() is the share of vineyard
# pixels in a patch, and > 0.5 makes it a label. Features F: the whole patch, centre
# included, flattened, d = 675 -- nothing is deleted here, the centre is not the label now.
c_train = (patches_at(mask, west) == 1).mean(axis=(1, 2)) > 0.5
c_val = (patches_at(mask, east) == 1).mean(axis=(1, 2)) > 0.5
F_train, F_val = train_patches.reshape(M_TRAIN, -1), val_patches.reshape(M_VAL, -1)

print(f"task 1: X_train {X_train.shape}, y_train {y_train.shape};  X_val {X_val.shape}")
print(f"task 2: F_train {F_train.shape}, c_train {c_train.shape}, vineyard share "
      f"{c_train.mean():.1%} train / {c_val.mean():.1%} val")


# ============================================================================================
# THE TWO BASELINES -- no training; every model has to do better than these
# ============================================================================================
def mse(a, b):
    "Task 1's number: the mean squared error, over data points and colour values. Lower is better."
    return float(np.mean((a - b) ** 2))

def accuracy(a, b):
    "Task 2's number: the share of data points labelled right, 1 minus the error rate. Higher is better."
    return float(np.mean(a == b))

# task 1: predict the centre pixel as the average R, G, B of its 8 neighbours in the patch.
# No training: the prediction is computed from the patch alone. A learned model that cannot
# get below this MSE is worse than no model at all.
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def neighbour_mean(patches):
    return np.stack([patches[:, C + a, C + b, :] for a, b in NEIGHBOURS]).mean(axis=0)   # (m, 3)

print(f"\nbaseline, task 1 -- the mean R, G, B of the centre's 8 neighbours:")
print(f"  MSE train {mse(y_train, neighbour_mean(train_patches)):8.1f}   "
      f"MSE val {mse(y_val, neighbour_mean(val_patches)):8.1f}")

# task 2: a threshold on the patch's greenness -- its mean green value -- midway between the
# mean greenness of the vineyard patches and of the other patches of the training set.
# The threshold is computed from the TRAINING set only; the validation set is only classified with it.
# Vineyards are the greener (and brighter) class here, so a patch above the threshold is
# called vineyard. The constant answer is printed beside it; a model should do better than both.
g_train, g_val = train_patches[..., 1].mean(axis=(1, 2)), val_patches[..., 1].mean(axis=(1, 2))
g_vine, g_other = g_train[c_train].mean(), g_train[~c_train].mean()
threshold = (g_vine + g_other) / 2
vineyard_is_greener = g_vine > g_other
base_train = (g_train > threshold) == vineyard_is_greener
base_val = (g_val > threshold) == vineyard_is_greener
print(f"\nbaseline, task 2 -- greenness threshold {threshold:.1f} (vineyard mean {g_vine:.1f}, other {g_other:.1f}):")
print(f"  accuracy train {accuracy(c_train, base_train):6.1%}   accuracy val {accuracy(c_val, base_val):6.1%}")
majority = c_train.mean() > 0.5                                   # and the constant answer, for comparison
print(f"  always '{'vineyard' if majority else 'not vineyard'}':  accuracy train {accuracy(c_train, majority):6.1%}"
      f"   accuracy val {accuracy(c_val, majority):6.1%}")


# ============================================================================================
# YOUR PART -- the eleven models  (the artifact, 70 of the 100 points)
# ============================================================================================
# For every model: fit it on the training set, then print its name and the two numbers,
# task 1: MSE train and MSE val, task 2: accuracy train and accuracy val -- and the fit time.
# Every one is a class of scikit-learn, already imported at the top of this file.
#
#   task 1 (a list model_list_task1), on X_train / y_train, measured on X_val / y_val:

model_list_task1 = [
    #   Ridge(alpha=10.0)
    ("Ridge(alpha=10)", Ridge(alpha=10.0), "RidgeWithAlpha10"),
    #   KNeighborsRegressor(n_neighbors=1)        
    ("kNN, k=1", KNeighborsRegressor(n_neighbors=1), "KNN_k1"),
    #   KNeighborsRegressor(n_neighbors=5)
    ("kNN, k=5", KNeighborsRegressor(n_neighbors=5), "KNN_k5"),
    #   KNeighborsRegressor(n_neighbors=50)
    ("kNN, k=50", KNeighborsRegressor(n_neighbors=50), "KNN_k50"),
    #    RandomForestRegressor(n_estimators=30, max_depth=8, max_features=0.1, n_jobs=-1, random_state=0)
    ("Random forest (30 trees, depth 8)",  RandomForestRegressor(n_estimators=30, max_depth=8, max_features=0.1, n_jobs=-1, random_state=0),  "RandomForest_30_d8"),



]

RESULTS = HERE / "results" / "task1"
RESULTS.mkdir(parents=True, exist_ok=True)          # creates results/task1/ if it is missing

header = f"  {'model':<32}{'MSE train':>11}{'MSE val':>11}{'fit s':>8}"
base = (f"  {'baseline: mean of 8 neighbours':<32}"
        f"{mse(y_train, neighbour_mean(train_patches)):>11.1f}"
        f"{mse(y_val, neighbour_mean(val_patches)):>11.1f}{'-':>8}")
print("\nTASK 1: regression, MSE (lower is better)")
print(header)
print(base)



for name, model, file_name in model_list_task1:
    t0 = time.time()                               # start the stopwatch
    model.fit(X_train, y_train)                    # LEARN: only on the training set
    seconds = time.time() - t0                     # fit time in seconds
    e_t = mse(y_train, model.predict(X_train))     # training error
    e_v = mse(y_val, model.predict(X_val))         # validation error (unseen eastern patches)
    row = f"  {name:<32}{e_t:>11.1f}{e_v:>11.1f}{seconds:>8.1f}"
    print(row, flush=True)
    # save this model's result as a text file: results/task1/<file_name>.txt
    (RESULTS / f"{file_name}.txt").write_text(header + "\n" + base + "\n" + row + "\n")


#   task 2 (a list model_list_task2), on F_train / c_train, measured on F_val / c_val:
#     RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0)
#     HistGradientBoostingClassifier(max_iter=50, random_state=0)

model_list_task2 = [
    # logistic regression = a linear classifier; scaled first, C=0.01 = strong penalty
    ("Logistic regression (C=0.01)",
     make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)), "LogReg_C001"),
    # kNN: the k most similar training patches vote, the majority class wins
    ("kNN, k=1",  KNeighborsClassifier(n_neighbors=1),  "KNN_k1"),
    ("kNN, k=5",  KNeighborsClassifier(n_neighbors=5),  "KNN_k5"),
    ("kNN, k=50", KNeighborsClassifier(n_neighbors=50), "KNN_k50"),
     # random forest: 50 trees, each votes, the majority wins
    ("Random forest (50 trees)",
     RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0), "RandomForest_50"),
   
  
]

RESULTS2 = HERE / "results" / "task2"
RESULTS2.mkdir(parents=True, exist_ok=True)

header2 = f"  {'model':<32}{'acc train':>11}{'acc val':>11}{'fit s':>8}"
base2 = (f"  {'baseline: always not vineyard':<32}"
         f"{accuracy(c_train, majority):>11.1%}{accuracy(c_val, majority):>11.1%}{'-':>8}")
print("\nTASK 2: classification, accuracy (higher is better)")
print(header2)
print(base2)

for name, model, file_name in model_list_task2:
    t0 = time.time()
    model.fit(F_train, c_train)                         # LEARN: only on the training set
    seconds = time.time() - t0
    a_t = accuracy(c_train, model.predict(F_train))     # training accuracy
    a_v = accuracy(c_val, model.predict(F_val))         # validation accuracy
    row = f"  {name:<32}{a_t:>11.1%}{a_v:>11.1%}{seconds:>8.1f}"
    print(row, flush=True)
    (RESULTS2 / f"{file_name}.txt").write_text(header2 + "\n" + base2 + "\n" + row + "\n")





# One table per task, the baseline's numbers next to them. Every fit takes seconds; k nearest
# neighbours has no training at all, its cost is at prediction time (5,000 validation patches
# against 20,000 training patches), a second or two. If a row runs for minutes, something is
# wrong: lower M_TRAIN and say so in a comment.

# TODO: your program from here


# ============================================================================================
# THE MANUAL COMPUTATION  (30 of the 100 points)
#
# Work this ON PAPER, no code. Then write the answer AND the working below, line by line.
# Part 1 of the written exam asks you to do exactly this.
#
# the confusion matrix, and what accuracy does not show
#
# A learned hypothesis for vineyard / not vineyard is evaluated on 100 data points of the
# validation set. It predicts *vineyard* 22 times; 14 of those really were vineyard. The 100
# data points contained 38 vineyards in total.
#
# 1. Fill in the 2×2 confusion matrix: rows = true label, columns = prediction
#    (true positives, false positives, false negatives, true negatives).
# 2. Compute accuracy, precision and recall.
# 3. What accuracy does the constant prediction "not vineyard" have on these same 100 data points?
# 4. One sentence to the agency paying the subsidies: is the learned hypothesis more accurate
#    than the constant prediction, and what does each kind of mistake cost them?
# --------------------------------------------------------------------------------------------
# YOUR ANSWER AND YOUR WORKING, from here down:
#
#
# ============================================================================================
