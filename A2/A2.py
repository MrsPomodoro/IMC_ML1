# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow", "scikit-learn"]
# ///
"""
A2 -- An Imputer class for the centre pixel of a patch
Concepts: regression, data imputation, two baselines, training error vs validation error, a model behind an interface other people call
Starts: Session 4 (Fri 09.10, online)   |   Due: Tue 20.10.2026, 23:59

WRITE YOUR PROGRAM IN THIS FILE. You get the full task in A2.pdf. This file is
the ONLY thing you hand in: attach it to the A2 Aufgabe in Teams as you go
(one file, no ZIP, Abgeben by the deadline). There is no why-trail.

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
  3. Run this file from its folder:                  uv run A2.py
     Need a library? One command adds it to the header at the top of this file:
                                  uv add --script A2.py numpy pillow scikit-learn
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
  All 100 points are in this file: it runs and the contract holds (30),
  how far your imputer gets below the baseline (40, measured on my own
  patches), and the manual computation at the bottom (30). Using an LLM to
  write it is expected; code you cannot explain at the five-minute
  spot-check is worth nothing.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None       # the photograph is 42 megapixels; Pillow refuses it by default
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------
# HOW THIS FILE IS PUT TOGETHER
#
#   top half     the class Imputer and two small helpers. THIS is what I import from your file.
#   bottom half  everything that actually runs, under  if __name__ == "__main__".
#
# When I grade, I import this file. An import runs the top half only: it defines names, it does
# not open the photograph and it does not train anything. Then I build an Imputer of my own,
# call train() on my patches and call predict() on patches you have never seen. That is why
# everything your model needs has to end up on `self` inside train(), and not in a variable
# somewhere else in the file: when I import, none of the bottom half has run.
# ---------------------------------------------------------------------------------------------

PATCH = 15              # the patch side you train on. ANY ODD NUMBER; change it if you want to.
WEST_SHARE = 0.70       # the western 70 % of the columns train, the eastern 30 % validate


# ============================================================================================
# THE TWO HELPERS. They turn a stack of patches into the two things training needs:
# the labels (what you want to predict) and the features (what you predict it from).
# ============================================================================================

def labels_of(patches):
    """The centre pixel of every patch: shape (m, 3). This is the LABEL.

    `patches` has shape (m, H, W, 3): m data points, each H rows by W columns, each pixel three
    colour values. H and W are odd, so each patch has one pixel exactly in the middle, at row
    H // 2 and column W // 2 (for 15 x 15 that is row 7, column 7, counting from 0).

    The line below says: take every patch (the `:`), then that one row, then that one column,
    then all three colour values (the last `:`). What comes back is one row of three numbers
    per patch, so (m, 3) in total. For 15 x 15 patches: (m, 15, 15, 3) -> (m, 3).

    m is the number of data points and d the number of features, as in the lectures.
    """
    return patches[:, patches.shape[1] // 2, patches.shape[2] // 2, :]


def features_of(patches):
    """Every pixel of a patch EXCEPT the centre: shape (m, d) with d = 3 * (H * W - 1). The FEATURES.

    A model wants one flat row of numbers per data point, not a little picture. Three steps:

      1. `patches.reshape(m, Hp * Wp, 3)` lays each patch out as a list of its pixels, in
         reading order: row 0 left to right, then row 1, and so on. A 15 x 15 patch becomes a
         list of 225 pixels, each still a triple (R, G, B).

      2. The centre pixel is the label, so it must go. In that list its position is
         (Hp // 2) * Wp + Wp // 2: skip whole rows above it, then count across to its column.
         For 15 x 15 that is 7 * 15 + 7 = 112. `np.delete(..., axis=1)` drops it, leaving 224.

      3. The last `.reshape(m, -1)` flattens each remaining pixel triple into the row, so one
         data point is 224 * 3 = 672 plain numbers: R, G, B of pixel 0, then of pixel 1, ...

    If you ever change PATCH, this keeps working: it reads the shape out of what it is given.
    """
    m, Hp, Wp, _ = patches.shape                           # m data points, H rows, W columns
    flat = patches.reshape(m, Hp * Wp, 3)                  # (m, H*W, 3): the pixels in a row
    centre = (Hp // 2) * Wp + Wp // 2                      # where the middle pixel sits in that row
    return np.delete(flat, centre, axis=1).reshape(m, -1)  # (m, d): drop it, flatten


# ============================================================================================
# THE CLASS YOU HAND IN. I build it, train it and ask it to predict.
# ============================================================================================

class Imputer:
    """Predicts the colour of the centre pixel of a patch from all the other pixels.

    Use it like scikit-learn's estimators, except that the two methods are called train and
    predict here (scikit-learn says fit and predict), and train takes no labels.
    """

    def __init__(self):
        # Everything train() learns lives here. It starts empty, so that predict() can tell
        # "no training has happened yet" apart from "trained, here is the answer".
        self.model = None

    def train(self, patches):
        """Learn from a stack of patches. Returns self, so  Imputer().train(...)  works.

        patches : numpy array (m, H, W, 3), m data points, all of ONE odd size, values 0..255

        No labels are passed in, and none are needed: the label of a patch is its own centre
        pixel, so this method takes the labels out of the patches itself. That is the whole
        trick of an imputer -- the answer is already in the data, you just hide it from the
        features and ask the model to put it back.

        It has to finish within two minutes on 5,000 patches of 15 x 15.
        """
        patches = np.asarray(patches, dtype=np.float32)
        y = labels_of(patches)         # (m, 3)  what you are learning to predict
        X = features_of(patches)       # (m, d)  what you are allowed to look at, d = 672

        # ---- My solution ----------------------------------------------
    
        # dummy setup self.model = patches.reshape(-1, 3).mean(axis=0)      # one colour: (3,)
        #
        # TODO: replace the line above with a model that actually looks at X, for example
        #           from sklearn.linear_model import LinearRegression
        #           self.model = LinearRegression().fit(X, y)
        #       and store anything predict() will need, such as the patch side you trained on:
        #           self.patch_side = patches.shape[1]
        # -----------------------------------------------------------------------------------
        from sklearn.linear_model import LinearRegression, Ridge

        #  Model 1: Linear Regression  (E_v = 7.6) 
        # self.model = LinearRegression().fit(X, y)

        # Model 2: Ridge regression 
        self.model = Ridge(alpha=1000).fit(X, y)
        # self.model = Ridge(alpha=10000000).fit(X, y)
        self.patch_side = patches.shape[1]      
        self.mean_colour = patches.reshape(-1, 3).mean(axis=0)   # the baseline as a fallback
        return self

    def _to_train_size(self, patch):
        """Cut or pad ONE patch (H, W, 3) to the training size (P, P, 3). The centre stays centred."""
        H, W, _ = patch.shape
        P = self.patch_side

        # mean colour of all pixels EXCEPT the centre: used to fill the padding
        keep = np.ones((H, W), dtype=bool)
        keep[H // 2, W // 2] = False
        fill = patch[keep].mean(axis=0) if keep.any() else self.mean_colour

        # 1. cut: keep at most P rows and P columns, symmetrically around the centre
        r0 = max((H - P) // 2, 0)
        c0 = max((W - P) // 2, 0)
        part = patch[r0:r0 + min(H, P), c0:c0 + min(W, P)]
        h, w = part.shape[:2]

        # 2. pad: place that piece in the middle of a P x P canvas filled with `fill`
        out = np.empty((P, P, 3), dtype=np.float32)
        out[:] = fill
        R0, C0 = (P - h) // 2, (P - w) // 2
        out[R0:R0 + h, C0:C0 + w] = part
        return out

    
    def predict(self, patch):
        """Predict the centre pixel of ONE patch.

        patch   : numpy array (H, W, 3), H and W ODD, ANY odd size, colour values 0..255
        returns : an array of three numbers, the predicted R, G, B of the centre pixel,
                  or the integer -1 if the input is not of that form, or if train() has
                  not been called yet.

        Two rules the test run checks:
          * never read patch[H // 2, W // 2]. That pixel is the answer, not a feature.
          * accept any odd size. I will hand you 9 x 21 and 21 x 21 as well as 15 x 15,
            so a model trained on one size has to cut or pad what it is given.
        """
        patch = np.asarray(patch)

        # three numbers come back only for a patch that really is a patch:
        if patch.ndim != 3 or patch.shape[2] != 3:
            return -1                                      # not (H, W, 3) at all
        if patch.shape[0] % 2 == 0 or patch.shape[1] % 2 == 0:
            return -1                                      # an even side has no middle pixel
        if self.model is None:
            return -1                                      # nobody has called train() yet

        # ---- My solution ----------------------------------------------

        # baseline:         return self.model         # It answers the same colour for every patch, without looking at a single pixel of it.

        # TODO: look at the pixels instead. features_of() expects a STACK of patches, so one
        #       patch becomes patch[None] of shape (1, H, W, 3), and its feature row is
        #           features_of(patch[None].astype(np.float32))[0]
        #       Remember that `patch` may be bigger or smaller than what you trained on.
        # -----------------------------------------------------------------------------------
        p = self._to_train_size(patch.astype(np.float32))  # (15, 15, 3), centre still in the middle
        x = features_of(p[None])                            # (1, 672): neighbours as one row
        guess = self.model.predict(x)[0]                  # (3,): R, G, B of the centre pixel
        return np.clip(guess, 0, 255)                      # colors are from 0 to 255

# ============================================================================================
# YOUR OWN EXPERIMENTS. None of this runs when I import the file.
# ============================================================================================

def load(name):
    path = HERE / name
    if not path.exists():
        sys.exit(f"{name} not found next to A2.py: run  uv run get_wachau_photo.py  once.")
    return Image.open(path)


def cut(photo, centres):
    "The patch around every centre: shape (m, PATCH, PATCH, 3), values 0..255."
    C = PATCH // 2
    # dr and dc hold the row and column offsets of a patch around its centre: -C ... +C.
    dr, dc = np.meshgrid(np.arange(-C, PATCH - C), np.arange(-C, PATCH - C), indexing="ij")
    # Adding them to every centre at once gives all the patches in one go, no Python loop.
    return photo[centres[:, 0][:, None, None] + dr,
                 centres[:, 1][:, None, None] + dc].astype(np.float32)


def build_patches(m_train=20000, m_val=5000, seed=0):
    """Patches from the west to train on, patches from the east to validate on.

    Two things to notice, both of them on purpose:

      * WEST AND EAST. Training and validation patches come from different ground, not from
        shuffled rows of one table. Two patches cut next to each other show almost the same
        piece of the valley, so a random split would put near-copies on both sides and your
        validation error would be flattering nonsense.

      * RANDOM POSITIONS, NOT A GRID. The centres are drawn anywhere. A model trained only at,
        say, every 32nd pixel quietly learns that position inside the photograph's JPEG
        compression blocks, scores beautifully at those positions and badly everywhere else --
        and a dead pixel sits anywhere. I test your Imputer at random positions too.
    """
    photo = np.asarray(load("wachau_ortho.jpg").convert("RGB"))      # values 0..255
    H, W, _ = photo.shape
    if (H, W) != (4608, 9216):
        print(f"WARNING: wachau_ortho.jpg here is {W} x {H}, not the 9216 x 4608 photograph")
        print("         of A1. Run  uv run get_wachau_photo.py  once to fetch the real one.")
        print("         Everything still runs, but your numbers will not match the sheet.")
    C = PATCH // 2
    split = int(W * WEST_SHARE)                       # the column where the west ends
    if split - C <= C or W - C <= split + C:
        sys.exit(f"the photograph is only {W} px wide: too narrow to split into a west and an east")
    rng = np.random.default_rng(seed)                 # a fixed seed, so your runs repeat
    west = np.c_[rng.integers(C, H - C, m_train), rng.integers(C, split - C, m_train)]
    east = np.c_[rng.integers(C, H - C, m_val), rng.integers(split + C, W - C, m_val)]
    return photo, cut(photo, west), cut(photo, east)


def error_of(imputer, patches):
    """The mean squared error of an Imputer over `patches`.

    It goes through predict() one patch at a time, exactly the way I will, so it measures
    whatever your class really does, not what you meant it to do. If that gets slow while you
    experiment, pass a slice: error_of(imputer, val_patches[:500]).
    """
    guess = np.array([imputer.predict(p) for p in patches], dtype=float)
    return float(np.mean((labels_of(patches) - guess) ** 2))


if __name__ == "__main__":
    photo, train_patches, val_patches = build_patches()
    print(f"patches of {PATCH} x {PATCH}: m = {len(train_patches):,} train, "
          f"m = {len(val_patches):,} validate, d = {features_of(train_patches[:1]).shape[1]}")

    # HOW YOUR IMPUTER IS JUDGED, so that you can watch it yourself while you work.
    #   The baseline needs no training at all: the average colour of the training pixels,
    #   answered for every data point.
    #       baseline = train_patches.reshape(-1, 3).mean(axis=0)
    #   Every error here is the mean squared error, over data points and colour values:
    #       np.mean((labels_of(val_patches) - prediction) ** 2)
    #   That baseline reaches about 1300. Print your own validation error next to it and
    #   watch the gap while you try things. You are not asked to hand any number in; I
    #   measure your Imputer myself, on my own patches.
    
    print("\n=== Model: Linear Regression ===")
    # print("\n=== Model: Ridge regression, alpha=1000 000 ===")
    imputer = Imputer().train(train_patches)
    print(f"training error   E_t = {error_of(imputer, train_patches):8.1f}")
    print(f"validation error E_v = {error_of(imputer, val_patches):8.1f}   "
          f"<- the baseline was about 1300")

    # The contract the test run depends on. As shipped this prints the same three numbers
    # twice (the mean label), then -1 twice. It must keep printing three numbers, three
    # numbers, -1, -1 once your own model is in there.
    first = val_patches[0]                                 # a real 15 x 15 patch of the east
    print(f"{PATCH} x {PATCH}      ->", imputer.predict(first))
    print(" 9 x 21      ->", imputer.predict(photo[:9, :21]))
    print("16 x 16      ->", imputer.predict(photo[:16, :16]))
    print("before train ->", Imputer().predict(first))


# ==============================================================================================
# THE MANUAL COMPUTATION (30 of the 100 points)
#
# Work this ON PAPER. No code, no LLM. Then write the answer AND the working into the comment
# block below, line by line. Part 1 of the written exam asks you to do exactly this.
#
# one step of gradient descent, by hand
#
# Linear regression with one feature and one model parameter: the hypothesis is h(x) = w·x
# (a line through the origin, no offset), the loss is the squared error loss. The training set
# is three days of Krems 2023, x the daily minimum and y the daily maximum temperature in
# °C: (4.2, 11.9), (9.4, 23.9), (8.0, 19.9). The current weight is w⁽⁰⁾ = 1.5 and the
# learning rate is η = 0.01.
#
# 1. Write the training error f(w), the average squared error loss on the three days, as a
#    formula. Compute the three prediction errors y − w·x at w = 1.5, and f(1.5).
# 2. Write the derivative f′(w) as a formula and compute f′(1.5).
# 3. One gradient descent step: w⁽¹⁾ = w⁽⁰⁾ − η·f′(w⁽⁰⁾). Compute w⁽¹⁾ and f(w⁽¹⁾).
#    Did the training error go down?
# ----------------------------------------------------------------------------------------------
# YOUR ANSWER AND YOUR WORKING, from here down:
#
# 1)
#
# 2)
#
# 3)
#
# ==============================================================================================
