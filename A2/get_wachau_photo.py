# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
# ///
"""
get_wachau_photo.py -- download the two data files used in A1-A4. Run it once:

    uv run get_wachau_photo.py

It writes, next to this file,
  wachau_ortho.jpg    the aerial photograph: 9216 x 4608 pixels, 0.79 m per pixel,
                      7.3 x 3.7 km of the Wachau with Weissenkirchen, Rossatz, Duernstein
                      and the Loiben slopes. About 10 MB, one to three minutes.
  wachau_labels.png   the parcel mask on the same pixel grid, one number per pixel:
                      0 = no parcel, 1 = vineyard, 2 = other crop. About one minute.
A file that is already there, with the right size, is not downloaded again.

Where the two files come from
  The photograph is served by basemap.at, the open orthophoto service of the Austrian
  federal states (CC BY 4.0). Map services cut the globe into 256 x 256 pixel tiles,
  addressed by zoom level, column x and row y. This script asks for the 36 x 18 tiles
  at zoom level 17 that cover one box of longitudes and latitudes, and pastes them
  into one image.
  The parcels come from INVEKOS, the register in which every Austrian farmer declares
  the fields they farm to receive subsidies, published by AgrarMarkt Austria as open
  data (CC BY 3.0 AT). The script asks the INVEKOS web service for every parcel inside
  the same box, as polygons in longitude and latitude, and draws each polygon onto
  the photograph's pixel grid: value 1 if the declared crop is WEIN (vines), 2 for
  any other crop. Pixels no polygon covers keep the value 0: forest, roads, roofs,
  water and every field nobody declared. Both files use the same box, the same
  zoom level and the same tile grid, which is why they line up pixel for pixel.

Do not crop, resize or re-save either file: they would no longer line up.

Credit both sources wherever you show the data:
  Datenquelle: basemap.at (CC BY 4.0); INVEKOS Schlaege 2025, AMA (CC BY 3.0 AT)
"""
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None            # the photograph is 42 megapixels; Pillow refuses it by default

HERE = Path(__file__).resolve().parent   # both files are written next to this script
PHOTO = HERE / "wachau_ortho.jpg"
MASK = HERE / "wachau_labels.png"

# The photographed box, in longitude / latitude (WGS84, the coordinates of any GPS).
BOX = dict(west=15.452, south=48.380, east=15.548, north=48.411)
ZOOM = 17                                # zoom level 17 = 0.79 m per pixel at this latitude
TILE = 256                               # every map tile is 256 x 256 pixels
EXPECTED = (9216, 4608)                  # width x height of the finished photograph
ORTHO_URL = "https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg"
INVEKOS_URL = ("https://gis.lfrz.gv.at/api/geodata/i009501/ogc/features/v1/collections/"
               "i009501:invekos_schlaege_2025_1_polygon/items")
HEADERS = {"User-Agent": "MLR1ILV-course/1.0 (IMC Krems teaching material)"}


# ---------------------------------------------------------------------------
# Map geometry: longitude / latitude  <->  tile numbers  <->  pixels
# ---------------------------------------------------------------------------
def tile_xy(lat, lon, z=ZOOM):
    """Tile coordinates (column x, row y) of the point lat, lon at zoom level z.

    Web maps use the Mercator projection and cut the world into 2**z by 2**z tiles.
    Longitude maps linearly onto the column; latitude is stretched by the Mercator
    formula and maps onto the row. The integer parts are the tile numbers, the
    fractional parts say where inside the tile the point lies.
    """
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_deg(x, y, z=ZOOM):
    """The inverse: latitude and longitude of the north-west corner of tile (x, y)."""
    n = 2 ** z
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))), x / n * 360 - 180


def geometry():
    """The tile grid that covers BOX, and the exact bounds of that grid.

    The grid runs from the tile of the north-west corner to the tile of the
    south-east corner (tile rows count from the north). Its bounds are a little
    larger than BOX, because tiles are never cut.
    """
    x0, y0 = tile_xy(BOX["north"], BOX["west"])
    x1, y1 = tile_xy(BOX["south"], BOX["east"])
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1) + 1, int(y1) + 1
    north, west = tile_deg(tx0, ty0)
    south, east = tile_deg(tx1, ty1)
    return dict(x0=tx0, y0=ty0, nx=tx1 - tx0, ny=ty1 - ty0,
                width=(tx1 - tx0) * TILE, height=(ty1 - ty0) * TILE,
                north=north, south=south, west=west, east=east)


def to_pixel(lat, lon, meta):
    """Pixel coordinates (column, row) of the point lat, lon on the photograph."""
    px, py = tile_xy(lat, lon)
    return (px - meta["x0"]) * TILE, (py - meta["y0"]) * TILE


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def fetch(url, tries=4, timeout=90):
    """Download one URL; retry a few times, because a single request can time out."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:            # noqa: BLE001 -- any network error: wait, then retry
            if k == tries - 1:
                raise
            time.sleep(2 + 2 * k)


def download_photo(meta) -> Image.Image:
    """Assemble the photograph from map tiles and save it, unless it is already here."""
    if PHOTO.exists():
        im = Image.open(PHOTO)
        if im.size == EXPECTED:
            print(f"{PHOTO.name} is already here ({im.size[0]} x {im.size[1]} px). Not downloading again.")
            return im
        print(f"{PHOTO.name} exists but is {im.size[0]} x {im.size[1]} px, not "
              f"{EXPECTED[0]} x {EXPECTED[1]}: downloading it again.")
    nx, ny = meta["nx"], meta["ny"]
    print(f"Photograph: downloading {nx * ny} tiles from basemap.at ({meta['width']} x {meta['height']} px) ...")
    img = Image.new("RGB", (meta["width"], meta["height"]))     # an empty canvas
    for i in range(nx):
        for j in range(ny):
            tx, ty = meta["x0"] + i, meta["y0"] + j
            tile = Image.open(io.BytesIO(fetch(ORTHO_URL.format(z=ZOOM, x=tx, y=ty))))
            img.paste(tile, (i * TILE, j * TILE))                # tile (i, j) goes to pixel (i*256, j*256)
        print(f"  column {i + 1}/{nx}", end="\r", flush=True)
    print()
    if img.size != EXPECTED:
        sys.exit(f"unexpected size {img.size}; expected {EXPECTED}. Tell the instructor.")
    img.save(PHOTO, quality=88)
    print(f"  wrote {PHOTO.name}: {img.size[0]} x {img.size[1]} px, {PHOTO.stat().st_size / 1e6:.1f} MB")
    return img


def fetch_parcels(meta, pad=0.004, page=100, cap=20000):
    """Every INVEKOS parcel that touches the box, as GeoJSON features.

    The web service answers at most 100 parcels per request, so the parcels are
    fetched page by page; the box is padded a little so parcels cut by the edge are
    included. Each feature carries the parcel outline (a polygon in longitude and
    latitude) and the declared crop, in the property snar_bezeichnung.
    """
    bbox = f"{meta['west'] - pad},{meta['south'] - pad},{meta['east'] + pad},{meta['north'] + pad}"
    feats, start, total = [], 0, None
    while start < cap:
        url = f"{INVEKOS_URL}?bbox={bbox}&limit={page}&startIndex={start}&f=application%2Fgeo%2Bjson"
        data = json.loads(fetch(url))
        got = data["features"]
        feats += got
        total = data.get("numberMatched") or total
        print(f"  {len(feats)}/{total} parcels", end="\r", flush=True)
        start += len(got)                                  # advance by what actually arrived
        if not got or (total and start >= total):
            break
        time.sleep(0.15)                                   # be polite to the server
    print()
    return feats


def rasterise(feats, meta) -> np.ndarray:
    """Draw every parcel polygon onto the pixel grid: 1 = WEIN (vines), 2 = other crop.

    Other crops are drawn first and vineyards last, so where two declared parcels
    overlap the vineyard wins. Pixels no polygon covers stay 0.
    """
    mask = Image.new("L", (meta["width"], meta["height"]), 0)
    draw = ImageDraw.Draw(mask)
    for want_wein in (False, True):
        for f in feats:
            if (f["properties"].get("snar_bezeichnung") == "WEIN") != want_wein:
                continue
            geom = f["geometry"]
            rings = geom["coordinates"] if geom["type"] == "Polygon" else \
                [ring for polygon in geom["coordinates"] for ring in polygon]
            for ring in rings:
                pts = [to_pixel(lat, lon, meta) for lon, lat, *_ in ring]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=1 if want_wein else 2)
    return np.asarray(mask)


def download_mask(meta) -> np.ndarray:
    """Fetch the parcels and burn the mask, unless a mask of the right size is here."""
    if MASK.exists():
        labels = np.asarray(Image.open(MASK))
        if labels.shape == (meta["height"], meta["width"]):
            print(f"{MASK.name} is already here ({labels.shape[1]} x {labels.shape[0]} px). Not downloading again.")
            return labels
        print(f"{MASK.name} exists but has the wrong size: downloading it again.")
    print("Parcel mask: fetching the INVEKOS parcels inside the box ...")
    feats = fetch_parcels(meta)
    n_wein = sum(f["properties"].get("snar_bezeichnung") == "WEIN" for f in feats)
    print(f"  {len(feats)} parcels, {n_wein} of them WEIN (vines)")
    labels = rasterise(feats, meta)
    Image.fromarray(labels).save(MASK, optimize=True)
    print(f"  wrote {MASK.name}: {labels.shape[1]} x {labels.shape[0]} px")
    return labels


# ---------------------------------------------------------------------------
def main() -> int:
    meta = geometry()
    photo = download_photo(meta)
    labels = download_mask(meta)
    if labels.shape != (photo.size[1], photo.size[0]):     # PIL size is (width, height)
        sys.exit("the mask and the photograph have different sizes. Tell the instructor.")
    counts = np.bincount(labels.ravel(), minlength=3)      # counts[v] = pixels with label v
    print("\nhistogram of the labels:")
    for value, meaning in ((0, "no parcel"), (1, "vineyard"), (2, "other crop")):
        print(f"   {value} = {meaning:<11}{counts[value] / counts.sum() * 100:5.1f} % of the pixels")
    print("The photograph and the mask share one pixel grid. You are ready for A1.py.")
    print("Datenquelle: basemap.at (CC BY 4.0); INVEKOS Schlaege 2025, AMA (CC BY 3.0 AT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
