"""Recover plot geometry from the locally archived PNGs; never rerun the model.

Requires Pillow and NumPy. Run from any directory. All image readings retain
their provenance: this is not an export of the original numerical experiment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def pixels(name):
    return np.array(Image.open(ROOT / 'figures' / f'{name}.png').convert('RGB')).astype(int)


def breaks():
    image = pixels('break-heatmap')
    results = []
    for top, bottom, channel in [(33, 362, 0), (408, 737, 2)]:
        recovered = []
        for h in range(24):
            y = round(top + (h + .5) * (bottom - top) / 24)
            row = image[y, 58:1275]
            mask = row[:, channel] > row[:, 2-channel] + 50
            xs = np.flatnonzero(mask) + 58
            assert len(xs) in (1, 2, 3), (h, xs)
            index = round((float(xs.mean()) - 57.5) / 1217.5 * 604 - .5)
            assert index % 5 == 0 and 0 < index < 604, (h, index)
            # At this resolution a matrix column is ~2 pixels wide. Check
            # every colored pixel falls within the recovered cell footprint.
            center = 57.5 + (index + .5) * 1217.5 / 604
            assert np.max(np.abs(xs-center)) < 1.6
            recovered.append(index)
        results.append(recovered)
    assert results[0][0] == 445  # Also printed numerically in evaluations.ipynb.
    return results


def length_histograms(locations):
    result = []
    for label, color, locations_one in zip(['Predicted', 'Observed'], ['#ff0000', '#0000ff'], locations):
        lengths = [length for b in locations_one for length in [b, 604-b]]
        counts, edges = np.histogram(lengths, bins=20)
        assert sum(counts) == 48
        result.append(dict(name=label, color=color, bins=[
            dict(left=float(edges[i]), right=float(edges[i+1]), count=int(n), countApproximate=False)
            for i, n in enumerate(counts)]))
    return result


def slope_histograms():
    image = pixels('segment-trends')
    output = []
    for channel, label, color in [(0, 'Predicted', '#ff0000'), (2, 'Observed', '#0000ff')]:
        mask = image[:, :, channel] > image[:, :, 1] + 20
        mask[:33] = False
        mask[494:] = False
        mask[:80, 710:] = False  # Exclude the source legend.
        _, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max()) + 1
        bins = []
        for i in range(20):
            x0, x1 = left + (right-left)*i/20, left+(right-left)*(i+1)/20
            x = round((x0+x1)/2)
            ys = np.flatnonzero(mask[:, x])
            count = round((494-int(ys.min()))/(414/16)) if len(ys) else 0
            # Calibrate using the printed -0.0005 and +0.0001 ticks.
            to_value = lambda px: -.0005 + (px-92)/671*.0006
            bins.append(dict(left=to_value(x0), right=to_value(x1), count=count, countApproximate=False))
        assert sum(b['count'] for b in bins) == 48
        output.append(dict(name=label, color=color, bins=bins))
    return output


def score_histogram():
    image = pixels('trust-score')
    bins = []
    for i in range(30):
        x0, x1 = 115+820*i/30, 115+820*(i+1)/30
        x = round((x0+x1)/2)
        column = image[33:337, x]
        mask = ((column[:, 0] > column[:, 1]+20) & (column[:, 2] > column[:, 1]+20)) | (column.max(axis=1) < 60)
        ys = np.flatnonzero(mask)
        height = 337-int(ys.min()+33) if len(ys) else 0
        # 1000 counts span 270 pixels. Subpixel bar heights and outlines
        # prevent exact integer recovery; retain a +/-4 count reading range.
        estimate = height/270*1000
        to_value = lambda px: .2+(px-160)/808*.5
        bins.append(dict(left=to_value(x0), right=to_value(x1), count=estimate,
                         countApproximate=True, low=max(0, estimate-4), high=estimate+4))
    return [dict(name='Composite diagnostic', color='#800080', bins=bins)]


def trace_path(mask):
    """Merge horizontal pixel runs into vector rectangles without interpolation."""
    active, rectangles = {}, []
    for y, row in enumerate(mask):
        changes = np.diff(np.r_[False, row, False].astype(int))
        runs = set(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        for run in list(active):
            if run not in runs:
                start = active.pop(run)
                rectangles.append((int(run[0]), start, int(run[1]-run[0]), y-start))
        for run in sorted(runs):
            if run not in active:
                active[run] = y
    for (left, right), start in active.items():
        rectangles.append((int(left), start, int(right-left), len(mask)-start))
    assert sum(w*h for _, _, w, h in rectangles) == int(mask.sum())
    return ''.join(f'M{x} {y}h{w}v{h}h-{w}z' for x, y, w, h in rectangles)


def overlay_traces(locations):
    """Digitize the visible colored traces; do not invent hidden samples.

    Source-over blue followed by red on white gives B=255*(1-alpha_red),
    G=B*(1-alpha_blue). This separates the visible color coverages where
    neither curve is completely occluded. Quantized masks retain only
    coverage >= 8%. Readouts are visible vertical ranges, not sample values.
    """
    image = pixels('horizon-overlays')
    columns, rows = [70, 469, 868, 1267, 1666, 2065], [46, 434, 821, 1209]
    # Linear y calibrations from the printed tick centers in the archived PNG.
    # These are image coordinates, deliberately not model-output precision.
    scales = [(-.000744041, .2214275), (-.0010324394, .30752075),
              (-.0005465132, .16277061), (-.0008305568, .24721619),
              (-.0006908917, .19696612), (-.0008305556, .23189223),
              (-.0007462687, .20746269), (-.0005555556, .15333333),
              (-.0011272054, .33486169), (-.0021570148, .63899625),
              (-.0012555953, .37291321), (-.0015625, .4640625),
              (-.0011235868, .33458926), (-.0018181818, .54181818),
              (-.0015625, .465625), (-.002, .596),
              (-.0021355256, .63590383), (-.0021276596, .63404255),
              (-.002014376, .59841458), (-.0019230769, .57115385),
              (-.001797167, .53388779), (-.0016393443, .48688525),
              (-.0013992424, .4156388), (-.0011513033, .3279586)]
    horizons = []
    for h in range(24):
        left, top = columns[h % 6], rows[h // 6]
        crop = image[top:top+310, left:left+312].astype(float)
        red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        colored = (crop.max(axis=2)-crop.min(axis=2) > 10) & (np.abs(red-(255-blue+green)) < 4)
        blue_alpha = 1-np.divide(green, blue, out=np.ones_like(blue), where=blue > 0)
        masks = [colored & (blue_alpha >= .08), colored & (1-blue/255 >= .08)]
        series = []
        for si, (name, color, mask) in enumerate(zip(['Forecast', 'Observed'], ['#0000ff', '#ff0000'], masks)):
            if h == 0:
                mask[5:57, 7:125] = False  # Source legend, not signal.
            # Remove dashed break guides outside the neighboring visible trace
            # envelope; the browser draws independent vector guides instead.
            bx = round((locations[si][h]*10+301.5)/6633*312)
            neighbors = [i for i in range(max(0, bx-7), min(312, bx+8)) if abs(i-bx) >= 4]
            ys = np.flatnonzero(mask[:, neighbors].any(axis=1))
            if len(ys):
                a, b = max(0, int(ys.min())-2), min(310, int(ys.max())+3)
                mask[:a, max(0, bx-2):min(312, bx+3)] = False
                mask[b:, max(0, bx-2):min(312, bx+3)] = False
            ranges = []
            for column in mask.T:
                indices = np.flatnonzero(column)
                ranges.append([int(indices.min()), int(indices.max())] if len(indices) else None)
            assert mask.sum() > 1000 and sum(v is not None for v in ranges) > 250
            series.append(dict(name=name, color=color, path=trace_path(mask), ranges=ranges))
        horizons.append(dict(yScale=scales[h], series=series))
    return dict(width=312, height=310, xDomain=[-301.5, 6331.5], horizons=horizons)


def build():
    locations = breaks()
    return dict(
        provenance='Geometry recovered from archived figures, not original experiment arrays.',
        sourceRevision='eb46a634c309312aff3721841beb81142d0dac4e',
        entries=604, downsampleFactor=10, predictedBreaks=locations[0], observedBreaks=locations[1],
        histograms={
            'segment-lengths': dict(title='Segment length distribution', xLabel='Segment length (downsampled entries)',
                domain=[0, 604], ticks=[0, 100, 200, 300, 400, 500, 600], yMax=16,
                note='Lengths reconstructed from the break map; two segments per horizon, 48 per series. The original 20-bin, separate-range histograms are retained.',
                series=length_histograms(locations)),
            'segment-trends': dict(title='Segment slope distribution', xLabel='Segment slope along entry order',
                domain=[-.00053, .00016], ticks=[-.0005, -.0004, -.0003, -.0002, -.0001, 0, .0001], yMax=18,
                note='Bin frequencies read from the saved figure (48 segments per series). Slope-bin boundaries are approximate image readings; original slopes were not saved.',
                approximateX=True, series=slope_histograms()),
            'trust-score': dict(title='Composite diagnostic distribution', xLabel='Composite diagnostic score',
                domain=[.1468, .705], ticks=[.2, .3, .4, .5, .6, .7], yMax=1130,
                note='Bin boundaries and counts are approximate readings of the saved plot. Small bars can be below image resolution. Count ranges reflect ±4 counts of image-reading uncertainty, not statistical confidence. The recorded run contains 6,034 entries; exact bin counts cannot be recovered.',
                approximateX=True, series=score_histogram()),
        },
        overlays=overlay_traces(locations),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'figures').glob('*.png'))})


if __name__ == '__main__':
    data = build()
    output = ROOT / 'charts-data.js'
    output.write_text('/* Generated from archived figures by scripts/read_figures.py. */\nwindow.TIMESERIES_CHARTS = '+json.dumps(data, separators=(',', ':'), allow_nan=False)+';\n')
    print('Recovered 48 break cells, 24 two-series vector trace views, two length histograms, two slope histograms, and 30 approximate diagnostic bins.')
