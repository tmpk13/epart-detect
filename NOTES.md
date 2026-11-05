# Notes

## Scope: pose of a known package, not identification

The part type and package are known before the camera looks (the machine knows what it picked, or which reel a tray of loose parts came from). The tool only answers where each part is and how it is turned. This is why the package table is data rather than a set of trained models: a new package is a few rectangles from its datasheet, and nothing has to tell an 0402 from an 0603.

## One library, two front ends

The nozzle camera and the loose-parts tray need the same three things: the package outline, segmentation against a plain background, and the outline fit. They differ only in which blobs they keep and what they report, so `detect.py` holds two thin front ends over one fit:

- group: every blob is fitted. Clipped, low-score and scale-outlier parts are flagged, not dropped, so a picker can skip or nudge them.
- nozzle: the blob covering the nozzle point (default: image center), else the nearest one. The output is the offset of the body center from the nozzle and the angle.

## Scale from proportions ("ratio" mode)

By default the scale (px/mm) is fitted per part along with position and angle; only the package's proportions constrain the fit. The coarse scale is the square root of the blob-to-outline area ratio. A group gets a relative size check for free: a part whose fitted scale is more than `--scale-tol` (10%) off the group median is a `scale_outlier` (wrong package mixed in, or two parts merged end to end).

The catch is that millimeter output then rests on nominal dimensions. Real parts vary: an 0603 is 1.6 +/- 0.15 mm long, about +/- 9%. In nozzle mode `offset_mm` scales by the same amount (0.03 mm on a 0.3 mm offset). Give `--px-per-mm` from a one-time calibration when millimeters matter; with a fixed scale a wrong-size part fails the outline score instead of becoming a scale outlier.

## Output conventions

- Image frame: +x right, +y down, (0, 0) at the center of the top-left pixel. Offsets (`offset_px`, `offset_mm`) are in these axes.
- `x`, `y` are the package origin: the body center, which is what a nozzle centers on. It is not the outline centroid (SOT-23's two-lead side pulls the centroid 0.046 mm off).
- `angle` is degrees from image +x to the package +u axis, counterclockwise as seen on screen. Angle 0 is the package as drawn in `packages.toml` (+u right, +v up; pin 1 is the first lead listed).
- The angle is reported in [0, 360 / symmetry). `symmetry` is the rotational symmetry order of the outline (chips, SOT-23-6, SOD-123 and SOIC-8: 2; SOT-23 and SOT-23-5: 1; QFN-16: 4), computed exactly from the rectangles in `package.py`.
- Mapping to machine coordinates (camera rotation, mirroring, mm per px) is the caller's job.

## Pin 1 and polarity are not resolved for symmetric outlines

The fit sees only the silhouette, so it fixes the full 360 degrees only when the leads make the outline asymmetric (SOT-23, SOT-23-5). A chip is truly symmetric, but SOD-123's cathode band, SOIC-8's pin-1 dot or bevel and QFN's pin-1 mark are on the body surface. These need a small appearance check per package (compare the two or four candidate orientations on body texture). It is not built yet.

## Bottom camera: --mirror

A bottom camera sees the underside, which is the top view flipped. `Package.mirrored()` negates v. For SOT-23 the unmirrored model fits a bottom view turned 180 degrees from the truth (the outline is symmetric left to right), so a missing `--mirror` reports pin 1 on the wrong side without any drop in score.

## Segmentation

- The foreground is Lab color distance from the background color, which is the median of a 3% border frame unless `--background` is given. Lab distance, not gray level, keeps dark bodies and bright tinned terminals at once on a mid-tone mat. On a white or silver background the terminals vanish and the outline shrinks to the body, which fails the fit. Use a mid-tone mat (blue, green, gray) that contrasts with both black epoxy and tin.
- A pixel must exceed the noise floor (border median + 6 robust sigma, at least 12 Lab units) and half the local contrast (largest distance within 2 px). The half-contrast rule puts the mask edge where a pixel is half covered. With the noise floor alone, the blur tail made every blob 0.3-0.5 px fatter on each side and the fitted scale read 2-9% high.
- No morphological opening: a 3x3 opening erased 0201s imaged 3 px wide. Specks are dropped by blob area instead (20 px, and a quarter of the median blob area in a group).
- The background is one global color, so strong vignetting or shadows next to parts are not modeled. The synthetic scenes have 15% falloff and no shadows.

## Fitting (fit.py)

1. Candidates: angles from the min-area rectangle and from the principal axis, each at four 90-degree turns, centered by matching centroids, ranked by IoU. The min-area rectangle alone fails on near-square outlines, and the principal axis alone fails on squares, so both are tried.
2. Refine the best 3 orientations (distinct modulo symmetry) by least squares over (x, y, angle, log scale) on the blob's signed distance field, sampled at outline points 1 px apart, with a soft-L1 loss. The best IoU wins.
3. Polish on the coverage map (color distance over local contrast, clipped to [0, 1]) by putting the outline on its 0.5 level. The binary mask turns a near-axis-aligned edge into a single-step staircase, and a line fitted to one step overstates the tilt by up to half. A 0603 at 1.09 degrees came out at about 1.8 degrees on every noise seed. The polish cut most 95th-percentile angle errors 2-5x (0603 at 20 px/mm: 0.44 to 0.15 degrees; QFN-16 at 10 px/mm: 1.0 to 0.21). The polish is discarded if it moves the pose more than 1 px or 2 degrees.
4. Accept: the outline score is 1 - (pixels where blob and model disagree, more than 1 px from the model outline) / model area. IoU was the first choice, but a correct fit of a 10 x 5 px 0402 scores only 0.69-0.82 IoU from edge pixels alone, so small parts all failed. The outline score ignores the edge band: it gives that 0402 0.98-1.0, and two 0603s merged side by side 0.07 (their IoU is 0.51).

## Accuracy

Measured on synthetic single parts only: [docs/bench-2026-11-05.md](docs/bench-2026-11-05.md). In short, from 20 px/mm up every package lands within 0.07 px of center and 0.26 degrees at the 95th percentile. 0402 at 10 px/mm (10 x 5 px) is down to 3.3 degrees, and 0201 at 10 px/mm is below the blob-area floor and never found. Rule of thumb: keep the part at least 15-20 px long. Real images will be worse; nothing has been run on a camera yet.

## Not handled yet

- Nozzle tip visible around a part smaller than the nozzle (0402 on a 0.7 mm tip): it merges with the part's blob. The fix is a nozzle-tip mask or subtracting a nozzle-only reference frame.
- Touching parts are flagged `low_score`, not split.
- Upside-down or tombstoned loose parts: an upside-down chip resistor shows white ceramic instead of the black coating, and a tombstoned chip shows its end face. Both have outlines that may still pass.
- Real images: lighting, shadows, specular tin and lens distortion are untested.

## Why Python

The compute target is a laptop or desktop CPU. The user accepted "Python + OpenCV" for it. Contours, distance transforms, min-area rectangles and least squares come from OpenCV and scipy, and thresholds will need tuning against real camera images, where Python is the quicker loop. The per-part work (a few hundred outline points, three least-squares runs plus the polish) costs 25-55 ms per part (0603 to SOT-23, 1280 x 960 image). That is fine for one nozzle part but slow for a large tray. The obvious savings are the numeric Jacobian (it is analytic for this residual) and refining fewer candidates when the IoU ranking is clear.
