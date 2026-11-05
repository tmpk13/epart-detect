# Architecture

```mermaid
flowchart LR
    toml[packages.toml<br/>datasheet rectangles] --> pkg[package.py<br/>Package: silhouette, area,<br/>centroid, symmetry, outline]
    img[camera image] --> seg[segment.py<br/>Lab distance from background<br/>mask + coverage map]
    seg --> fe{detect.py<br/>front end}
    fe -->|every blob| group[group mode<br/>clipped / low_score /<br/>scale_outlier / ok]
    fe -->|blob at nozzle| nozzle[nozzle mode<br/>offset from nozzle]
    group --> fit
    nozzle --> fit
    pkg --> fit[fit.py<br/>candidates, IoU rank<br/>SDF least squares<br/>coverage polish<br/>outline score]
    fit --> json[JSON on stdout<br/>cli.py]
    fit --> ov[overlay.py<br/>annotated image]
    pkg --> synth[synth.py<br/>anti-aliased scenes<br/>with known poses]
    synth --> tests[tests/]
    synth --> bench[bench.py<br/>accuracy table]
    bench --> fe
    render[render.py<br/>rectangles at a pose] -.-> fit
    render -.-> synth
    render -.-> ov
```

`render.py` is the one place rectangles become pixels, so the fit's IoU and score masks, the synthetic scenes and the overlay all use the same geometry.

Per blob, inside `fit.fit_blob`:

```mermaid
flowchart TD
    blob[blob crop, padded<br/>+ its coverage] --> cand[8 candidates:<br/>min-area rect and principal axis<br/>x four 90-degree turns]
    cand --> rank[rank by IoU, keep 3<br/>distinct modulo symmetry]
    rank --> sdf[least squares on signed<br/>distance of the binary mask]
    sdf --> pick[best IoU]
    pick --> polish[least squares on the<br/>coverage map's 0.5 level]
    polish -->|moved under 1 px and 2 degrees| out[pose, IoU, outline score, rms]
    polish -->|moved more| keep[keep the mask fit] --> out
```
