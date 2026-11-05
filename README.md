# epart-detect

Finds the position and rotation of electronic parts whose package is already known, from a camera image: one part held on a pick-and-place nozzle, or a group of loose parts on a plain background. The package outline (body plus leads, from a table of datasheet dimensions in `epart_detect/packages.toml`) is fitted up to scale, so no camera calibration is needed; a known px/mm can be given instead.

- `mise run detect IMAGE --package 0603 --overlay out/x.png` - poses of a group as JSON; add `--nozzle` for one part on a nozzle
- `mise run synth out/x.png --package SOT-23 --truth out/x.json` - synthetic scene with known poses
- `mise run demo` - a group and a nozzle example in `out/`
- `mise run bench` - accuracy table on synthetic parts
- `mise run test` - test suite

Output conventions, design decisions and limits: [NOTES.md](NOTES.md). Components: [ARCHITECTURE.md](ARCHITECTURE.md). Measured accuracy: [docs/bench-2026-11-05.md](docs/bench-2026-11-05.md).
