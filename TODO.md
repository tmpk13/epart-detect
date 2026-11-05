# TODO

- Run both front ends on real camera images (nozzle bottom camera, tray of loose parts) and retune segmentation
- Nozzle-tip mask or nozzle-only reference frame for parts smaller than the nozzle
- Appearance check for pin 1 and polarity on symmetric outlines (SOD-123 band, SOIC-8 dot, QFN mark)
- Upside-down and tombstoned detection for loose chips
- Split touching parts instead of only flagging them
- Analytic Jacobian in fit.refine to cut the 25-55 ms per part
