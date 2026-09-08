# G1 Dex1 Multi-Shape Tray And Scale Design

## Goal

Extend the existing G1 Dex1 multi-shape PnP experiment so the policy must handle more object geometries, uniform object-size variation, and a shallow-tray support geometry while keeping the existing center-based grasp gates unchanged.

## Scope

- Keep the existing eight training shapes and three held-out shapes.
- Add wine, coke, soup can, mango, pear, candle, sneaker, soap dispenser, sponge, toy car, toy ship, orange peel, rotten apple, rotten banana, trash can, and bowl to the training split. Ketchup already exists.
- Use the object scales from Mobile Bench. Use `0.005` for sneaker and half of the Mobile Bench `bowl_0` scale for bowl.
- Spawn training objects at uniform scale factors `0.8`, `0.9`, `1.0`, `1.1`, and `1.2`. Each environment keeps one fixed shape/scale pair so physics scaling is authored at spawn time.
- Keep visual evaluation at one standard-size environment per shape.
- Replace both multi-shape support pedestals with compound shallow trays: a 0.5 m pedestal plus four 0.12 m rim walls. The physical tray remains hidden and the existing marker path renders the correct pose during play.
- Preserve all grasp-window and center-distance reward logic.

## Shape Encoding

The stored BPS descriptor remains canonical per object. Runtime uniform scaling reconstructs each canonical nearest surface point as `basis + offset`, multiplies that point by the active size factor, then converts it back to a descriptor relative to the unchanged BPS basis. Geometry centers and bounds use the same factor. This keeps the USD, placement geometry, and BPS observation mutually consistent without changing observation dimensionality.

## Logging

Keep global DRC distance/contact/grasp/progress/weight metrics, global mass and fall metrics, action saturation, non-finite safety metrics, task success, and a small motion-quality subset. Remove per-link forces, intermediate close gates, duplicated hand/contact summaries, command-coordinate summaries, curriculum range endpoints, and gripper target-buffer diagnostics.

For every shape, log only `Shape/<name>_grasp_rate`, defined as the fraction of that shape's environments whose EMA grasp progress is at least `0.5` at the current step.

## Validation

- Static tests cover the requested asset split/scales, five size factors, spawn-variant decoding, BPS scale consistency, shallow-tray configuration, compact logging, and one-environment-per-shape play configuration.
- Regenerate the BPS archive from vendored USD files and verify every requested asset appears.
- Run the focused multi-shape and reward tests, then a small headless Isaac Lab reset/step smoke test if the local simulator is available.
