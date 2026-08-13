# OmniQueue Companion (iOS)

Native SwiftUI app for the live park planner. It runs the bundled **v2** rank-route ONNX model on-device and fetches wait times from [ThemeParks.wiki](https://themeparks.wiki). There is no GPS, and no model switcher.

Open `OmniQueueCompanion.xcodeproj` in Xcode 16 (iOS 17+), pick a Development Team for signing, and run on a device or simulator. First resolve the ONNX Runtime Swift package when Xcode prompts.

After ride or walk-graph changes, regenerate `OmniQueueCompanion/Resources/ParkData.json` from the repo root:

```bash
python3 "Companion app/export_park_data.py"
```

Copy an updated `v2.onnx` from `Park/companion/model/` into `OmniQueueCompanion/Resources/` if the checkpoint changes.

OmniQueue is not affiliated with, authorized by, or endorsed by The Walt Disney Company. Wait times are unofficial estimates powered by ThemeParks.wiki.
