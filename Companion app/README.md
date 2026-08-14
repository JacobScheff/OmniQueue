# OmniQueue Companion (iOS)

Native SwiftUI app for the live park planner. It runs the bundled **v2** rank-route ONNX model on-device and fetches wait times from [ThemeParks.wiki](https://themeparks.wiki). There is no GPS, and no model switcher.

Open `OmniQueueCompanion.xcodeproj` in Xcode 16 (iOS 17+), pick a Development Team for signing, and run on a device or simulator. First resolve the ONNX Runtime Swift package when Xcode prompts.

After ride or walk-graph changes, regenerate `OmniQueueCompanion/Resources/ParkData.json` from the repo root:

```bash
python3 "Companion app/export_park_data.py"
```

Copy an updated `v2.onnx` from `Park/companion/model/` into `OmniQueueCompanion/Resources/` if the checkpoint changes.

## App Store Connect

Public site URLs (also linked from the Me tab and the disclaimer):

| Field | URL |
|-------|-----|
| Marketing | https://www.jacobscheff.com/OmniQueue |
| Privacy Policy | https://www.jacobscheff.com/OmniQueue#privacy |
| Support | https://www.jacobscheff.com/OmniQueue#support |

Bundle ID: `app.omniqueue` · Display name: OmniQueue · Category: Travel · Version `1.0` / build `2` · iPhone, iOS 17+.

A Run Script build phase stamps `MinimumOSVersion` into the bundled `onnxruntime.framework` Info.plist (the SPM XCFramework ships that key empty, which App Store Connect rejects as ITMS-90208). After pulling this change, Archive **1.0 (2)** — do not reuse build `1`.

In Xcode **Build Phases**, keep **Patch ONNX Runtime Info.plist** last (below **Embed Frameworks** if Xcode inserted that phase). In the Organizer archive, confirm `OmniQueueCompanion.app/Frameworks/onnxruntime.framework/Info.plist` has `MinimumOSVersion` = `17.0` before uploading.

### Before Archive → Upload

1. Confirm the marketing / privacy / support pages above are live and open in Safari.
2. In Xcode: select the **Any iOS Device** destination → **Product → Archive** → **Distribute App → App Store Connect**.
3. Export compliance: the project sets `ITSAppUsesNonExemptEncryption = NO` (HTTPS only). Answer the Connect questionnaire the same way if asked again.
4. App Privacy nutrition labels: **no data collected**. Guest prefs / history stay in on-device `UserDefaults`. Network: ThemeParks.wiki wait board only. No tracking, no accounts.
5. Age rating: no restricted content; travel utility.
6. Screenshots: show Plan / Rides / Me without implying Disney affiliation. Keep the “Not a Disney product” framing visible or stated in the description.
7. App Review Information: paste [`app_review_notes.txt`](app_review_notes.txt) into **Notes** (under the 4,000-byte cap). Leave **Sign-In Required** off. For a Guideline 2.1 “Information Needed” rejection, also attach a physical-iPhone screen recording; the shot list is in [`APP_REVIEW_INFORMATION.md`](APP_REVIEW_INFORMATION.md).
8. Confirm the marketing page matches this binary (iPhone only, iOS 17+). Do not list iPad or Vision Pro unless those targets ship.

OmniQueue is not affiliated with, authorized by, or endorsed by The Walt Disney Company. Wait times are unofficial estimates powered by ThemeParks.wiki.
