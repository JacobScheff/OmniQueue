# OmniQueue Companion (iOS)

OmniQueue has a native **Planner** for Disneyland and **Forecasts** for the ten parks in the Python wait-time lab. Planner uses the bundled v2 ONNX route model and keeps guest state on-device. Forecasts uses MapKit, Swift Charts, and the forecast service over HTTPS, with the selected operating day held in memory.

Open `OmniQueueCompanion.xcodeproj` in a current Xcode release with the iOS 17+ deployment target, select a Development Team for signing, and resolve the ONNX Runtime Swift package. Set the target’s user-defined `OMNIQUEUE_FORECAST_URL` build setting to the service’s HTTPS base URL. The equivalent simulator build from this directory is:

```bash
xcodebuild -project OmniQueueCompanion.xcodeproj -scheme OmniQueueCompanion \
  -destination 'generic/platform=iOS Simulator' \
  OMNIQUEUE_FORECAST_URL="https://YOUR-SERVICE.onrender.com" build
```

See the [forecast service setup](../Wait%20Time%20Predictions/README.md#ios-forecast-service) to deploy, publish a model, and seed history before testing Forecasts.

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

A Run Script build phase stamps `MinimumOSVersion` into the bundled `onnxruntime.framework` Info.plist (the SPM XCFramework ships that key empty, which App Store Connect rejects as ITMS-90208). Set a build number higher than the last uploaded build before archiving.

In Xcode **Build Phases**, keep **Patch ONNX Runtime Info.plist** last (below **Embed Frameworks** if Xcode inserted that phase). In the Organizer archive, confirm `OmniQueueCompanion.app/Frameworks/onnxruntime.framework/Info.plist` has `MinimumOSVersion` = `17.0` before uploading.

### Before Archive → Upload

1. Confirm the marketing / privacy / support pages above are live and open in Safari.
2. In Xcode: select the **Any iOS Device** destination → **Product → Archive** → **Distribute App → App Store Connect**.
3. Export compliance: the project sets `ITSAppUsesNonExemptEncryption = NO` (HTTPS only). Answer the Connect questionnaire the same way if asked again.
4. Review App Privacy responses and the privacy policy against the deployed service’s request logging and retention. Forecast requests contain park, date, and start time. Guest preferences and visit history stay in on-device `UserDefaults`.
5. Age rating: no restricted content; travel utility.
6. Show Planner and Forecasts in screenshots and describe the independent app in the listing.
7. Suggested review notes:

> OmniQueue is an independent app with a Disneyland day planner and wait-time forecasts for ten theme parks. There is no login. Open Planner to set ride preferences and Me to set your current location. Open Forecasts, choose a park, and use the timeline to explore recorded waits and predictions. Planner recommendations run on-device. Wait forecasts come from the OmniQueue forecast service.

OmniQueue is not affiliated with, authorized by, or endorsed by The Walt Disney Company. Wait times are unofficial estimates powered by ThemeParks.wiki.
