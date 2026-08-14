# App Store Review Information (Guideline 2.1)

Apple rejected the new-app submission because App Review Information was incomplete. This file is the source of truth for the Resolution Center reply and for the **Notes** field on future submissions.

Exact Notes text (under the 4,000-byte App Store Connect limit) lives in [`app_review_notes.txt`](app_review_notes.txt). Keep that file ASCII-heavy so byte count stays under the cap.

## Reply to this rejection

App Store Connect cannot be updated from this repo. Do this on a Mac with a physical iPhone:

1. Install the **same build** Apple is reviewing (TestFlight or the submitted archive) on a physical iPhone running the **latest iOS**.
2. Delete the app first if it was already opened, so the recording starts at a cold launch and shows the disclaimer.
3. Record the script below (Control Center → Screen Recording, or QuickTime). Start recording, then tap the OmniQueue icon.
4. In App Store Connect → the rejection thread (Resolution Center / App Review messages):
   - Paste [`app_review_notes.txt`](app_review_notes.txt)
   - Attach the `.mp4`
   - Confirm **Sign-In Required** is **off** (no username/password)
5. Copy the same notes into the version’s **App Review Information → Notes** so the next upload is not missing them.
6. Before you send, open https://www.jacobscheff.com/OmniQueue in Safari. The binary is **iPhone-only**. If that page still lists iPad or Vision Pro, change it to iPhone so reviewers are not told the app ships on platforms this build does not support.

## Screen recording script (physical iPhone, latest iOS)

Start recording, then launch OmniQueue. Keep the clip to a few minutes. Narration is optional; the UI should carry the flow.

| Beat | What to show | Why Apple asked |
| --- | --- | --- |
| 1 | Home Screen → tap OmniQueue | Recording begins with launch |
| 2 | Boot card (“Fetching the wait board”) | Core network + on-device model load |
| 3 | Disclaimer sheet → **Got it — plan my day** | First-run legal copy; not a login |
| 4 | **Plan**: UP NEXT, **Why this pick?**, **Refresh waits**, **Your route** | Typical user flow |
| 5 | Tap a route stop → tap another legal option to **pin** it; wait for the ticket to rewrite; **Clear** the pin if the banner appears | Core planner feature |
| 6 | **Rides**: search a ride, set Skip/Low/Medium/High, star **Must-do**, bump **Rode it** | Prefs that drive the plan |
| 7 | **Me**: **Current stop** → pick a land or ride (this is not GPS) → set leave time | “Location” without a system permission |
| 8 | Me → **Clear data** → confirm | Closest analog to account deletion (local only) |
| 9 | Optional: sun/moon control in the top bar to flip light/dark | Harmless extra; skip if the clip is long |

Do **not** wait for a permission dialog. There are none (no Location, Camera, Contacts, or App Tracking Transparency).

Do **not** show a login, paywall, or report/block UI. Those flows do not exist.

If waits fail, stay on Plan, tap **Refresh waits**, and keep recording the error copy plus a retry. Overnight, rides may show CLOSED; that is still a live board and is enough for review.

## What the notes already tell Review

Mapped to Apple’s seven asks:

1. **Recording** — attach the physical-device clip; notes state it starts at launch and that login / IAP / UGC / ATT do not exist.
2. **Devices / OS** — physical iPhone on latest iOS (the recording) plus Xcode 16 iPhone Simulator, iOS 17+. Binary: iPhone, iOS 17+, portrait.
3. **Functions / audience** — unofficial next-ride planner for Disneyland guests; value is a short plan from waits, walking, time left, and prefs.
4. **Setup** — no credentials or sample files; numbered path through Plan / Rides / Me.
5. **External services** — ThemeParks.wiki public wait API; bundled ONNX Runtime on-device. No OmniQueue backend, auth, ads, or payments.
6. **Regions** — same app worldwide; catalog is Disneyland Anaheim; English only.
7. **Regulated / protected material** — not a regulated industry; no Disney license (nominative ride names + on-screen disclaimer); ThemeParks.wiki use is allowed under their public API terms with in-app credit.

If you tested extra physical phones, add them as extra bullets under **DEVICES / OS TESTED** in `app_review_notes.txt` and re-check `python3 -c "print(len(open('Companion app/app_review_notes.txt','rb').read()))"` stays ≤ 4000.
