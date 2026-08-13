#!/bin/sh
# Apple rejects archives when onnxruntime.framework's Info.plist has an empty or
# missing MinimumOSVersion (ITMS-90208 / ITMS-90530). The ONNX Runtime SPM
# XCFramework still ships that key blank as of 1.24.2, so stamp it after the
# framework is copied into the .app.
set -e

MIN_OS="${IPHONEOS_DEPLOYMENT_TARGET:-17.0}"
FOUND=0

patch_plist() {
  plist="$1"
  if [ ! -f "$plist" ]; then
    return 0
  fi
  echo "note: setting MinimumOSVersion=${MIN_OS} in ${plist}"
  /usr/libexec/PlistBuddy -c "Delete :MinimumOSVersion" "$plist" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Add :MinimumOSVersion string ${MIN_OS}" "$plist"
  echo "note: MinimumOSVersion is now $(/usr/libexec/PlistBuddy -c 'Print :MinimumOSVersion' "$plist")"
  FOUND=1
}

patch_app_frameworks() {
  app_dir="$1"
  [ -n "$app_dir" ] && [ -d "$app_dir" ] || return 0
  for name in onnxruntime onnxruntime_extensions; do
    patch_plist "${app_dir}/Frameworks/${name}.framework/Info.plist"
  done
}

if [ -n "${CODESIGNING_FOLDER_PATH}" ]; then
  patch_app_frameworks "${CODESIGNING_FOLDER_PATH}"
fi

if [ -n "${TARGET_BUILD_DIR}" ] && [ -n "${FRAMEWORKS_FOLDER_PATH}" ]; then
  for name in onnxruntime onnxruntime_extensions; do
    patch_plist "${TARGET_BUILD_DIR}/${FRAMEWORKS_FOLDER_PATH}/${name}.framework/Info.plist"
  done
fi

# Last-resort search so Archive still patches if Xcode used a different copy path.
# Avoid `find | while` so FOUND stays in this shell.
for root in "${TARGET_BUILD_DIR}" "${BUILT_PRODUCTS_DIR}"; do
  [ -n "$root" ] && [ -d "$root" ] || continue
  for plist in \
      "$root"/*.app/Frameworks/onnxruntime.framework/Info.plist \
      "$root"/*.app/Frameworks/onnxruntime_extensions.framework/Info.plist \
      "$root"/*/*.app/Frameworks/onnxruntime.framework/Info.plist \
      "$root"/*/*.app/Frameworks/onnxruntime_extensions.framework/Info.plist
  do
    patch_plist "$plist"
  done
done

if [ "$FOUND" -eq 0 ]; then
  echo "warning: onnxruntime.framework Info.plist was not in the app bundle yet; MinimumOSVersion was not patched" >&2
fi
