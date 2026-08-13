import Foundation

/// Public URLs for App Store Connect and in-app About links.
enum AppLinks {
    static let marketing = URL(string: "https://www.jacobscheff.com/OmniQueue")!
    static let privacy = URL(string: "https://www.jacobscheff.com/OmniQueue#privacy")!
    static let support = URL(string: "https://www.jacobscheff.com/OmniQueue#support")!
    static let themeParksWiki = URL(string: "https://themeparks.wiki")!
    static let onnxRuntimeLicense = URL(string: "https://github.com/microsoft/onnxruntime/blob/main/LICENSE")!

    static var marketingVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    }

    static var buildNumber: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
    }

    static var versionLabel: String {
        "Version \(marketingVersion) (\(buildNumber))"
    }
}
