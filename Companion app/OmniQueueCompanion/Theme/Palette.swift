import SwiftUI

/// Copper-ticket palette. Warm stock paper and oxidized metal — not the usual
/// purple-on-black dashboard look.
enum TicketInk {
    static let copper = Color(red: 0.710, green: 0.322, blue: 0.102)
    static let copperDeep = Color(red: 0.478, green: 0.188, blue: 0.055)
    static let teal = Color(red: 0.122, green: 0.373, blue: 0.353)
    static let tealBright = Color(red: 0.494, green: 0.722, blue: 0.698)
    static let oxblood = Color(red: 0.545, green: 0.180, blue: 0.180)
    static let mustard = Color(red: 0.760, green: 0.525, blue: 0.165)

    static func paper(for scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 0.086, green: 0.075, blue: 0.063)
            : Color(red: 0.953, green: 0.902, blue: 0.816)
    }

    static func stock(for scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 0.129, green: 0.110, blue: 0.094)
            : Color(red: 0.980, green: 0.937, blue: 0.863)
    }

    static func ink(for scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 0.953, green: 0.902, blue: 0.816)
            : Color(red: 0.110, green: 0.098, blue: 0.078)
    }

    static func muted(for scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 0.690, green: 0.635, blue: 0.557)
            : Color(red: 0.420, green: 0.365, blue: 0.302)
    }

    static func rule(for scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color.white.opacity(0.10)
            : Color.black.opacity(0.10)
    }

    static func copperAccent(for scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.878, green: 0.478, blue: 0.227) : copper
    }
}

enum TicketType {
    static let mark = Font.system(.largeTitle, design: .serif).weight(.bold)
    static let display = Font.system(.title, design: .serif).weight(.semibold)
    static let headline = Font.system(.title3, design: .rounded).weight(.semibold)
    static let body = Font.system(.body, design: .rounded)
    static let caption = Font.system(.caption, design: .rounded).weight(.medium)
    static let mono = Font.system(.caption, design: .monospaced).weight(.semibold)
}

extension View {
    func ticketShadow(_ scheme: ColorScheme) -> some View {
        shadow(
            color: scheme == .dark ? .black.opacity(0.45) : Color(red: 0.35, green: 0.22, blue: 0.10).opacity(0.18),
            radius: 18,
            y: 8
        )
    }
}
