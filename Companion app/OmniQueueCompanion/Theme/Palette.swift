import SwiftUI

/// Copper-ticket palette. Warm stock paper and oxidized metal — not the usual
/// purple-on-black dashboard look.
///
/// `blend` is 0 for dark and 1 for light. Animate it for a gradual theme shift.
enum TicketInk {
    static let copper = Color(red: 0.710, green: 0.322, blue: 0.102)
    static let copperDeep = Color(red: 0.478, green: 0.188, blue: 0.055)
    static let teal = Color(red: 0.16, green: 0.41, blue: 0.26)
    static let tealBright = Color(red: 0.50, green: 0.78, blue: 0.55)
    static let oxblood = Color(red: 0.545, green: 0.180, blue: 0.180)
    static let mustard = Color(red: 0.760, green: 0.525, blue: 0.165)

    /// Low waits: sage, not cyan. `blend` 0 = dark, 1 = light.
    static func waitGood(blend t: CGFloat) -> Color {
        mix((0.50, 0.78, 0.55, 1), (0.16, 0.41, 0.26, 1), t)
    }

    static func paper(blend t: CGFloat) -> Color {
        mix((0.086, 0.075, 0.063, 1), (0.953, 0.902, 0.816, 1), t)
    }

    static func stock(blend t: CGFloat) -> Color {
        mix((0.129, 0.110, 0.094, 1), (0.980, 0.937, 0.863, 1), t)
    }

    static func ink(blend t: CGFloat) -> Color {
        mix((0.953, 0.902, 0.816, 1), (0.110, 0.098, 0.078, 1), t)
    }

    static func muted(blend t: CGFloat) -> Color {
        mix((0.690, 0.635, 0.557, 1), (0.420, 0.365, 0.302, 1), t)
    }

    static func rule(blend t: CGFloat) -> Color {
        mix((1, 1, 1, 0.10), (0, 0, 0, 0.10), t)
    }

    static func copperAccent(blend t: CGFloat) -> Color {
        mix((0.878, 0.478, 0.227, 1), (0.710, 0.322, 0.102, 1), t)
    }

    static func paper(for scheme: ColorScheme) -> Color {
        paper(blend: scheme == .light ? 1 : 0)
    }

    /// Scalar mix: `t` 0 = dark value, 1 = light value.
    static func lerp(_ dark: CGFloat, _ light: CGFloat, _ t: CGFloat) -> CGFloat {
        let t = min(max(t, 0), 1)
        return dark + (light - dark) * t
    }

    static func mix(
        _ dark: (CGFloat, CGFloat, CGFloat, CGFloat),
        _ light: (CGFloat, CGFloat, CGFloat, CGFloat),
        _ t: CGFloat
    ) -> Color {
        let t = min(max(t, 0), 1)
        return Color(
            red: dark.0 + (light.0 - dark.0) * t,
            green: dark.1 + (light.1 - dark.1) * t,
            blue: dark.2 + (light.2 - dark.2) * t,
            opacity: dark.3 + (light.3 - dark.3) * t
        )
    }
}

private struct ThemeBlendKey: EnvironmentKey {
    static let defaultValue: CGFloat = 0
}

extension EnvironmentValues {
    /// 0 = dark ticket, 1 = light ticket. Interpolated during theme changes.
    var themeBlend: CGFloat {
        get { self[ThemeBlendKey.self] }
        set { self[ThemeBlendKey.self] = newValue }
    }
}

/// Interpolates `blend` inside the current animation so Canvas and colors ease, not snap.
struct ThemeBlendHost<Content: View>: View, Animatable {
    var blend: CGFloat
    var content: Content

    init(blend: CGFloat, @ViewBuilder content: () -> Content) {
        self.blend = blend
        self.content = content()
    }

    var animatableData: CGFloat {
        get { blend }
        set { blend = newValue }
    }

    var body: some View {
        content.environment(\.themeBlend, blend)
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
    func ticketShadow(_ blend: CGFloat) -> some View {
        let t = min(max(blend, 0), 1)
        return shadow(
            color: Color(
                red: 0.35 * t,
                green: 0.22 * t,
                blue: 0.10 * t,
                opacity: 0.45 - 0.27 * t
            ),
            radius: 18,
            y: 8
        )
    }
}
