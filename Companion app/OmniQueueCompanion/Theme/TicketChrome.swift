import SwiftUI

enum TicketLayout {
    /// Extra left padding so copy sits past the copper stub and punch holes.
    static func leading(_ stubWidth: CGFloat) -> CGFloat { stubWidth + 18 }
}

/// Perforated ticket edge + faint fiber grain. Drawn, not generated art.
struct TicketStock: View {
    var corner: CGFloat = 22
    var stubWidth: CGFloat = 18
    var punched: Bool = true

    @Environment(\.themeBlend) private var blend

    var body: some View {
        Canvas { context, size in
            let paper = TicketInk.stock(blend: blend)
            let stub = TicketInk.copperAccent(blend: blend)
            let hole = TicketInk.paper(blend: blend)

            let body = Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: corner)
            context.fill(body, with: .color(paper))
            context.clip(to: body)

            // Left stub — clipped so it follows the rounded ticket, not a square strip.
            let stubRect = Path(CGRect(x: 0, y: 0, width: stubWidth, height: size.height))
            context.fill(stubRect, with: .color(stub.opacity(TicketInk.lerp(0.85, 1, blend))))

            // Fiber grain — hashed, not random-looking noise
            context.opacity = TicketInk.lerp(0.04, 0.07, blend)
            for i in stride(from: 0, through: Int(size.width + size.height), by: 11) {
                var line = Path()
                line.move(to: CGPoint(x: CGFloat(i), y: 0))
                line.addLine(to: CGPoint(x: CGFloat(i) - size.height, y: size.height))
                context.stroke(line, with: .color(TicketInk.ink(blend: blend)), lineWidth: 0.6)
            }
            context.opacity = 1

            if punched {
                let holeSize: CGFloat = 9
                let holeX = (stubWidth - holeSize) / 2
                let count = max(3, Int((size.height - 28) / 22))
                let spacing = (size.height - 24) / CGFloat(count)
                for i in 0..<count {
                    let y = 12 + spacing * CGFloat(i) + spacing * 0.28
                    let holeRect = CGRect(x: holeX, y: y, width: holeSize, height: holeSize)
                    context.fill(Path(ellipseIn: holeRect), with: .color(hole))
                }
            }

            context.stroke(
                Path(roundedRect: CGRect(origin: .zero, size: size).insetBy(dx: 0.6, dy: 0.6), cornerRadius: corner),
                with: .color(TicketInk.rule(blend: blend)),
                lineWidth: 1
            )
        }
        .clipShape(RoundedRectangle(cornerRadius: corner, style: .continuous))
        .allowsHitTesting(false)
    }
}

struct PerforationDivider: View {
    @Environment(\.themeBlend) private var blend

    var body: some View {
        Canvas { context, size in
            let n = Int(size.width / 10)
            let spacing = size.width / CGFloat(max(n, 1))
            for i in 0..<n {
                let x = spacing * CGFloat(i) + 3
                let r = CGRect(x: x, y: size.height / 2 - 1.4, width: 2.8, height: 2.8)
                context.fill(Path(ellipseIn: r), with: .color(TicketInk.muted(blend: blend).opacity(0.55)))
            }
        }
        .frame(height: 10)
        .allowsHitTesting(false)
    }
}

struct GuestCopyStamp: View {
    @Environment(\.themeBlend) private var blend

    var body: some View {
        Text("GUEST COPY")
            .font(TicketType.mono)
            .tracking(1.6)
            .foregroundStyle(TicketInk.copperAccent(blend: blend))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .overlay(
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .stroke(TicketInk.copperAccent(blend: blend).opacity(0.7), lineWidth: 1.2)
            )
            .rotationEffect(.degrees(-8))
            .accessibilityHidden(true)
    }
}

struct PaperclipMark: View {
    @Environment(\.themeBlend) private var blend

    var body: some View {
        Canvas { context, size in
            let c = TicketInk.copperAccent(blend: blend)
            var path = Path()
            path.addRoundedRect(in: CGRect(x: 6, y: 2, width: 10, height: size.height - 4), cornerSize: CGSize(width: 5, height: 5))
            path.addRoundedRect(in: CGRect(x: 10, y: 8, width: 10, height: size.height - 14), cornerSize: CGSize(width: 5, height: 5))
            context.stroke(path, with: .color(c), lineWidth: 2.2)
        }
        .frame(width: 28, height: 36)
        .accessibilityHidden(true)
    }
}

struct TicketSerial: View {
    var body: some View {
        Text(Self.today)
            .font(TicketType.mono)
            .tracking(0.8)
    }

    private static var today: String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "America/Los_Angeles")
        f.dateFormat = "dd MMM yy"
        return f.string(from: Date()).uppercased()
    }
}

struct InkingDots: View {
    @Environment(\.themeBlend) private var blend
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 7) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(TicketInk.copperAccent(blend: blend))
                    .frame(width: 7, height: 7)
                    .opacity(phase == i ? 1 : 0.22)
                    .scaleEffect(phase == i ? 1.18 : 0.82)
            }
        }
        .accessibilityHidden(true)
        .task {
            while !Task.isCancelled {
                withAnimation(.easeInOut(duration: 0.22)) {
                    phase = (phase + 1) % 3
                }
                try? await Task.sleep(for: .milliseconds(260))
            }
        }
    }
}

struct SpinningRefreshIcon: View {
    var spinning: Bool

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: !spinning)) { timeline in
            let turns = spinning
                ? timeline.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 0.75) / 0.75
                : 0
            Image(systemName: "arrow.clockwise")
                .rotationEffect(.degrees(turns * 360))
        }
    }
}
