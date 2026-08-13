import SwiftUI

struct WaitChip: View {
    var wait: Double?
    var open: Bool
    var status: String
    var small = false

    @Environment(\.themeBlend) private var blend

    var body: some View {
        let tone = WaitTone.of(wait: wait, open: open, status: status)
        Text(WaitFormatting.label(wait: wait, open: open, status: status))
            .font(small ? TicketType.mono : TicketType.caption.weight(.semibold))
            .padding(.horizontal, small ? 7 : 9)
            .padding(.vertical, small ? 3 : 5)
            .foregroundStyle(fg(tone))
            .background(bg(tone), in: Capsule(style: .continuous))
            .overlay(Capsule(style: .continuous).stroke(fg(tone).opacity(0.18), lineWidth: 1))
    }

    private func fg(_ tone: WaitTone) -> Color {
        switch tone {
        case .good: return TicketInk.waitGood(blend: blend)
        case .warn: return TicketInk.mustard
        case .bad: return TicketInk.oxblood
        case .closed: return TicketInk.muted(blend: blend)
        case .unknown: return TicketInk.muted(blend: blend)
        }
    }

    private func bg(_ tone: WaitTone) -> Color {
        fg(tone).opacity(TicketInk.lerp(0.18, 0.12, blend))
    }
}
