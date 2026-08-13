import SwiftUI

struct DisclaimerBlock: View {
    var compact = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 12) {
            Text("Not a Disney product")
                .font(TicketType.caption.weight(.bold))
                .foregroundStyle(TicketInk.ink(for: scheme))
            Text("OmniQueue is an independent planner. It is not affiliated with, authorized by, sponsored by, or endorsed by The Walt Disney Company or any of its affiliates. Attraction names appear only to identify rides.")
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.muted(for: scheme))
                .fixedSize(horizontal: false, vertical: true)

            Link(destination: URL(string: "https://themeparks.wiki")!) {
                HStack(spacing: 6) {
                    Image(systemName: "link")
                    Text("Wait times powered by ThemeParks.wiki")
                }
                .font(TicketType.caption.weight(.semibold))
                .foregroundStyle(TicketInk.teal)
            }
            Text("Those waits are unofficial estimates and can be wrong, stale, or missing. This is not official park data.")
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.muted(for: scheme))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(compact ? 12 : 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(TicketInk.stock(for: scheme), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(TicketInk.rule(for: scheme), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        )
    }
}

struct DisclaimerSheet: View {
    @Environment(AppSession.self) private var session
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Before you ride")
                        .font(TicketType.display)
                    Text("This is a guest-made day planner. It runs a routing model on your phone and pulls live waits from a third-party feed.")
                        .font(TicketType.body)
                        .foregroundStyle(TicketInk.muted(for: scheme))
                    DisclaimerBlock()
                    Button {
                        session.acknowledgeDisclaimer()
                    } label: {
                        Text("Got it — plan my day")
                            .font(TicketType.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .foregroundStyle(TicketInk.stock(for: scheme))
                            .background(TicketInk.copperAccent(for: scheme), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 8)
                }
                .padding(20)
            }
            .background(TicketInk.paper(for: scheme).ignoresSafeArea())
        }
    }
}
