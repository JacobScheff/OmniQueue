import SwiftUI

struct RouteTimeline: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend
    @State private var expanded: Int?
    @State private var showAll = false

    private let stopWords = ["Now", "Next", "Then", "Then", "Then"]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Your route")
                    .font(TicketType.headline)
                    .foregroundStyle(TicketInk.ink(blend: blend))
                Spacer()
                Text("Tap a stop to pin another")
                    .font(TicketType.caption)
                    .foregroundStyle(TicketInk.muted(blend: blend))
            }

            if let route = session.recommendation?.route, !route.isEmpty {
                VStack(spacing: 0) {
                    ForEach(Array(route.enumerated()), id: \.element.id) { index, stop in
                        stopRow(stop, index: index)
                            .opacity(1)
                            .offset(y: 0)
                            .animation(.spring(duration: 0.5, bounce: 0.22).delay(Double(index) * 0.05), value: stop.actionId)
                    }
                }
                .background {
                    TicketStock(corner: 20, stubWidth: 14)
                }
                .ticketShadow(blend)
            } else {
                Text(session.busy ? "Building your plan…" : "No plan yet.")
                    .font(TicketType.body)
                    .foregroundStyle(TicketInk.muted(blend: blend))
                    .padding(.vertical, 18)
            }
        }
    }

    @ViewBuilder
    private func stopRow(_ stop: RouteStop, index: Int) -> some View {
        let live = stop.isRide ? session.board.wait(for: stop.actionId) : nil
        let isOpen = expanded == stop.slot
        let isForced = session.forcedPick?.slot == stop.slot

        VStack(spacing: 0) {
            Button {
                withAnimation(.spring(duration: 0.38, bounce: 0.18)) {
                    expanded = isOpen ? nil : stop.slot
                    showAll = false
                }
            } label: {
                HStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(dotColor(stop: stop, live: live))
                            .frame(width: 12, height: 12)
                        if isForced {
                            Image(systemName: "pin.fill")
                                .font(.system(size: 8))
                                .foregroundStyle(TicketInk.copperAccent(blend: blend))
                                .offset(x: 8, y: -8)
                                .transition(.scale.combined(with: .opacity))
                        }
                    }
                    .frame(width: 20)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(stopWords.indices.contains(stop.slot) ? stopWords[stop.slot] : "Then")
                            .font(TicketType.mono)
                            .foregroundStyle(TicketInk.muted(blend: blend))
                        Text(stop.label)
                            .font(TicketType.body.weight(.semibold))
                            .foregroundStyle(TicketInk.ink(blend: blend))
                            .multilineTextAlignment(.leading)
                    }
                    Spacer()
                    if stop.isRide {
                        WaitChip(wait: live?.waitMin, open: live?.open ?? false, status: live?.status ?? "UNKNOWN", small: true)
                    }
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .rotationEffect(.degrees(isOpen ? 90 : 0))
                        .foregroundStyle(TicketInk.muted(blend: blend))
                }
                .padding(.leading, TicketLayout.leading(14))
                .padding(.trailing, 16)
                .padding(.vertical, 12)
                .background(stop.slot == 0 ? TicketInk.copperAccent(blend: blend).opacity(0.08) : Color.clear)
            }
            .buttonStyle(.plain)

            if isOpen {
                alternatives(for: stop)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            if index < (session.recommendation?.route.count ?? 1) - 1 {
                PerforationDivider()
                    .padding(.leading, TicketLayout.leading(14))
                    .padding(.trailing, 12)
            }
        }
    }

    @ViewBuilder
    private func alternatives(for stop: RouteStop) -> some View {
        let dist = session.recommendation?.distributionsBySlot.indices.contains(stop.slot) == true
            ? session.recommendation!.distributionsBySlot[stop.slot]
            : []
        let visible = showAll ? dist : Array(dist.filter(\.legal).prefix(6))

        VStack(alignment: .leading, spacing: 8) {
            Text("Tap another option to pin it here — the rest of the plan rewrites around it.")
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.muted(blend: blend))

            ForEach(visible) { row in
                let active = session.forcedPick?.slot == stop.slot && session.forcedPick?.actionId == row.actionId
                Button {
                    guard row.legal, !session.busy else { return }
                    withAnimation(.spring(duration: 0.42, bounce: 0.3)) {
                        session.setForce(ForcedPick(slot: stop.slot, actionId: row.actionId))
                    }
                } label: {
                    HStack {
                        Text(row.label)
                            .font(TicketType.caption.weight(.semibold))
                            .foregroundStyle(row.legal ? TicketInk.ink(blend: blend) : TicketInk.muted(blend: blend))
                            .strikethrough(!row.legal, color: TicketInk.muted(blend: blend))
                        Spacer()
                        if row.isRide {
                            WaitChip(wait: row.waitMin, open: row.open, status: row.status, small: true)
                        }
                        Text(WaitFormatting.percent(row.prob))
                            .font(TicketType.mono)
                            .foregroundStyle(TicketInk.muted(blend: blend))
                        if active {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(TicketInk.copperAccent(blend: blend))
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(active ? TicketInk.copperAccent(blend: blend).opacity(0.14) : TicketInk.paper(blend: blend).opacity(0.5))
                    )
                }
                .buttonStyle(.plain)
                .disabled(!row.legal || session.busy)
                .sensoryFeedback(.impact(weight: .light), trigger: active)
            }

            if dist.count > 6 {
                Button(showAll ? "Show top options" : "Show all options") {
                    withAnimation { showAll.toggle() }
                }
                .font(TicketType.caption.weight(.semibold))
                .foregroundStyle(TicketInk.copperAccent(blend: blend))
            }
        }
        .padding(.leading, TicketLayout.leading(14))
        .padding(.trailing, 16)
        .padding(.bottom, 12)
    }

    private func dotColor(stop: RouteStop, live: RideWait?) -> Color {
        guard stop.isRide else { return TicketInk.muted(blend: blend) }
        switch WaitTone.of(wait: live?.waitMin, open: live?.open ?? false, status: live?.status ?? "UNKNOWN") {
        case .good: return TicketInk.waitGood(blend: blend)
        case .warn: return TicketInk.mustard
        case .bad: return TicketInk.oxblood
        case .closed: return TicketInk.muted(blend: blend)
        case .unknown: return TicketInk.copper
        }
    }
}
