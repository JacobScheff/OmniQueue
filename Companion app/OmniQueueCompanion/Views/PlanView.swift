import SwiftUI

struct PlanView: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend
    @State private var whyOpen = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let error = session.errorMessage {
                    Text(error)
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.oxblood)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(TicketInk.oxblood.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }

                TicketHero()

                if session.forcedPick?.slot == 0, session.hasLiveWaits {
                    pinBanner
                }

                RouteTimeline()

                DisclaimerBlock(compact: true)
                    .padding(.top, 8)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 28)
            .animation(.spring(duration: 0.5, bounce: 0.18), value: session.recommendation?.recommended.actionId)
        }
        .scrollIndicators(.hidden)
        .refreshable {
            await session.refreshWaits(force: true)
        }
    }

    private var pinBanner: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: "pin.fill")
                .foregroundStyle(TicketInk.copperAccent(blend: blend))
            Text("You pinned the next stop. The rest of the ticket still rewrites around it.")
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.ink(blend: blend))
            Spacer(minLength: 4)
            Button("Clear") {
                withAnimation(.spring(duration: 0.4, bounce: 0.25)) { session.clearForce() }
            }
            .font(TicketType.caption.weight(.semibold))
            .foregroundStyle(TicketInk.copperAccent(blend: blend))
        }
        .padding(12)
        .background(TicketInk.stock(blend: blend), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(TicketInk.rule(blend: blend), lineWidth: 1))
    }
}

private struct TicketHero: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend
    @State private var whyOpen = false

    var body: some View {
        let rec = session.hasLiveWaits ? session.recommendation : nil
        let live = rec.flatMap { $0.recommended.isRide ? session.board.wait(for: $0.recommended.actionId) : nil }

        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(waitingOnBoard ? "WAIT BOARD" : (session.forcedPick?.slot == 0 ? "PINNED NEXT" : "UP NEXT"))
                    .font(TicketType.mono)
                    .tracking(1.4)
                    .foregroundStyle(TicketInk.copperAccent(blend: blend))
                GuestCopyStamp()
                    .scaleEffect(0.78)
                    .offset(y: -3)
                Spacer()
                if rec?.recommended.isRide == true {
                    WaitChip(wait: live?.waitMin, open: live?.open ?? false, status: live?.status ?? "UNKNOWN")
                }
            }
            .padding(.leading, TicketLayout.leading(16))
            .padding(.trailing, 16)
            .padding(.top, 16)

            PerforationDivider()
                .padding(.leading, TicketLayout.leading(16))
                .padding(.trailing, 16)

            Text(heroTitle)
                .font(TicketType.mark)
                .foregroundStyle(TicketInk.ink(blend: blend))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.leading, TicketLayout.leading(16))
                .padding(.trailing, 16)
                .padding(.top, 6)
                .contentTransition(.opacity)
                .id(waitingOnBoard ? "no-board" : (rec.map { String($0.recommended.actionId) } ?? "empty"))

            if waitingOnBoard {
                Text(session.waitBoardSummary)
                    .font(TicketType.body)
                    .foregroundStyle(TicketInk.muted(blend: blend))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, TicketLayout.leading(16))
                    .padding(.trailing, 16)
                    .padding(.top, 4)
            } else if let rec {
                Text("\(confidence(rec.recommended.prob)) · \(WaitFormatting.percent(rec.recommended.prob))")
                    .font(TicketType.body)
                    .foregroundStyle(TicketInk.muted(blend: blend))
                    .padding(.leading, TicketLayout.leading(16))
                    .padding(.trailing, 16)
                    .padding(.top, 4)
            }

            VStack(alignment: .leading, spacing: 8) {
                if waitingOnBoard {
                    HStack(spacing: 8) {
                        InkingDots()
                        Text(session.isRefreshingWaits ? "Trying the wait feed…" : "No wait times to plan from.")
                            .font(TicketType.caption)
                            .foregroundStyle(TicketInk.muted(blend: blend))
                    }
                } else {
                    Button {
                        withAnimation(.spring(duration: 0.35, bounce: 0.2)) { whyOpen.toggle() }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "info.circle")
                            Text("Why this pick?")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .rotationEffect(.degrees(whyOpen ? 90 : 0))
                        }
                        .font(TicketType.caption.weight(.semibold))
                        .foregroundStyle(TicketInk.ink(blend: blend))
                    }
                    .buttonStyle(.plain)

                    if whyOpen {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(whyLines, id: \.self) { line in
                                HStack(alignment: .top, spacing: 8) {
                                    Circle()
                                        .fill(TicketInk.copperAccent(blend: blend))
                                        .frame(width: 5, height: 5)
                                        .padding(.top, 5)
                                    Text(line)
                                        .font(TicketType.caption)
                                        .foregroundStyle(TicketInk.muted(blend: blend))
                                }
                            }
                        }
                        .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
            }
            .padding(.leading, TicketLayout.leading(16))
            .padding(.trailing, 16)
            .padding(.vertical, 12)

            HStack {
                statusPills
                Spacer()
                HStack(spacing: 6) {
                    SpinningRefreshIcon(spinning: session.isRefreshingWaits)
                    Text("Refresh waits")
                }
                .font(TicketType.caption.weight(.semibold))
                .foregroundStyle(TicketInk.copperAccent(blend: blend))
                .opacity(session.isRefreshingWaits ? 0.85 : 1)
                .contentShape(Rectangle())
                .onTapGesture {
                    guard !session.isRefreshingWaits else { return }
                    Task { await session.refreshWaits(force: true) }
                }
                .accessibilityAddTraits(.isButton)
                .accessibilityLabel("Refresh waits")
            }
            .padding(.leading, TicketLayout.leading(16))
            .padding(.trailing, 16)
            .padding(.bottom, 16)
        }
        .background {
            TicketStock(corner: 24, stubWidth: 16)
        }
        .ticketShadow(blend)
        .accessibilityElement(children: .combine)
    }

    private var waitingOnBoard: Bool { !session.hasLiveWaits }

    private var heroTitle: String {
        if waitingOnBoard { return "No wait times" }
        if let rec = session.recommendation { return rec.recommended.label }
        if session.isPlanning || session.isRefreshingWaits { return "Still inking…" }
        return "—"
    }

    private var statusPills: some View {
        HStack(spacing: 6) {
            if let rec = session.recommendation, session.hasLiveWaits {
                Text("\(rec.openRides) open")
                Text("·")
                Text("avg \(Int(rec.meanWaitMin.rounded())) min")
            } else if session.isRefreshingWaits {
                Text("Updating")
            } else if !session.hasLiveWaits {
                Text("No board")
            } else {
                Text("Waiting on the board")
            }
        }
        .font(TicketType.caption)
        .foregroundStyle(TicketInk.muted(blend: blend))
    }

    private var whyLines: [String] {
        guard let rec = session.recommendation else { return [] }
        if session.forcedPick?.slot == 0 {
            return ["You pinned this stop yourself — the rest of the plan re-routes around it."]
        }
        var reasons: [String] = []
        let recIsRide = rec.recommended.isRide
        if !recIsRide {
            reasons.append(
                rec.recommended.label.lowercased().contains("exit")
                    ? "The model thinks it’s time to head out given how much daylight is left."
                    : "The model thinks waiting a moment beats moving right now."
            )
            return reasons
        }
        if let live = session.board.wait(for: rec.recommended.actionId) {
            reasons.append("Current wait is \(WaitFormatting.label(wait: live.waitMin, open: live.open, status: live.status)).")
        }
        if rec.recommended.actionId < session.state.mustDos.count, session.state.mustDos[rec.recommended.actionId] > 0 {
            reasons.append("It’s on your must-do list.")
        }
        let top = session.catalog.rides
            .filter { session.state.preferenceWeights.indices.contains($0.id) && session.state.preferenceWeights[$0.id] > 0 }
            .sorted { session.state.preferenceWeights[$0.id] > session.state.preferenceWeights[$1.id] }
            .prefix(3)
            .map(\.id)
        if top.contains(rec.recommended.actionId) {
            reasons.append("It’s one of your top preferences.")
        }
        if rec.naturalRecommended.actionId != rec.recommended.actionId {
            reasons.append("Runner-up was \(rec.naturalRecommended.label).")
        }
        if reasons.isEmpty {
            reasons.append("Best balance of wait time and your preferences right now.")
        }
        return reasons
    }

    private func confidence(_ p: Double) -> String {
        if p >= 0.66 { return "Confident pick" }
        if p >= 0.4 { return "Leaning this way" }
        return "Close call"
    }
}
