import SwiftUI

struct PlanView: View {
    @Environment(AppSession.self) private var session

    var body: some View {
        List {
            Section {
                if let error = session.errorMessage {
                    Label(error, systemImage: "exclamationmark.circle").foregroundStyle(.secondary)
                }
                if session.isBooting && !session.hasLiveWaits {
                    ProgressView("Loading Disneyland waits")
                } else if !session.hasLiveWaits {
                    Text(session.waitBoardSummary).foregroundStyle(.secondary)
                } else if let recommendation = session.recommendation {
                    nextStop(recommendation)
                } else {
                    Text(session.isPlanning ? "Finding your next stop…" : "No recommendation available")
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Up Next")
            }
            if let route = session.recommendation?.route, session.hasLiveWaits, !route.isEmpty {
                Section("Your Route") {
                    ForEach(route) { stop in
                        NavigationLink {
                            RouteAlternatives(stop: stop)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(stop.slot == 0 ? "Next stop" : "Stop \(stop.slot + 1)")
                                        .font(.caption).foregroundStyle(.secondary)
                                    Text(stop.label)
                                }
                                Spacer()
                                if session.forcedPick?.slot == stop.slot {
                                    Image(systemName: "pin.fill").foregroundStyle(Color.accentColor)
                                }
                                if stop.isRide, let live = session.board.wait(for: stop.actionId) {
                                    Text(WaitFormatting.label(wait: live.waitMin, open: live.open, status: live.status))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            Section("Make It Yours") {
                NavigationLink {
                    RidesView()
                } label: {
                    Label("Ride Preferences", systemImage: "slider.horizontal.3")
                }
                Link("Powered by ThemeParks.wiki", destination: AppLinks.themeParksWiki)
                    .font(.footnote)
            }
        }
        .navigationTitle("Planner")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task { await session.refreshWaits(force: true) }
                } label: {
                    if session.isRefreshingWaits {
                        ProgressView()
                    } else {
                        Label("Refresh live waits", systemImage: "arrow.clockwise")
                    }
                }
                .disabled(session.isRefreshingWaits || session.isBooting)
            }
        }
        .refreshable {
            if !session.isBooting, !session.isRefreshingWaits {
                await session.refreshWaits(force: true)
            }
        }
    }

    private func nextStop(_ recommendation: Recommendation) -> some View {
        let pick = recommendation.recommended
        return VStack(alignment: .leading, spacing: 8) {
            Text(pick.label).font(.title2.bold())
            if pick.isRide, let live = session.board.wait(for: pick.actionId) {
                Text(WaitFormatting.label(wait: live.waitMin, open: live.open, status: live.status))
                    .foregroundStyle(.secondary)
            }
            if session.forcedPick?.slot == 0 {
                Button("Clear Pinned Stop", systemImage: "pin.slash") { session.clearForce() }
            } else {
                DisclosureGroup("Why this pick?") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Model preference \(WaitFormatting.percent(pick.prob))")
                        if session.state.mustDos.indices.contains(pick.actionId),
                           session.state.mustDos[pick.actionId] > 0 {
                            Text("On your must-do list")
                        }
                        Text("\(recommendation.openRides) rides open · \(Int(recommendation.meanWaitMin.rounded())) min average wait")
                    }
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 6)
                }
            }
        }
    }
}

private struct RouteAlternatives: View {
    @Environment(AppSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @State private var showAll = false
    let stop: RouteStop

    private var alternatives: [DistRow] {
        guard let scores = session.recommendation?.distributionsBySlot, scores.indices.contains(stop.slot) else {
            return []
        }
        return showAll ? scores[stop.slot] : Array(scores[stop.slot].filter(\.legal).prefix(6))
    }

    var body: some View {
        List {
            if session.forcedPick?.slot == stop.slot {
                Button("Clear Pinned Stop", systemImage: "pin.slash") {
                    session.clearForce()
                    dismiss()
                }
            }
            Section("Choose an Alternative") {
                ForEach(alternatives) { row in
                    Button {
                        session.setForce(ForcedPick(slot: stop.slot, actionId: row.actionId))
                        dismiss()
                    } label: {
                        LabeledContent(row.label, value: WaitFormatting.percent(row.prob))
                    }
                    .disabled(!row.legal || session.busy)
                }
                Button(showAll ? "Show Top Options" : "Show All Options") { showAll.toggle() }
            }
        }
        .navigationTitle(stop.label)
        .navigationBarTitleDisplayMode(.inline)
    }
}
