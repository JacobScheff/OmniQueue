import SwiftUI

struct RidesView: View {
    @Environment(AppSession.self) private var session
    @State private var query = ""
    @State private var sort: Sort = .priority

    private enum Sort: String, CaseIterable {
        case priority = "My Priority"
        case wait = "Shortest Wait"
        case name = "Name"
    }

    private var rides: [ParkCatalog.Ride] {
        session.catalog.rides.filter { query.isEmpty || FuzzyMatch.matches(query, in: $0.name, $0.hubName) }
            .sorted { left, right in
                switch sort {
                case .priority:
                    let lhs = session.state.score(for: left.id)
                    let rhs = session.state.score(for: right.id)
                    if lhs != rhs { return lhs > rhs }
                case .wait:
                    let lhs = session.board.wait(for: left.id)?.waitMin ?? .infinity
                    let rhs = session.board.wait(for: right.id)?.waitMin ?? .infinity
                    if lhs != rhs { return lhs < rhs }
                case .name:
                    break
                }
                return left.name.localizedStandardCompare(right.name) == .orderedAscending
            }
    }

    var body: some View {
        List(rides) { ride in
            NavigationLink {
                RidePreferences(ride: ride)
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(ride.name)
                        Text(ride.hubName).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if session.state.mustDos[ride.id] > 0 {
                        Image(systemName: "star.fill").foregroundStyle(Color.accentColor)
                            .accessibilityLabel("Must-do")
                    }
                    if let live = session.board.wait(for: ride.id), live.status != "UNKNOWN" {
                        Text(WaitFormatting.label(wait: live.waitMin, open: live.open, status: live.status))
                            .font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .overlay {
            if rides.isEmpty { ContentUnavailableView.search(text: query) }
        }
        .searchable(text: $query, prompt: "Ride or land")
        .navigationTitle("Ride Preferences")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Picker("Sort by", selection: $sort) {
                        ForEach(Sort.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    Toggle("Advanced Scores", isOn: Binding(
                        get: { session.usesAdvancedPrefs }, set: session.setAdvancedPrefs
                    ))
                } label: {
                    Label("Sort and settings", systemImage: "ellipsis.circle")
                }
            }
        }
    }
}

private struct RidePreferences: View {
    @Environment(AppSession.self) private var session
    @State private var draft: Double?
    let ride: ParkCatalog.Ride

    var body: some View {
        Form {
            Section("Priority") {
                if session.usesAdvancedPrefs {
                    let score = session.state.score(for: ride.id)
                    LabeledContent("Score", value: Int(draft ?? score).formatted())
                    Slider(value: Binding(get: { draft ?? score }, set: { draft = $0 }), in: 0...100, step: 1) { editing in
                        if !editing { commitScore() }
                    }
                    .accessibilityLabel("Priority score")
                    .accessibilityAdjustableAction { direction in
                        switch direction {
                        case .increment: draft = min(100, score + 1)
                        case .decrement: draft = max(0, score - 1)
                        @unknown default: return
                        }
                        commitScore()
                    }
                } else {
                    Picker("Priority", selection: Binding(
                        get: { session.state.level(for: ride.id) },
                        set: { value in change { $0.setLevel(value, ride: ride.id) } }
                    )) {
                        ForEach(PriorityLevel.allCases) { level in
                            Text(level.title).tag(level)
                        }
                    }
                }
                Toggle("Must-do", isOn: Binding(
                    get: { session.state.mustDos[ride.id] > 0 },
                    set: { enabled in
                        if enabled != (session.state.mustDos[ride.id] > 0) { change { $0.toggleMustDo(ride: ride.id) } }
                    }
                ))
            }
            Section("Visit History") {
                let count = session.state.history[ride.id]
                Stepper("Times ridden · \(count)", value: Binding(
                    get: { session.state.history[ride.id] },
                    set: { value in change { $0.bumpDone(ride: ride.id, delta: value - count) } }
                ), in: 0...20)
            }
        }
        .navigationTitle(ride.name)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func change(_ update: (inout GuestState) -> Void) {
        var state = session.state
        update(&state)
        if state != session.state { session.commit(state) }
    }

    private func commitScore() {
        guard let draft else { return }
        change { $0.setScore(draft, ride: ride.id) }
        self.draft = nil
    }
}
