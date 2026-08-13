import SwiftUI

struct RidesView: View {
    @Environment(AppSession.self) private var session
    @Environment(\.colorScheme) private var scheme
    @State private var query = ""
    @State private var sort: SortMode = .pref

    enum SortMode: String, CaseIterable, Identifiable {
        case pref, wait, az
        var id: String { rawValue }
        var title: String {
            switch self {
            case .pref: return "My priority"
            case .wait: return "Shortest wait"
            case .az: return "A–Z"
            }
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            searchBar
            sortTabs
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(visibleRides) { ride in
                        RideCard(ride: ride)
                            .rotationEffect(.degrees(wobble(ride.id)))
                    }
                    if visibleRides.isEmpty {
                        Text("No rides match “\(query)”.")
                            .font(TicketType.body)
                            .foregroundStyle(TicketInk.muted(for: scheme))
                            .padding(.top, 24)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
                .animation(.spring(duration: 0.4, bounce: 0.15), value: sort)
            }
            .scrollIndicators(.hidden)
        }
    }

    private var searchBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(TicketInk.muted(for: scheme))
            TextField("Search rides or lands", text: $query)
                .font(TicketType.body)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            TicketStock(corner: 16, stubWidth: 12, punched: false)
        }
        .padding(.horizontal, 16)
        .padding(.top, 4)
    }

    private var sortTabs: some View {
        HStack(spacing: 6) {
            ForEach(SortMode.allCases) { mode in
                Button {
                    withAnimation(.spring(duration: 0.35, bounce: 0.2)) { sort = mode }
                } label: {
                    Text(mode.title)
                        .font(TicketType.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .foregroundStyle(sort == mode ? TicketInk.stock(for: scheme) : TicketInk.ink(for: scheme))
                        .background(
                            Capsule(style: .continuous)
                                .fill(sort == mode ? TicketInk.copperAccent(for: scheme) : TicketInk.stock(for: scheme))
                        )
                }
                .buttonStyle(.plain)
            }
            Spacer()
        }
        .padding(.horizontal, 16)
    }

    private var visibleRides: [ParkCatalog.Ride] {
        var rides = session.catalog.rides
        switch sort {
        case .az:
            rides.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        case .wait:
            rides.sort { a, b in
                let wa = session.board.wait(for: a.id)
                let wb = session.board.wait(for: b.id)
                let va = (wa?.open == true ? wa?.waitMin : nil) ?? .infinity
                let vb = (wb?.open == true ? wb?.waitMin : nil) ?? .infinity
                return va == vb ? a.id < b.id : va < vb
            }
        case .pref:
            rides.sort { a, b in
                let d = (session.state.preferenceWeights[safe: b.id] ?? 0) - (session.state.preferenceWeights[safe: a.id] ?? 0)
                return d == 0 ? a.id < b.id : d > 0
            }
        }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if q.isEmpty { return rides }
        return rides.filter { $0.name.lowercased().contains(q) || $0.hubName.lowercased().contains(q) }
    }

    private func wobble(_ id: Int) -> Double {
        Double((id * 13) % 5 - 2) * 0.18
    }
}

private struct RideCard: View {
    let ride: ParkCatalog.Ride
    @Environment(AppSession.self) private var session
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let live = session.board.wait(for: ride.id)
        let weight = session.state.preferenceWeights[safe: ride.id] ?? 0
        let must = (session.state.mustDos[safe: ride.id] ?? 0) > 0
        let done = session.state.history[safe: ride.id] ?? 0

        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(ride.name)
                            .font(TicketType.body.weight(.semibold))
                            .foregroundStyle(TicketInk.ink(for: scheme))
                        if must {
                            Text("MUST")
                                .font(TicketType.mono)
                                .foregroundStyle(TicketInk.copperAccent(for: scheme))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .overlay(Capsule().stroke(TicketInk.copperAccent(for: scheme), lineWidth: 1))
                        }
                        if done > 0 {
                            Text("DONE ×\(done)")
                                .font(TicketType.mono)
                                .foregroundStyle(TicketInk.teal)
                        }
                    }
                    Text(ride.hubName)
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(for: scheme))
                }
                Spacer()
                WaitChip(wait: live?.waitMin, open: live?.open ?? false, status: live?.status ?? "UNKNOWN")
            }

            PriorityControl(weight: weight) { next in
                var s = session.state
                s.setWeight(next, ride: ride.id)
                session.commit(s)
            }

            HStack {
                Button {
                    var s = session.state
                    s.toggleMustDo(ride: ride.id)
                    session.commit(s)
                } label: {
                    Label(must ? "Must-do on" : "Must-do", systemImage: must ? "star.fill" : "star")
                        .font(TicketType.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .foregroundStyle(must ? TicketInk.stock(for: scheme) : TicketInk.ink(for: scheme))
                        .background(
                            Capsule().fill(must ? TicketInk.copperAccent(for: scheme) : TicketInk.paper(for: scheme))
                        )
                }
                .buttonStyle(.plain)
                .sensoryFeedback(.impact(weight: .light), trigger: must)

                Spacer()

                HStack(spacing: 8) {
                    Text("Rode it")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(for: scheme))
                    stepperButton("minus") {
                        var s = session.state
                        s.bumpDone(ride: ride.id, delta: -1)
                        session.commit(s)
                    }
                    Text("\(done)")
                        .font(TicketType.mono)
                        .frame(minWidth: 16)
                    stepperButton("plus") {
                        var s = session.state
                        s.bumpDone(ride: ride.id, delta: 1)
                        session.commit(s)
                    }
                }
            }
        }
        .padding(14)
        .background { TicketStock(corner: 18, stubWidth: 12) }
        .ticketShadow(scheme)
    }

    private func stepperButton(_ system: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.caption.weight(.bold))
                .foregroundStyle(TicketInk.ink(for: scheme))
                .frame(width: 28, height: 28)
                .background(TicketInk.paper(for: scheme), in: Circle())
        }
        .buttonStyle(.plain)
    }
}

struct PriorityControl: View {
    var weight: Double
    var onChange: (Double) -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let selected = PriorityLevel.from(weight: weight)
        VStack(alignment: .leading, spacing: 6) {
            Text("How much do you want this?")
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.muted(for: scheme))
            HStack(spacing: 4) {
                ForEach(PriorityLevel.allCases) { level in
                    Button {
                        onChange(Double(level.rawValue))
                    } label: {
                        Text(level.title)
                            .font(TicketType.caption.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .foregroundStyle(selected == level ? TicketInk.stock(for: scheme) : TicketInk.ink(for: scheme))
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(selected == level ? TicketInk.copperAccent(for: scheme) : TicketInk.paper(for: scheme).opacity(0.7))
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
