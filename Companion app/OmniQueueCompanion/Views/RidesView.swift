import SwiftUI

struct RidesView: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend
    @State private var query = ""
    @State private var sort: SortMode = .pref
    @Namespace private var prefMorph

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
            PrefModeSwitch()
                .padding(.horizontal, 16)
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(visibleRides) { ride in
                        RideCard(ride: ride, prefMorph: prefMorph)
                            .rotationEffect(.degrees(wobble(ride.id)))
                    }
                    if visibleRides.isEmpty {
                        Text("No rides match “\(query)”.")
                            .font(TicketType.body)
                            .foregroundStyle(TicketInk.muted(blend: blend))
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
                .foregroundStyle(TicketInk.muted(blend: blend))
            TextField("Search rides or lands", text: $query)
                .font(TicketType.body)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        }
        .padding(.leading, TicketLayout.leading(12))
        .padding(.trailing, 14)
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
                        .foregroundStyle(sort == mode ? TicketInk.stock(blend: blend) : TicketInk.ink(blend: blend))
                        .background(
                            Capsule(style: .continuous)
                                .fill(sort == mode ? TicketInk.copperAccent(blend: blend) : TicketInk.stock(blend: blend))
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
                let wa = session.state.preferenceWeights[safe: a.id] ?? 0
                let wb = session.state.preferenceWeights[safe: b.id] ?? 0
                if wa == wb { return a.id < b.id }
                return wa > wb
            }
        }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if q.isEmpty { return rides }
        return rides.filter { FuzzyMatch.matches(q, in: $0.name, $0.hubName) }
    }

    private func wobble(_ id: Int) -> Double {
        Double((id * 13) % 5 - 2) * 0.18
    }
}

private struct RideCard: View {
    let ride: ParkCatalog.Ride
    var prefMorph: Namespace.ID
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend

    var body: some View {
        let live = session.board.wait(for: ride.id)
        let must = (session.state.mustDos[safe: ride.id] ?? 0) > 0
        let done = session.state.history[safe: ride.id] ?? 0

        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(ride.name)
                            .font(TicketType.body.weight(.semibold))
                            .foregroundStyle(TicketInk.ink(blend: blend))
                        if must {
                            Text("MUST")
                                .font(TicketType.mono)
                                .foregroundStyle(TicketInk.copperAccent(blend: blend))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .overlay(Capsule().stroke(TicketInk.copperAccent(blend: blend), lineWidth: 1))
                        }
                        if done > 0 {
                            Text("DONE ×\(done)")
                                .font(TicketType.mono)
                                .foregroundStyle(TicketInk.teal)
                        }
                    }
                    Text(ride.hubName)
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                }
                Spacer()
                WaitChip(wait: live?.waitMin, open: live?.open ?? false, status: live?.status ?? "UNKNOWN")
            }

            PriorityControl(rideId: ride.id, namespace: prefMorph)

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
                        .foregroundStyle(must ? TicketInk.stock(blend: blend) : TicketInk.ink(blend: blend))
                        .background(
                            Capsule().fill(must ? TicketInk.copperAccent(blend: blend) : TicketInk.paper(blend: blend))
                        )
                }
                .buttonStyle(.plain)
                .sensoryFeedback(.impact(weight: .light), trigger: must)

                Spacer()

                HStack(spacing: 8) {
                    Text("Rode it")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
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
        .padding(.leading, TicketLayout.leading(12))
        .padding(.trailing, 14)
        .padding(.vertical, 14)
        .background { TicketStock(corner: 18, stubWidth: 12) }
        .ticketShadow(blend)
        .animation(.spring(duration: 0.55, bounce: 0.22), value: session.usesAdvancedPrefs)
    }

    private func stepperButton(_ system: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.caption.weight(.bold))
                .foregroundStyle(TicketInk.ink(blend: blend))
                .frame(width: 28, height: 28)
                .background(TicketInk.paper(blend: blend), in: Circle())
        }
        .buttonStyle(.plain)
    }
}

struct PrefModeSwitch: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend

    var body: some View {
        GeometryReader { geo in
            let slot = geo.size.width / 2
            let pillW = slot - 8
            let pillH = geo.size.height - 8
            let pillX = 4 + (session.usesAdvancedPrefs ? slot : 0)

            ZStack(alignment: .topLeading) {
                TicketStock(corner: 14, stubWidth: 10, punched: false)
                Capsule(style: .continuous)
                    .fill(TicketInk.copperAccent(blend: blend).opacity(TicketInk.lerp(0.32, 0.22, blend)))
                    .overlay {
                        Capsule(style: .continuous)
                            .strokeBorder(Color.white.opacity(TicketInk.lerp(0.22, 0.45, blend)), lineWidth: 1)
                    }
                    .frame(width: pillW, height: pillH)
                    .offset(x: pillX, y: (geo.size.height - pillH) / 2)

                HStack(spacing: 0) {
                    modeButton("Basic", icon: "square.grid.2x2", advanced: false, width: slot)
                    modeButton("Advanced", icon: "slider.horizontal.3", advanced: true, width: slot)
                }
            }
        }
        .frame(height: 42)
        .sensoryFeedback(.selection, trigger: session.usesAdvancedPrefs)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Preference scale")
        .accessibilityValue(session.usesAdvancedPrefs ? "Advanced" : "Basic")
        .accessibilityAdjustableAction { direction in
            session.setAdvancedPrefs(direction == .increment)
        }
    }

    private func modeButton(_ title: String, icon: String, advanced: Bool, width: CGFloat) -> some View {
        Button {
            session.setAdvancedPrefs(advanced)
        } label: {
            Label(title, systemImage: icon)
                .font(TicketType.caption.weight(.semibold))
                .foregroundStyle(
                    session.usesAdvancedPrefs == advanced
                        ? TicketInk.copperAccent(blend: blend)
                        : TicketInk.muted(blend: blend)
                )
                .frame(width: width, height: 42)
        }
        .buttonStyle(.plain)
    }
}

struct PriorityControl: View {
    var rideId: Int
    var namespace: Namespace.ID
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend
    @State private var draft: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("How much do you want this?")
                    .font(TicketType.caption)
                    .foregroundStyle(TicketInk.muted(blend: blend))
                Spacer()
                if session.usesAdvancedPrefs {
                    Text("\(Int(shownScore.rounded()))")
                        .font(TicketType.mono)
                        .foregroundStyle(TicketInk.copperAccent(blend: blend))
                        .contentTransition(.numericText())
                }
            }
            ZStack {
                if session.usesAdvancedPrefs {
                    advancedSlider
                        .transition(.asymmetric(
                            insertion: .scale(scale: 0.92, anchor: .top).combined(with: .opacity),
                            removal: .scale(scale: 0.92, anchor: .bottom).combined(with: .opacity)
                        ))
                } else {
                    basicButtons
                        .transition(.asymmetric(
                            insertion: .scale(scale: 0.92, anchor: .bottom).combined(with: .opacity),
                            removal: .scale(scale: 0.92, anchor: .top).combined(with: .opacity)
                        ))
                }
            }
            .frame(minHeight: 36)
        }
        .onChange(of: session.usesAdvancedPrefs) { _, _ in
            draft = nil
        }
    }

    private var shownScore: Double {
        draft ?? session.state.score(for: rideId)
    }

    private var selected: PriorityLevel {
        session.state.level(for: rideId)
    }

    private var basicButtons: some View {
        HStack(spacing: 4) {
            ForEach(PriorityLevel.allCases) { level in
                Button {
                    var s = session.state
                    s.setLevel(level, ride: rideId)
                    session.commit(s)
                } label: {
                    Text(level.title)
                        .font(TicketType.caption.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .foregroundStyle(selected == level ? TicketInk.stock(blend: blend) : TicketInk.ink(blend: blend))
                        .background {
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(selected == level ? TicketInk.copperAccent(blend: blend) : TicketInk.paper(blend: blend).opacity(0.7))
                        }
                }
                .buttonStyle(.plain)
                .modifier(PrefPillMatch(active: selected == level, id: "pref-pill-\(rideId)", ns: namespace))
            }
        }
    }

    private var advancedSlider: some View {
        VStack(spacing: 0) {
            Slider(
                value: Binding(
                    get: { shownScore },
                    set: { draft = $0 }
                ),
                in: 0...100,
                step: 1
            ) { editing in
                if !editing, let draft {
                    var s = session.state
                    s.setScore(draft, ride: rideId)
                    session.commit(s)
                    self.draft = nil
                }
            }
            .tint(TicketInk.copperAccent(blend: blend))
            .matchedGeometryEffect(id: "pref-pill-\(rideId)", in: namespace)
        }
        .padding(.horizontal, 2)
        .padding(.vertical, 4)
        .background(TicketInk.paper(blend: blend).opacity(0.55), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct PrefPillMatch: ViewModifier {
    var active: Bool
    var id: String
    var ns: Namespace.ID

    @ViewBuilder
    func body(content: Content) -> some View {
        if active {
            content.matchedGeometryEffect(id: id, in: ns)
        } else {
            content
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
