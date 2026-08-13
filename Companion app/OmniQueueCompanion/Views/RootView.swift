import SwiftUI

struct RootView: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend

    var body: some View {
        @Bindable var session = session
        ZStack {
            TicketInk.paper(blend: blend).ignoresSafeArea()
            RuledBackdrop()
                .ignoresSafeArea()
                .opacity(TicketInk.lerp(0.28, 0.40, blend))
                .allowsHitTesting(false)

            VStack(spacing: 0) {
                TopBar()
                TabView(selection: $session.selectedTab) {
                    PlanView().tag(AppSession.Tab.plan)
                    RidesView().tag(AppSession.Tab.rides)
                    MeView().tag(AppSession.Tab.me)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .background(Color.clear)
                TicketTabBar()
            }

            if session.isBooting {
                BootLoadingView()
                    .transition(.opacity)
                    .zIndex(1)
            }
        }
        .animation(.easeInOut(duration: 0.45), value: session.isBooting)
        .sheet(isPresented: Binding(
            get: { session.showDisclaimer && !session.isBooting },
            set: { if !$0 { session.acknowledgeDisclaimer() } }
        )) {
            DisclaimerSheet()
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
    }
}

private struct RuledBackdrop: View {
    @Environment(\.themeBlend) private var blend

    var body: some View {
        Canvas { context, size in
            let ink = TicketInk.ink(blend: blend)
            var y: CGFloat = 18
            while y < size.height {
                var path = Path()
                path.move(to: CGPoint(x: 18, y: y))
                path.addLine(to: CGPoint(x: size.width - 18, y: y))
                context.stroke(path, with: .color(ink.opacity(0.11)), lineWidth: 1)
                y += 28
            }
            var margin = Path()
            margin.move(to: CGPoint(x: 36, y: 0))
            margin.addLine(to: CGPoint(x: 36, y: size.height))
            context.stroke(margin, with: .color(TicketInk.oxblood.opacity(0.24)), lineWidth: 1.2)
        }
    }
}

private struct TopBar: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            PaperclipMark()
                .offset(y: -6)
            VStack(alignment: .leading, spacing: 2) {
                Text("OmniQueue")
                    .font(TicketType.display)
                    .foregroundStyle(TicketInk.ink(blend: blend))
                HStack(spacing: 8) {
                    TicketSerial()
                        .foregroundStyle(TicketInk.muted(blend: blend))
                    Circle().fill(TicketInk.copperAccent(blend: blend)).frame(width: 4, height: 4)
                    Text(session.isRefreshingWaits || session.isPlanning ? "Rewriting the ticket…" : "Park-day planner")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                        .contentTransition(.opacity)
                }
            }
            Spacer(minLength: 8)
            Button {
                session.toggleTheme()
            } label: {
                Image(systemName: session.usesLightTheme ? "moon.fill" : "sun.max.fill")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(TicketInk.copperAccent(blend: blend))
                    .frame(width: 38, height: 38)
                    .background(TicketInk.stock(blend: blend), in: Circle())
                    .overlay(Circle().stroke(TicketInk.rule(blend: blend), lineWidth: 1))
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Toggle appearance")
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 8)
    }
}

private struct TicketTabBar: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend

    @State private var thumb = CGFloat(0)
    @State private var dragOrigin = CGFloat(0)
    @State private var dragging = false

    private let stub: CGFloat = 14
    private let inset: CGFloat = 6
    private let tabs = AppSession.Tab.allCases

    var body: some View {
        GeometryReader { geo in
            let trackLeading = stub + inset
            let usable = max(geo.size.width - trackLeading - inset, 1)
            let slot = usable / CGFloat(tabs.count)
            let pillW = slot - 6
            let pillH: CGFloat = 44
            let pillX = trackLeading + thumb * slot + (slot - pillW) / 2
            let pillY = (geo.size.height - pillH) / 2

            ZStack(alignment: .topLeading) {
                TicketStock(corner: 24, stubWidth: stub)

                Capsule(style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay {
                        Capsule(style: .continuous)
                            .fill(TicketInk.copperAccent(blend: blend).opacity(TicketInk.lerp(0.28, 0.20, blend)))
                    }
                    .overlay {
                        Capsule(style: .continuous)
                            .strokeBorder(Color.white.opacity(TicketInk.lerp(0.22, 0.45, blend)), lineWidth: 1)
                    }
                    .shadow(color: TicketInk.copperAccent(blend: blend).opacity(0.28), radius: 10, y: 2)
                    .frame(width: pillW, height: pillH)
                    .offset(x: pillX, y: pillY)
                    .animation(dragging ? nil : .spring(duration: 0.42, bounce: 0.28), value: thumb)

                HStack(spacing: 0) {
                    ForEach(tabs) { tab in
                        let distance = abs(thumb - CGFloat(tab.index))
                        VStack(spacing: 4) {
                            Image(systemName: icon(tab))
                                .font(.system(size: 16, weight: .semibold, design: .rounded))
                            Text(tab.title)
                                .font(TicketType.caption)
                        }
                        .foregroundStyle(
                            distance < 0.5
                                ? TicketInk.copperAccent(blend: blend)
                                : TicketInk.muted(blend: blend)
                        )
                        .scaleEffect(distance < 0.5 ? 1.04 : 1)
                        .frame(width: slot, height: geo.size.height)
                    }
                }
                .padding(.leading, trackLeading)
            }
            .contentShape(Rectangle())
            .gesture(slideGesture(slot: slot, trackLeading: trackLeading))
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Section")
            .accessibilityValue(session.selectedTab.title)
            .accessibilityAdjustableAction { direction in
                let delta = direction == .increment ? 1 : -1
                let next = AppSession.Tab.from(index: session.selectedTab.index + delta)
                session.selectedTab = next
                thumb = CGFloat(next.index)
            }
        }
        .frame(height: 64)
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
        .ticketShadow(blend)
        .onAppear { thumb = CGFloat(session.selectedTab.index) }
        .onChange(of: session.selectedTab) { _, tab in
            guard !dragging else { return }
            withAnimation(.spring(duration: 0.42, bounce: 0.28)) {
                thumb = CGFloat(tab.index)
            }
        }
        .sensoryFeedback(.selection, trigger: session.selectedTab)
    }

    private func slideGesture(slot: CGFloat, trackLeading: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                if !dragging {
                    dragging = true
                    dragOrigin = thumb
                }
                let next = dragOrigin + value.translation.width / slot
                thumb = min(max(next, 0), CGFloat(tabs.count - 1))
                let snapped = AppSession.Tab.from(index: Int(thumb.rounded()))
                if snapped != session.selectedTab {
                    session.selectedTab = snapped
                }
            }
            .onEnded { value in
                let travel = hypot(value.translation.width, value.translation.height)
                let idx: Int
                if travel < 14 {
                    let x = value.startLocation.x - trackLeading
                    idx = Int(min(max(floor(x / slot), 0), CGFloat(tabs.count - 1)))
                } else {
                    let predicted = dragOrigin + value.predictedEndTranslation.width / slot
                    idx = Int(min(max(predicted, 0), CGFloat(tabs.count - 1)).rounded())
                }
                dragging = false
                withAnimation(.spring(duration: 0.42, bounce: 0.28)) {
                    thumb = CGFloat(idx)
                    session.selectedTab = AppSession.Tab.from(index: idx)
                }
            }
    }

    private func icon(_ tab: AppSession.Tab) -> String {
        switch tab {
        case .plan: return "ticket.fill"
        case .rides: return "list.bullet.rectangle.fill"
        case .me: return "person.crop.circle"
        }
    }
}

private struct BootLoadingView: View {
    @Environment(\.themeBlend) private var blend
    @State private var stamped = false
    @State private var bob = false

    var body: some View {
        ZStack {
            TicketInk.paper(blend: blend).ignoresSafeArea()
            RuledBackdrop()
                .ignoresSafeArea()
                .opacity(TicketInk.lerp(0.28, 0.40, blend))
                .allowsHitTesting(false)

            VStack(spacing: 20) {
                PaperclipMark()
                    .offset(y: bob ? -5 : 4)

                Text("OmniQueue")
                    .font(TicketType.display)
                    .foregroundStyle(TicketInk.ink(blend: blend))

                VStack(alignment: .leading, spacing: 10) {
                    Text("INKING")
                        .font(TicketType.mono)
                        .tracking(1.6)
                        .foregroundStyle(TicketInk.copperAccent(blend: blend))
                    Text("Fetching the wait board")
                        .font(TicketType.headline)
                        .foregroundStyle(TicketInk.ink(blend: blend))
                    Text("Live times from ThemeParks.wiki, then the on-device planner inks your ticket.")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                        .fixedSize(horizontal: false, vertical: true)
                    InkingDots()
                        .padding(.top, 4)
                }
                .padding(.leading, TicketLayout.leading(16))
                .padding(.trailing, 18)
                .padding(.vertical, 18)
                .frame(maxWidth: 340, alignment: .leading)
                .background { TicketStock(corner: 22, stubWidth: 16) }
                .ticketShadow(blend)
                .scaleEffect(stamped ? 1 : 0.9)
                .opacity(stamped ? 1 : 0)
                .rotationEffect(.degrees(stamped ? -1.2 : 6))
            }
            .padding(.horizontal, 28)
        }
        .onAppear {
            withAnimation(.spring(duration: 0.7, bounce: 0.32)) { stamped = true }
            withAnimation(.easeInOut(duration: 1.15).repeatForever(autoreverses: true)) { bob = true }
        }
    }
}
