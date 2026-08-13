import SwiftUI

struct RootView: View {
    @Environment(AppSession.self) private var session
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            TicketInk.paper(for: scheme).ignoresSafeArea()
            RuledBackdrop()
                .ignoresSafeArea()
                .opacity(scheme == .dark ? 0.18 : 0.28)
                .allowsHitTesting(false)

            VStack(spacing: 0) {
                TopBar()
                tabBody
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                TicketTabBar()
            }
        }
        .sheet(isPresented: Binding(
            get: { session.showDisclaimer },
            set: { if !$0 { session.acknowledgeDisclaimer() } }
        )) {
            DisclaimerSheet()
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
    }

    @ViewBuilder
    private var tabBody: some View {
        switch session.selectedTab {
        case .plan:
            PlanView()
                .transition(.asymmetric(insertion: .move(edge: .leading).combined(with: .opacity), removal: .opacity))
        case .rides:
            RidesView()
                .transition(.opacity)
        case .me:
            MeView()
                .transition(.asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity), removal: .opacity))
        }
    }
}

private struct RuledBackdrop: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Canvas { context, size in
            let ink = TicketInk.ink(for: scheme)
            var y: CGFloat = 18
            while y < size.height {
                var path = Path()
                path.move(to: CGPoint(x: 18, y: y))
                path.addLine(to: CGPoint(x: size.width - 18, y: y))
                context.stroke(path, with: .color(ink.opacity(0.08)), lineWidth: 1)
                y += 28
            }
            var margin = Path()
            margin.move(to: CGPoint(x: 36, y: 0))
            margin.addLine(to: CGPoint(x: 36, y: size.height))
            context.stroke(margin, with: .color(TicketInk.oxblood.opacity(0.18)), lineWidth: 1.2)
        }
    }
}

private struct TopBar: View {
    @Environment(AppSession.self) private var session
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            PaperclipMark()
                .offset(y: -6)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text("OmniQueue")
                        .font(TicketType.display)
                        .foregroundStyle(TicketInk.ink(for: scheme))
                    GuestCopyStamp()
                        .scaleEffect(0.82)
                        .offset(y: -2)
                }
                HStack(spacing: 8) {
                    TicketSerial()
                        .foregroundStyle(TicketInk.muted(for: scheme))
                    Circle().fill(TicketInk.copperAccent(for: scheme)).frame(width: 4, height: 4)
                    Text(session.busy ? "Rewriting the ticket…" : "Park-day planner")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(for: scheme))
                        .contentTransition(.opacity)
                }
            }
            Spacer(minLength: 8)
            Button {
                withAnimation(.spring(duration: 0.45, bounce: 0.28)) { session.toggleTheme() }
            } label: {
                Image(systemName: session.preferredScheme == .light ? "moon.fill" : "sun.max.fill")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(TicketInk.copperAccent(for: scheme))
                    .frame(width: 38, height: 38)
                    .background(TicketInk.stock(for: scheme), in: Circle())
                    .overlay(Circle().stroke(TicketInk.rule(for: scheme), lineWidth: 1))
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
    @Environment(\.colorScheme) private var scheme
    @Namespace private var ns

    var body: some View {
        HStack(spacing: 4) {
            ForEach(AppSession.Tab.allCases) { tab in
                Button {
                    withAnimation(.spring(duration: 0.42, bounce: 0.22)) {
                        session.selectedTab = tab
                    }
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: icon(tab))
                            .font(.system(size: 16, weight: .semibold, design: .rounded))
                        Text(tab.title)
                            .font(TicketType.caption)
                    }
                    .foregroundStyle(session.selectedTab == tab ? TicketInk.copperAccent(for: scheme) : TicketInk.muted(for: scheme))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background {
                        if session.selectedTab == tab {
                            Capsule(style: .continuous)
                                .fill(TicketInk.copperAccent(for: scheme).opacity(0.14))
                                .matchedGeometryEffect(id: "tab", in: ns)
                        }
                    }
                }
                .buttonStyle(.plain)
                .sensoryFeedback(.selection, trigger: session.selectedTab)
            }
        }
        .padding(6)
        .background {
            TicketStock(corner: 22, stubWidth: 14)
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
        .ticketShadow(scheme)
    }

    private func icon(_ tab: AppSession.Tab) -> String {
        switch tab {
        case .plan: return "ticket.fill"
        case .rides: return "list.bullet.rectangle.fill"
        case .me: return "person.crop.circle"
        }
    }
}
