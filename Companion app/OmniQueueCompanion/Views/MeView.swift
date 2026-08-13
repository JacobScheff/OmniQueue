import SwiftUI

struct MeView: View {
    @Environment(AppSession.self) private var session
    @Environment(\.themeBlend) private var blend
    @State private var pickingLocation = false
    @State private var confirmClearData = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                section(title: "Where you are") {
                    Button { pickingLocation = true } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Current stop")
                                    .font(TicketType.caption)
                                    .foregroundStyle(TicketInk.muted(blend: blend))
                                Text(session.locationName)
                                    .font(TicketType.headline)
                                    .foregroundStyle(TicketInk.ink(blend: blend))
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundStyle(TicketInk.muted(blend: blend))
                        }
                        .padding(.leading, TicketLayout.leading(12))
                        .padding(.trailing, 14)
                        .padding(.vertical, 14)
                        .background { TicketStock(corner: 16, stubWidth: 12) }
                    }
                    .buttonStyle(.plain)

                    HStack(spacing: 10) {
                        HourField(title: "Arrived around", hour: session.state.arrivalHour, range: range) { value in
                            var s = session.state
                            s.arrivalHour = value
                            session.commit(s)
                        }
                        HourField(title: "Leaving around", hour: session.state.leaveHour, range: range) { value in
                            var s = session.state
                            s.leaveHour = value
                            session.commit(s)
                        }
                    }
                }

                section(title: "Preferences") {
                    PrefModeSwitch()
                    Text(session.usesAdvancedPrefs
                         ? "Each ride gets a 0–100 score. Switch back to Basic anytime — your numbers stay until you change Skip / Low / Medium / High."
                         : "Skip / Low / Medium / High is the simple scale. Advanced unlocks a 0–100 number per ride.")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                        .fixedSize(horizontal: false, vertical: true)
                        .contentTransition(.opacity)
                        .animation(.spring(duration: 0.55, bounce: 0.22), value: session.usesAdvancedPrefs)
                }

                section(title: "Edits") {
                    HStack(spacing: 8) {
                        ghost("arrow.uturn.backward", "Undo", enabled: session.canUndo) { session.undo() }
                        ghost("arrow.uturn.forward", "Redo", enabled: session.canRedo) { session.redo() }
                        ghost("trash", "Clear data", enabled: true, danger: true) { confirmClearData = true }
                    }
                    Text("Clear data removes preferences, must-dos, ride history, and undo history stored on this iPhone. Nothing is uploaded to OmniQueue servers.")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                        .fixedSize(horizontal: false, vertical: true)
                }

                section(title: "What these mean") {
                    helpRow("Priority", "How much you want to ride. In Basic, pick Skip / Low / Medium / High. In Advanced, set a 0–100 number. Switching to Advanced fills in numbers from the category; changing a category in Basic overwrites that ride’s number.")
                    helpRow("Must-do", "Star the rides you don’t want to miss. The plan works those in first.")
                    helpRow("Rode it", "Mark something done so it won’t keep coming back as the next stop.")
                    helpRow("Pinning", "On Plan, tap any stop and pick an alternative. Everything else rewrites around that pin.")
                    helpRow("Location", "Pick the land or ride you’re standing at. The app never uses GPS.")
                }

                section(title: "About") {
                    linkRow(title: "Website", systemImage: "globe", url: AppLinks.marketing)
                    linkRow(title: "Privacy Policy", systemImage: "hand.raised.fill", url: AppLinks.privacy)
                    linkRow(title: "Support", systemImage: "questionmark.circle.fill", url: AppLinks.support)
                    linkRow(title: "ThemeParks.wiki", systemImage: "link", url: AppLinks.themeParksWiki)
                    linkRow(title: "ONNX Runtime license", systemImage: "doc.text", url: AppLinks.onnxRuntimeLicense)

                    Text(AppLinks.versionLabel)
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                        .padding(.top, 4)
                    Text("Routing runs on-device. Live waits are requested from ThemeParks.wiki over HTTPS. OmniQueue does not create accounts or sell your data.")
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                        .fixedSize(horizontal: false, vertical: true)
                }

                DisclaimerBlock()
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 28)
        }
        .scrollIndicators(.hidden)
        .sheet(isPresented: $pickingLocation) {
            LocationPicker()
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .confirmationDialog(
            "Clear local data?",
            isPresented: $confirmClearData,
            titleVisibility: .visible
        ) {
            Button("Clear data", role: .destructive) { session.reset() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This resets preferences, must-dos, completions, location, leave time, and undo history on this device.")
        }
    }

    private var range: ClosedRange<Double> {
        Double(session.catalog.dayStartHour)...Double(session.catalog.dayEndHour)
    }

    private func section<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(TicketType.headline)
                .foregroundStyle(TicketInk.ink(blend: blend))
            content()
        }
    }

    private func helpRow(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(TicketType.caption.weight(.bold))
                .foregroundStyle(TicketInk.copperAccent(blend: blend))
            Text(body)
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.muted(blend: blend))
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(TicketInk.stock(blend: blend), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func linkRow(title: String, systemImage: String, url: URL) -> some View {
        Link(destination: url) {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(TicketInk.copperAccent(blend: blend))
                    .frame(width: 22)
                Text(title)
                    .font(TicketType.body.weight(.semibold))
                    .foregroundStyle(TicketInk.ink(blend: blend))
                Spacer()
                Image(systemName: "arrow.up.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(TicketInk.muted(blend: blend))
            }
            .padding(.leading, TicketLayout.leading(12))
            .padding(.trailing, 14)
            .padding(.vertical, 12)
            .background { TicketStock(corner: 14, stubWidth: 12, punched: false) }
        }
        .accessibilityLabel(title)
    }

    private func ghost(_ icon: String, _ title: String, enabled: Bool, danger: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .font(TicketType.caption.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.leading, TicketLayout.leading(10))
                .padding(.trailing, 10)
                .padding(.vertical, 10)
                .foregroundStyle(danger ? TicketInk.oxblood : TicketInk.ink(blend: blend))
                .background { TicketStock(corner: 12, stubWidth: 10, punched: false) }
                .opacity(enabled ? 1 : 0.4)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}

private struct HourField: View {
    let title: String
    let hour: Double
    let range: ClosedRange<Double>
    var onChange: (Double) -> Void
    @Environment(\.themeBlend) private var blend

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(TicketType.caption)
                .foregroundStyle(TicketInk.muted(blend: blend))
            HStack {
                Button { nudge(-0.5) } label: {
                    Image(systemName: "minus")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                Text(WaitFormatting.hourLabel(hour))
                    .font(TicketType.body.weight(.semibold))
                    .frame(maxWidth: .infinity)
                Button { nudge(0.5) } label: {
                    Image(systemName: "plus")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
            }
            .foregroundStyle(TicketInk.ink(blend: blend))
            .padding(.leading, TicketLayout.leading(10))
            .padding(.trailing, 10)
            .padding(.vertical, 10)
            .background { TicketStock(corner: 14, stubWidth: 10, punched: false) }
        }
        .frame(maxWidth: .infinity)
    }

    private func nudge(_ delta: Double) {
        let next = min(range.upperBound, max(range.lowerBound, hour + delta))
        onChange(next)
    }
}
