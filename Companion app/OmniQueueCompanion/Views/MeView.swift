import SwiftUI

struct MeView: View {
    @Environment(AppSession.self) private var session
    @State private var pickingLocation = false
    @State private var confirmClearData = false

    private var hours: [Double] {
        stride(from: session.catalog.dayStartHour * 2, through: session.catalog.dayEndHour * 2, by: 1)
            .map { Double($0) / 2 }
    }

    var body: some View {
        @Bindable var session = session
        Form {
            Section("Disneyland Planner") {
                Button { pickingLocation = true } label: {
                    LabeledContent("Current location", value: session.locationName)
                }
                Picker("Arrival", selection: hour(\.arrivalHour)) { hourChoices }
                Picker("Departure", selection: hour(\.leaveHour)) { hourChoices }
                Toggle("Advanced Ride Scores", isOn: Binding(
                    get: { session.usesAdvancedPrefs }, set: session.setAdvancedPrefs
                ))
            }
            Section("Appearance") {
                Picker("Theme", selection: $session.appearance) {
                    ForEach(AppSession.Appearance.allCases, id: \.self) { choice in
                        Text(choice.rawValue.capitalized).tag(choice)
                    }
                }
            }
            Section("Planner Data") {
                Button("Undo", systemImage: "arrow.uturn.backward") { session.undo() }
                    .disabled(!session.canUndo)
                Button("Redo", systemImage: "arrow.uturn.forward") { session.redo() }
                    .disabled(!session.canRedo)
                Button("Clear Planner Data", role: .destructive) { confirmClearData = true }
            }
            Section {
                Link("Website", destination: AppLinks.marketing)
                Link("Privacy Policy", destination: AppLinks.privacy)
                Link("Support", destination: AppLinks.support)
                Link("ONNX Runtime License", destination: AppLinks.onnxRuntimeLicense)
                Text(AppLinks.versionLabel).foregroundStyle(.secondary)
            } header: {
                Text("About")
            } footer: {
                Text("OmniQueue is an independent app. Planner preferences and visit history stay on this device. The Forecasts service receives the park, date, and forecast start you choose.")
            }
        }
        .navigationTitle("Me")
        .sheet(isPresented: $pickingLocation) { LocationPicker() }
        .confirmationDialog("Clear planner data?", isPresented: $confirmClearData, titleVisibility: .visible) {
            Button("Clear Data", role: .destructive) { session.reset() }
        } message: {
            Text("Clears your ride preferences, must-dos, visit history, and undo history.")
        }
    }

    private var hourChoices: some View {
        ForEach(hours, id: \.self) { Text(WaitFormatting.hourLabel($0)).tag($0) }
    }

    private func hour(_ keyPath: WritableKeyPath<GuestState, Double>) -> Binding<Double> {
        Binding(get: { session.state[keyPath: keyPath] }, set: { value in
            var updated = session.state
            updated[keyPath: keyPath] = value
            if updated != session.state { session.commit(updated) }
        })
    }
}
