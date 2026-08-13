import SwiftUI

struct LocationPicker: View {
    @Environment(AppSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @Environment(\.themeBlend) private var blend
    @State private var query = ""

    var body: some View {
        NavigationStack {
            List {
                Section("Lands") {
                    ForEach(filteredHubs) { hub in
                        row(key: hub.key, name: hub.name, subtitle: hub.kind == "entrance" ? "Gate" : "Land")
                    }
                }
                Section("Rides") {
                    ForEach(filteredRides) { ride in
                        row(key: ride.locationKey, name: ride.name, subtitle: ride.hubName)
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(TicketInk.paper(blend: blend))
            .searchable(text: $query, prompt: "Land or ride")
            .navigationTitle("Where are you?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }

    private var filteredHubs: [ParkCatalog.Hub] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if q.isEmpty { return session.catalog.hubs }
        return session.catalog.hubs.filter { FuzzyMatch.matches(q, in: $0.name) }
    }

    private var filteredRides: [ParkCatalog.Ride] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if q.isEmpty { return session.catalog.rides }
        return session.catalog.rides.filter { FuzzyMatch.matches(q, in: $0.name, $0.hubName) }
    }

    private func row(key: String, name: String, subtitle: String) -> some View {
        Button {
            var s = session.state
            s.location = key
            session.commit(s)
            dismiss()
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .foregroundStyle(TicketInk.ink(blend: blend))
                    Text(subtitle)
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(blend: blend))
                }
                Spacer()
                if session.state.location == key {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(TicketInk.copperAccent(blend: blend))
                }
            }
        }
    }
}
