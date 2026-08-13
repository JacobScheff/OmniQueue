import SwiftUI

struct LocationPicker: View {
    @Environment(AppSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var scheme
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
            .background(TicketInk.paper(for: scheme))
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
        let q = query.lowercased()
        if q.isEmpty { return session.catalog.hubs }
        return session.catalog.hubs.filter { $0.name.lowercased().contains(q) }
    }

    private var filteredRides: [ParkCatalog.Ride] {
        let q = query.lowercased()
        if q.isEmpty { return session.catalog.rides }
        return session.catalog.rides.filter {
            $0.name.lowercased().contains(q) || $0.hubName.lowercased().contains(q)
        }
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
                        .foregroundStyle(TicketInk.ink(for: scheme))
                    Text(subtitle)
                        .font(TicketType.caption)
                        .foregroundStyle(TicketInk.muted(for: scheme))
                }
                Spacer()
                if session.state.location == key {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(TicketInk.copperAccent(for: scheme))
                }
            }
        }
    }
}
