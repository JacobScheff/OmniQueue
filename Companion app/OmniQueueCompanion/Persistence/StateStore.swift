import Foundation

@MainActor
final class StateStore {
    private let key = "omniqueue.companion.ios.v1"
    private let undoKey = "omniqueue.companion.ios.undo.v1"
    private let maxUndo = 50

    func load(catalog: ParkCatalog) -> (GuestState, [GuestState], [GuestState]) {
        let fallback = GuestState.default(catalog: catalog)
        guard let data = UserDefaults.standard.data(forKey: key) else {
            return (fallback, [], [])
        }
        do {
            let box = try JSONDecoder().decode(Box.self, from: data)
            return (
                normalize(box.state, catalog: catalog),
                box.past.suffix(maxUndo).map { normalize($0, catalog: catalog) },
                box.future.suffix(maxUndo).map { normalize($0, catalog: catalog) }
            )
        } catch {
            return (fallback, [], [])
        }
    }

    func save(state: GuestState, past: [GuestState], future: [GuestState]) {
        let box = Box(state: state, past: Array(past.suffix(maxUndo)), future: Array(future.suffix(maxUndo)))
        if let data = try? JSONEncoder().encode(box) {
            UserDefaults.standard.set(data, forKey: key)
            UserDefaults.standard.set(past.count, forKey: undoKey)
        }
    }

    private func normalize(_ state: GuestState, catalog: ParkCatalog) -> GuestState {
        func pad(_ arr: [Double], fill: Double) -> [Double] {
            var out = Array(arr.prefix(catalog.numRides))
            while out.count < catalog.numRides { out.append(fill) }
            return out
        }
        func padInt(_ arr: [Int], fill: Int) -> [Int] {
            var out = Array(arr.prefix(catalog.numRides))
            while out.count < catalog.numRides { out.append(fill) }
            return out
        }
        return GuestState(
            preferenceWeights: pad(state.preferenceWeights, fill: 1),
            mustDos: padInt(state.mustDos, fill: 0).map { $0 == 0 ? 0 : 1 },
            history: padInt(state.history, fill: 0).map { max(0, $0) },
            location: state.location.isEmpty ? "entrance" : state.location,
            leaveHour: state.leaveHour,
            arrivalHour: state.arrivalHour
        )
    }

    private struct Box: Codable {
        var state: GuestState
        var past: [GuestState]
        var future: [GuestState]
    }
}
