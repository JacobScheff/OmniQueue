import Foundation

struct GuestState: Equatable, Codable {
    var preferenceWeights: [Double]
    var mustDos: [Int]
    var history: [Int]
    var location: String
    var leaveHour: Double
    var arrivalHour: Double

    static func `default`(catalog: ParkCatalog) -> GuestState {
        GuestState(
            preferenceWeights: catalog.defaultPreferenceWeights,
            mustDos: Array(repeating: 0, count: catalog.numRides),
            history: Array(repeating: 0, count: catalog.numRides),
            location: "entrance",
            leaveHour: Double(catalog.dayEndHour),
            arrivalHour: Double(catalog.dayStartHour)
        )
    }

    mutating func setWeight(_ value: Double, ride: Int) {
        guard preferenceWeights.indices.contains(ride) else { return }
        preferenceWeights[ride] = max(0, min(250, value))
    }

    mutating func toggleMustDo(ride: Int) {
        guard mustDos.indices.contains(ride) else { return }
        mustDos[ride] = mustDos[ride] == 0 ? 1 : 0
    }

    mutating func bumpDone(ride: Int, delta: Int) {
        guard history.indices.contains(ride) else { return }
        history[ride] = max(0, min(20, history[ride] + delta))
    }
}

struct ForcedPick: Equatable {
    var slot: Int
    var actionId: Int
}
