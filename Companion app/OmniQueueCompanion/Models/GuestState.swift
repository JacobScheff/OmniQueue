import Foundation

struct GuestState: Equatable, Codable {
    var preferenceWeights: [Double]
    /// Last Skip/Low/Medium/High choice. Kept even while the 0–100 score is edited.
    var preferenceLevels: [PriorityLevel]
    /// Fine 0–100 scores. `nil` means this ride has never been given a number.
    var preferenceFine: [Double?]
    var mustDos: [Int]
    var history: [Int]
    var location: String
    var leaveHour: Double
    var arrivalHour: Double

    static func `default`(catalog: ParkCatalog) -> GuestState {
        let weights = catalog.defaultPreferenceWeights
        return GuestState(
            preferenceWeights: weights,
            preferenceLevels: weights.map { PriorityLevel.from(weight: $0) },
            preferenceFine: Array(repeating: nil, count: weights.count),
            mustDos: Array(repeating: 0, count: catalog.numRides),
            history: Array(repeating: 0, count: catalog.numRides),
            location: "entrance",
            leaveHour: Double(catalog.dayEndHour),
            arrivalHour: Double(catalog.dayStartHour)
        )
    }

    func level(for ride: Int) -> PriorityLevel {
        preferenceLevels[safe: ride] ?? .medium
    }

    func score(for ride: Int) -> Double {
        if let stored = preferenceFine[safe: ride], let stored {
            return stored
        }
        return Double(level(for: ride).score)
    }

    mutating func setLevel(_ level: PriorityLevel, ride: Int) {
        ensurePrefCapacity(ride)
        guard preferenceLevels[ride] != level else { return }
        preferenceLevels[ride] = level
        preferenceFine[ride] = Double(level.score)
        preferenceWeights[ride] = level.modelWeight
    }

    mutating func setScore(_ score: Double, ride: Int) {
        ensurePrefCapacity(ride)
        let clamped = min(max(score.rounded(), 0), 100)
        preferenceFine[ride] = clamped
        preferenceWeights[ride] = PriorityLevel.modelWeight(fromScore: clamped)
    }

    /// Assigns category defaults to any ride that still has no 0–100 number.
    @discardableResult
    mutating func fillMissingScores() -> Bool {
        var changed = false
        let n = max(preferenceWeights.count, preferenceLevels.count)
        ensurePrefCapacity(max(n - 1, 0))
        for i in 0..<preferenceFine.count {
            if preferenceFine[i] == nil {
                let level = preferenceLevels[safe: i] ?? .medium
                preferenceFine[i] = Double(level.score)
                if preferenceWeights.indices.contains(i) {
                    preferenceWeights[i] = level.modelWeight
                }
                changed = true
            }
        }
        return changed
    }

    mutating func toggleMustDo(ride: Int) {
        guard mustDos.indices.contains(ride) else { return }
        mustDos[ride] = mustDos[ride] == 0 ? 1 : 0
    }

    mutating func bumpDone(ride: Int, delta: Int) {
        guard history.indices.contains(ride) else { return }
        history[ride] = max(0, min(20, history[ride] + delta))
    }

    private mutating func ensurePrefCapacity(_ ride: Int) {
        let need = ride + 1
        while preferenceWeights.count < need { preferenceWeights.append(1) }
        while preferenceLevels.count < need { preferenceLevels.append(.medium) }
        while preferenceFine.count < need { preferenceFine.append(nil) }
    }

    enum CodingKeys: String, CodingKey {
        case preferenceWeights, preferenceLevels, preferenceFine
        case mustDos, history, location, leaveHour, arrivalHour
    }

    init(
        preferenceWeights: [Double],
        preferenceLevels: [PriorityLevel],
        preferenceFine: [Double?],
        mustDos: [Int],
        history: [Int],
        location: String,
        leaveHour: Double,
        arrivalHour: Double
    ) {
        self.preferenceWeights = preferenceWeights
        self.preferenceLevels = preferenceLevels
        self.preferenceFine = preferenceFine
        self.mustDos = mustDos
        self.history = history
        self.location = location
        self.leaveHour = leaveHour
        self.arrivalHour = arrivalHour
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        preferenceWeights = try c.decode([Double].self, forKey: .preferenceWeights)
        mustDos = try c.decode([Int].self, forKey: .mustDos)
        history = try c.decode([Int].self, forKey: .history)
        location = try c.decode(String.self, forKey: .location)
        leaveHour = try c.decode(Double.self, forKey: .leaveHour)
        arrivalHour = try c.decode(Double.self, forKey: .arrivalHour)
        let n = preferenceWeights.count
        if let levels = try c.decodeIfPresent([PriorityLevel].self, forKey: .preferenceLevels), !levels.isEmpty {
            preferenceLevels = Self.pad(levels, count: n, fill: .medium)
        } else {
            preferenceLevels = preferenceWeights.map { PriorityLevel.from(weight: $0) }
        }
        if let fine = try c.decodeIfPresent([Double?].self, forKey: .preferenceFine) {
            preferenceFine = Self.pad(fine, count: n, fill: nil)
        } else {
            preferenceFine = Array(repeating: nil, count: n)
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(preferenceWeights, forKey: .preferenceWeights)
        try c.encode(preferenceLevels, forKey: .preferenceLevels)
        try c.encode(preferenceFine, forKey: .preferenceFine)
        try c.encode(mustDos, forKey: .mustDos)
        try c.encode(history, forKey: .history)
        try c.encode(location, forKey: .location)
        try c.encode(leaveHour, forKey: .leaveHour)
        try c.encode(arrivalHour, forKey: .arrivalHour)
    }

    private static func pad<T>(_ arr: [T], count: Int, fill: T) -> [T] {
        var out = Array(arr.prefix(count))
        while out.count < count { out.append(fill) }
        return out
    }
}

struct ForcedPick: Equatable {
    var slot: Int
    var actionId: Int
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
