import Foundation

struct ParkCatalog: Decodable, Sendable {
    let numRides: Int
    let numActions: Int
    let guestFeatDim: Int
    let rideFeatDim: Int
    let envFeatDim: Int
    let flatObsDim: Int
    let routeK: Int
    let dayStartHour: Int
    let dayEndHour: Int
    let daySeconds: Int
    let closeDrainSec: Int
    let baseWalkingSpeed: Double
    let prefRewardExp: Double
    let weightSliderMax: Double
    let numNodes: Int
    let parkEntityId: String
    let waitApiBase: String
    let defaultPreferenceWeights: [Double]
    let hubs: [Hub]
    let rides: [Ride]
    let nameAliases: [String: Int]
    let actionLabels: [String]
    let model: ModelInfo

    struct Hub: Decodable, Identifiable, Hashable, Sendable {
        let id: Int
        let key: String
        let name: String
        let kind: String
        let nodeId: Int
        let nodeIdx: Int
        let atRide: Int
        let walkSec: [Int]
    }

    struct Ride: Decodable, Identifiable, Hashable, Sendable {
        let id: Int
        let name: String
        let hubId: Int
        let hubName: String
        let locationKey: String
        let popularity: Int
        let durationSec: Double
        let durationMin: Double
        let capacityPerHour: Double
        let nodeId: Int
        let nodeIdx: Int
        let entityId: String?
        let walkSec: [Int]
    }

    struct ModelInfo: Decodable, Sendable {
        let id: String
        let filename: String
        let routeK: Int
        let rideFeatDim: Int
        let supportsForceAnySlot: Bool
    }

    static func loadFromBundle() throws -> ParkCatalog {
        guard let url = Bundle.main.url(forResource: "ParkData", withExtension: "json") else {
            throw CatalogError.missingFile
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ParkCatalog.self, from: data)
    }

    func ride(id: Int) -> Ride? {
        rides.first { $0.id == id }
    }

    func location(for key: String) -> LocationRef? {
        if let hub = hubs.first(where: { $0.key == key }) {
            return LocationRef(key: hub.key, name: hub.name, kind: .hub, nodeId: hub.nodeId, nodeIdx: hub.nodeIdx, atRide: -1, walkSec: hub.walkSec)
        }
        if let ride = rides.first(where: { $0.locationKey == key }) {
            return LocationRef(key: ride.locationKey, name: ride.name, kind: .ride, nodeId: ride.nodeId, nodeIdx: ride.nodeIdx, atRide: ride.id, walkSec: ride.walkSec)
        }
        return hubs.first.map {
            LocationRef(key: $0.key, name: $0.name, kind: .hub, nodeId: $0.nodeId, nodeIdx: $0.nodeIdx, atRide: -1, walkSec: $0.walkSec)
        }
    }

    func actionLabel(_ id: Int) -> String {
        if id >= 0 && id < actionLabels.count { return actionLabels[id] }
        return "action \(id)"
    }

    enum CatalogError: Error { case missingFile }
}

struct LocationRef: Hashable, Identifiable, Sendable {
    enum Kind { case hub, ride }
    var id: String { key }
    let key: String
    let name: String
    let kind: Kind
    let nodeId: Int
    let nodeIdx: Int
    let atRide: Int
    let walkSec: [Int]
}

enum PriorityLevel: Int, CaseIterable, Identifiable {
    case skip = 0
    case low = 50
    case medium = 130
    case high = 220

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .skip: return "Skip"
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }

    static func from(weight: Double) -> PriorityLevel {
        if weight <= 0 { return .skip }
        if weight < 90 { return .low }
        if weight < 175 { return .medium }
        return .high
    }
}
