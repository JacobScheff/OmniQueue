import Foundation

struct DistRow: Identifiable, Equatable {
    var id: Int { actionId }
    let actionId: Int
    let label: String
    let prob: Double
    let legal: Bool
    let isRide: Bool
    var waitMin: Double? = nil
    var status: String = "UNKNOWN"
    var open: Bool = false
}

struct RouteStop: Identifiable, Equatable {
    var id: Int { slot }
    let actionId: Int
    let label: String
    let slot: Int
    let isRide: Bool
    let probSlot: Double?
}

struct Recommendation: Equatable {
    let recommended: DistRow
    let naturalRecommended: DistRow
    let forcedSlot: Int?
    let forcedAction: Int?
    let route: [RouteStop]
    let distribution: [DistRow]
    let distributionsBySlot: [[DistRow]]
    let openRides: Int
    let meanWaitMin: Double
    let warnings: [String]
    let nowSec: Int
}
