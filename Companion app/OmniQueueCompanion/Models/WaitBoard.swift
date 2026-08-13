import Foundation

struct RideWait: Identifiable, Equatable {
    var id: Int { rideId }
    let rideId: Int
    let name: String
    var waitMin: Double?
    var status: String
    var open: Bool
    var entityId: String?
}

struct WaitBoard: Equatable {
    var rides: [RideWait]
    var fetchedAt: Date
    var error: String?

    func wait(for rideId: Int) -> RideWait? {
        rides.first { $0.rideId == rideId }
    }
}

enum WaitTone: String {
    case good, warn, bad, closed, unknown

    static func of(wait: Double?, open: Bool, status: String) -> WaitTone {
        let upper = status.uppercased()
        if upper != "OPERATING" && upper != "UNKNOWN" { return .closed }
        if !open { return .closed }
        guard let wait else { return .unknown }
        if wait <= 20 { return .good }
        if wait <= 45 { return .warn }
        return .bad
    }
}

enum WaitFormatting {
    static func label(wait: Double?, open: Bool, status: String) -> String {
        let upper = status.uppercased()
        switch upper {
        case "DOWN": return "Down"
        case "CLOSED": return "Closed"
        case "REFURBISHMENT": return "Refurb."
        default: break
        }
        if !open { return "Closed" }
        guard let wait else { return "—" }
        return "\(Int(wait.rounded())) min"
    }

    static func hourLabel(_ hour: Double) -> String {
        let h = Int(hour.rounded(.down))
        let m = Int(((hour - Double(h)) * 60).rounded())
        var hour12 = h % 12
        if hour12 == 0 { hour12 = 12 }
        let period = h >= 12 ? "PM" : "AM"
        if m == 0 { return "\(hour12):00 \(period)" }
        return String(format: "%d:%02d %@", hour12, m, period)
    }

    static func percent(_ p: Double) -> String {
        "\(Int((p * 100).rounded()))%"
    }
}
