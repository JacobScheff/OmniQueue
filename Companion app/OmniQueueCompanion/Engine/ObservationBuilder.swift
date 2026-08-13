import Foundation

enum ObservationBuilder {
    struct Built {
        var guest: [Float]
        var ride: [Float] // flattened [numRides, featDim]
        var env: [Float]
        var openRides: Int
        var meanWaitMin: Double
        var warnings: [String]
        var nowSec: Int
        var mustRemaining: [Int]
    }

    static func nowSecOfDay(catalog: ParkCatalog, now: Date = Date()) -> Int {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Los_Angeles") ?? .current
        let hour = calendar.component(.hour, from: now)
        let minute = calendar.component(.minute, from: now)
        let absSec = hour * 3600 + minute * 60
        let openSec = catalog.dayStartHour * 3600
        return min(max(absSec - openSec, 0), catalog.daySeconds)
    }

    static func hourToSecSinceOpen(_ hour: Double, catalog: ParkCatalog, defaultSec: Int) -> Int {
        let absSec = Int(hour * 3600)
        let openSec = catalog.dayStartHour * 3600
        let rel = absSec - openSec
        let cap = catalog.daySeconds + catalog.closeDrainSec
        return min(max(rel, 0), cap)
    }

    static func normalize(_ weights: [Double], count: Int) -> [Float] {
        var w = weights.map { Float(max($0, 0)) }
        if w.count < count { w.append(contentsOf: Array(repeating: 0, count: count - w.count)) }
        if w.count > count { w = Array(w.prefix(count)) }
        let total = w.reduce(0, +)
        if total <= 1e-8 {
            return Array(repeating: 1.0 / Float(count), count: count)
        }
        return w.map { $0 / total }
    }

    static func build(
        catalog: ParkCatalog,
        state: GuestState,
        board: WaitBoard,
        now: Date = Date()
    ) throws -> Built {
        guard let loc = catalog.location(for: state.location) else {
            throw PlannerError.unknownLocation(state.location)
        }
        let prefs = normalize(state.preferenceWeights, count: catalog.numRides)
        let history = padded(state.history, count: catalog.numRides, fill: 0)
        let must = padded(state.mustDos, count: catalog.numRides, fill: 0).map { $0 > 0 ? 1 : 0 }
        var mustRemaining = [Int](repeating: 0, count: catalog.numRides)
        for r in 0..<catalog.numRides {
            mustRemaining[r] = (must[r] > 0 && history[r] == 0) ? 1 : 0
        }

        let nowSec = nowSecOfDay(catalog: catalog, now: now)
        var leaveSec = hourToSecSinceOpen(state.leaveHour, catalog: catalog, defaultSec: catalog.daySeconds)
        leaveSec = max(leaveSec, nowSec)
        let spawnSec = hourToSecSinceOpen(state.arrivalHour, catalog: catalog, defaultSec: 0)

        var guest = [Float](repeating: 0, count: catalog.guestFeatDim)
        for r in 0..<catalog.numRides { guest[r] = prefs[r] }
        let prefExp = Float(catalog.prefRewardExp)
        var remainingMass: Float = 0
        for r in 0..<catalog.numRides where history[r] == 0 {
            remainingMass += pow(max(prefs[r], 0), prefExp)
        }
        guest[34] = remainingMass
        guest[35] = Float(catalog.baseWalkingSpeed) / 2.0
        guest[36] = Float(leaveSec - nowSec) / Float(catalog.daySeconds)
        guest[37] = Float(loc.nodeIdx) / Float(catalog.numNodes)
        guest[38] = Float(min(history.reduce(0, +), 40)) / 20.0
        guest[39] = Float(mustRemaining.reduce(0, +)) / 5.0
        let atRide = loc.atRide
        guest[40] = atRide >= 0 ? 1.0 : 0.0
        guest[41] = 1.0 / 16.0
        guest[42] = Float(max(0, nowSec - spawnSec)) / Float(catalog.daySeconds)

        var ride = [Float](repeating: 0, count: catalog.numRides * catalog.rideFeatDim)
        var waitFeats = [Float](repeating: 0, count: catalog.numRides)
        var warnings: [String] = []
        var openCount = 0
        var waitSum: Double = 0
        var waitN = 0
        var broken = 0

        for r in 0..<catalog.numRides {
            let live = board.wait(for: r)
            var waitSec: Double = 0
            var isOpen = false
            if let live {
                isOpen = live.open
                if let wait = live.waitMin {
                    waitSec = wait * 60
                } else if isOpen {
                    waitSec = 5 * 60
                    warnings.append("no posted wait for \(catalog.rides[r].name); using 5 min")
                }
            }
            if !isOpen { broken += 1 }
            else {
                openCount += 1
                waitSum += waitSec
                waitN += 1
            }

            let waitFeat = Float(min(waitSec, 3600) / 3600)
            waitFeats[r] = waitFeat
            let walkRaw = loc.walkSec.indices.contains(r) ? loc.walkSec[r] : 1
            let walkFeat: Float = (atRide == r) ? 0 : Float(min(Double(walkRaw), 3600) / 3600)
            let duration = Float(catalog.rides[r].durationSec / 900.0)
            let capacity = Float(catalog.rides[r].capacityPerHour / 3600.0)
            let histFeat = Float(min(history[r], 10)) / 10.0
            let unfinished: Float = history[r] == 0 ? pow(max(prefs[r], 0), prefExp) : 0

            let base = r * catalog.rideFeatDim
            ride[base + 0] = waitFeat
            ride[base + 1] = 0
            ride[base + 2] = isOpen ? 1 : 0
            ride[base + 3] = duration
            ride[base + 4] = capacity
            ride[base + 5] = walkFeat
            ride[base + 6] = histFeat
            ride[base + 7] = mustRemaining[r] == 1 ? 1 : 0
            ride[base + 8] = unfinished
            ride[base + 9] = min(walkFeat + waitFeat, 2)
        }

        warnings.append("incoming queue pressure set to 0 (not provided by wait API)")
        let meanWait = waitN > 0 ? waitSum / Double(waitN) : 0
        let meanWaitFeat = Float(meanWait / 3600.0)
        for r in 0..<catalog.numRides {
            ride[r * catalog.rideFeatDim + 10] = max(-1, min(1, waitFeats[r] - meanWaitFeat))
        }

        var env = [Float](repeating: 0, count: catalog.envFeatDim)
        env[0] = Float(nowSec) / Float(catalog.daySeconds)
        env[1] = meanWaitFeat
        env[2] = Float(broken) / Float(catalog.numRides)

        return Built(
            guest: guest,
            ride: ride,
            env: env,
            openRides: openCount,
            meanWaitMin: meanWait / 60.0,
            warnings: warnings,
            nowSec: nowSec,
            mustRemaining: mustRemaining
        )
    }

    static func actionMask(catalog: ParkCatalog, guest: [Float], ride: [Float], env: [Float]) -> [Bool] {
        var mask = [Bool](repeating: false, count: catalog.numActions)
        let timeLeft = max(guest[36], 0)
        let remainingSec = timeLeft * Float(catalog.daySeconds)
        let dayFrac = env[0]
        let softClosed = dayFrac >= 1.0 || timeLeft <= 0
        let drain: Float = dayFrac < 1.0 ? Float(catalog.closeDrainSec) : 0
        let remainingForFeas = remainingSec + drain
        let atRideNode = guest[40] > 0.5

        for r in 0..<catalog.numRides {
            let base = r * catalog.rideFeatDim
            let openOk = ride[base + 2] > 0.5
            let walk = max(ride[base + 5], 0) * 3600
            let wait = max(ride[base + 0], 0) * 3600
            let duration = max(ride[base + 3], 0) * 900
            let timeOk = (walk + wait + duration) <= remainingForFeas
            let alreadyHere = atRideNode && ride[base + 5] <= 1e-6
            mask[r] = openOk && timeOk && !alreadyHere && !softClosed
        }
        mask[catalog.numRides] = true // exit
        if catalog.numRides + 1 < catalog.numActions {
            mask[catalog.numRides + 1] = !softClosed // idle
        }
        return mask
    }

    private static func padded(_ values: [Int], count: Int, fill: Int) -> [Int] {
        var out = values
        if out.count < count { out.append(contentsOf: Array(repeating: fill, count: count - out.count)) }
        if out.count > count { out = Array(out.prefix(count)) }
        return out
    }
}

enum PlannerError: LocalizedError {
    case unknownLocation(String)
    case missingModel
    case inference(String)
    case illegalForce(String)

    var errorDescription: String? {
        switch self {
        case .unknownLocation(let key): return "Unknown location \(key)"
        case .missingModel: return "The on-device model file is missing."
        case .inference(let msg): return msg
        case .illegalForce(let msg): return msg
        }
    }
}
