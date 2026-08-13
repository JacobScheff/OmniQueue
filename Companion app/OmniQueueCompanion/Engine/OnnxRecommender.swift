import Foundation
#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

/// Runs the bundled v2 rank-route ONNX graph on-device.
final class OnnxRecommender: @unchecked Sendable {
    private let env: ORTEnv
    private let session: ORTSession
    private let catalog: ParkCatalog

    init(catalog: ParkCatalog) throws {
        self.catalog = catalog
        guard let url = Bundle.main.url(forResource: "v2", withExtension: "onnx") else {
            throw PlannerError.missingModel
        }
        env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        try options.setGraphOptimizationLevel(.all)
        session = try ORTSession(env: env, modelPath: url.path, sessionOptions: options)
    }

    func recommend(
        built: ObservationBuilder.Built,
        board: WaitBoard,
        force: ForcedPick?
    ) throws -> Recommendation {
        let legal = ObservationBuilder.actionMask(
            catalog: catalog,
            guest: built.guest,
            ride: built.ride,
            env: built.env
        )

        var forceSlot = -1
        var forceAction = -1
        if let force {
            try validate(force: force, legal: legal)
            forceSlot = force.slot
            forceAction = force.actionId
        }

        let outputs = try session.run(
            withInputs: [
                "guest": try tensor(built.guest, shape: [1, catalog.guestFeatDim]),
                "ride": try tensor(built.ride, shape: [1, catalog.numRides, catalog.rideFeatDim]),
                "env": try tensor(built.env, shape: [1, catalog.envFeatDim]),
                "force_slot": try int64Tensor([Int64(forceSlot)], shape: [1]),
                "force_action": try int64Tensor([Int64(forceAction)], shape: [1]),
            ],
            outputNames: Set(["route", "slot0_logits", "slot_logits", "slot_masks"]),
            runOptions: nil
        )

        guard
            let routeVal = outputs["route"],
            let slot0Val = outputs["slot0_logits"],
            let slotLogitsVal = outputs["slot_logits"],
            let slotMasksVal = outputs["slot_masks"]
        else {
            throw PlannerError.inference("Model did not return route outputs.")
        }

        var slot0 = try floats(from: slot0Val)
        if slot0.count > catalog.numActions {
            slot0 = Array(slot0.prefix(catalog.numActions))
        }
        for i in slot0.indices where i < legal.count && !legal[i] {
            slot0[i] = -1e9
        }

        let naturalProbs = softmax(slot0, legal: legal)
        let naturalAction = naturalProbs.enumerated().max(by: { $0.element < $1.element })?.offset ?? catalog.numRides

        var routeIds = try int64s(from: routeVal).map(Int.init).filter { $0 >= 0 }
        if routeIds.isEmpty {
            routeIds = [forceSlot == 0 ? forceAction : naturalAction]
        }
        if forceSlot >= 0 {
            if forceSlot >= routeIds.count || routeIds[forceSlot] != forceAction {
                throw PlannerError.inference(
                    "Couldn’t pin \(catalog.actionLabel(forceAction)) at stop \(forceSlot + 1). Try another option."
                )
            }
        }

        let slotLogits = try floats(from: slotLogitsVal)
        let slotMasks = try floats(from: slotMasksVal).map { $0 > 0.5 }
        let k = catalog.routeK
        let a = catalog.numActions

        var distributionsBySlot: [[DistRow]] = []
        let nSlots = min(routeIds.count, k)
        for slot in 0..<nSlots {
            let dim = slot == 0 ? catalog.numActions : catalog.numRides
            var logits = [Float](repeating: -1e9, count: dim)
            var mask = [Bool](repeating: false, count: dim)
            for i in 0..<dim {
                let idx = slot * a + i
                if idx < slotLogits.count { logits[i] = slotLogits[idx] }
                if idx < slotMasks.count { mask[i] = slotMasks[idx] }
            }
            if slot == 0 {
                logits = slot0
                mask = Array(legal.prefix(dim))
            }
            distributionsBySlot.append(distRows(logits: logits, legal: mask))
        }
        if distributionsBySlot.isEmpty {
            distributionsBySlot = [distRows(logits: slot0, legal: legal)]
        }

        annotate(distributionsBySlot: &distributionsBySlot, board: board)

        let action = routeIds[0]
        let slot0Dist = distributionsBySlot[0]
        let recProb = slot0Dist.first(where: { $0.actionId == action })?.prob ?? Double(naturalProbs[safe: action] ?? 0)

        let route: [RouteStop] = routeIds.enumerated().map { slot, aid in
            let p = distributionsBySlot.indices.contains(slot)
                ? distributionsBySlot[slot].first(where: { $0.actionId == aid })?.prob
                : nil
            return RouteStop(
                actionId: aid,
                label: catalog.actionLabel(aid),
                slot: slot,
                isRide: aid < catalog.numRides,
                probSlot: p
            )
        }

        return Recommendation(
            recommended: DistRow(
                actionId: action,
                label: catalog.actionLabel(action),
                prob: recProb,
                legal: action < legal.count ? legal[action] : false,
                isRide: action < catalog.numRides
            ),
            naturalRecommended: DistRow(
                actionId: naturalAction,
                label: catalog.actionLabel(naturalAction),
                prob: Double(naturalProbs[safe: naturalAction] ?? 0),
                legal: naturalAction < legal.count ? legal[naturalAction] : false,
                isRide: naturalAction < catalog.numRides
            ),
            forcedSlot: forceSlot >= 0 ? forceSlot : nil,
            forcedAction: forceAction >= 0 ? forceAction : nil,
            route: route,
            distribution: distributionsBySlot[0],
            distributionsBySlot: distributionsBySlot,
            openRides: built.openRides,
            meanWaitMin: built.meanWaitMin,
            warnings: built.warnings,
            nowSec: built.nowSec
        )
    }

    private func validate(force: ForcedPick, legal: [Bool]) throws {
        if force.slot < 0 || force.slot >= catalog.routeK {
            throw PlannerError.illegalForce("Pin slot out of range.")
        }
        if force.actionId < 0 || force.actionId >= catalog.numActions {
            throw PlannerError.illegalForce("Pinned action is unknown.")
        }
        if force.slot > 0 && force.actionId >= catalog.numRides {
            throw PlannerError.illegalForce("Later stops have to be rides.")
        }
        if force.slot == 0, force.actionId < legal.count, !legal[force.actionId] {
            throw PlannerError.illegalForce("\(catalog.actionLabel(force.actionId)) isn’t available right now.")
        }
    }

    private func distRows(logits: [Float], legal: [Bool]) -> [DistRow] {
        let n = min(logits.count, legal.count)
        let probs = softmax(Array(logits.prefix(n)), legal: Array(legal.prefix(n)))
        var rows: [DistRow] = []
        for i in 0..<n {
            rows.append(
                DistRow(
                    actionId: i,
                    label: catalog.actionLabel(i),
                    prob: Double(probs[i]),
                    legal: legal[i],
                    isRide: i < catalog.numRides
                )
            )
        }
        return rows.sorted { $0.prob > $1.prob }
    }

    private func annotate(distributionsBySlot: inout [[DistRow]], board: WaitBoard) {
        for s in distributionsBySlot.indices {
            for i in distributionsBySlot[s].indices where distributionsBySlot[s][i].isRide {
                if let live = board.wait(for: distributionsBySlot[s][i].actionId) {
                    distributionsBySlot[s][i].waitMin = live.waitMin
                    distributionsBySlot[s][i].status = live.status
                    distributionsBySlot[s][i].open = live.open
                }
            }
        }
    }

    private func softmax(_ logits: [Float], legal: [Bool]) -> [Float] {
        var masked = logits
        for i in masked.indices where i >= legal.count || !legal[i] {
            masked[i] = -1e9
        }
        guard legal.contains(true) else { return [Float](repeating: 0, count: logits.count) }
        let peak = masked.max() ?? 0
        let exps = masked.map { Double(exp($0 - peak)) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return [Float](repeating: 0, count: logits.count) }
        return exps.map { Float($0 / sum) }
    }

    private func tensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        let data = NSMutableData(length: values.count * MemoryLayout<Float>.stride)!
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                data.replaceBytes(in: NSRange(location: 0, length: data.length), withBytes: base)
            }
        }
        return try ORTValue(
            tensorData: data,
            elementType: .float,
            shape: shape.map { NSNumber(value: $0) }
        )
    }

    private func int64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let data = NSMutableData(length: values.count * MemoryLayout<Int64>.stride)!
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                data.replaceBytes(in: NSRange(location: 0, length: data.length), withBytes: base)
            }
        }
        return try ORTValue(
            tensorData: data,
            elementType: .int64,
            shape: shape.map { NSNumber(value: $0) }
        )
    }

    private func floats(from value: ORTValue) throws -> [Float] {
        copyTensor(try value.tensorData(), as: Float.self)
    }

    private func int64s(from value: ORTValue) throws -> [Int64] {
        let nsData = try value.tensorData()
        let info = try value.tensorTypeAndShapeInfo()
        if info.elementType == .int64 {
            return copyTensor(nsData, as: Int64.self)
        }
        // Some graphs emit floats for route ids.
        return copyTensor(nsData, as: Float.self).map { Int64($0.rounded()) }
    }

    /// ORT returns NSMutableData, which has no Swift `withUnsafeBytes`.
    private func copyTensor<T>(_ nsData: NSMutableData, as type: T.Type) -> [T] {
        let data = Data(referencing: nsData)
        let count = data.count / MemoryLayout<T>.stride
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: T.self).prefix(count))
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
