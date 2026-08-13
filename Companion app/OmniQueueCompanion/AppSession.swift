import Foundation
import SwiftUI

@MainActor
@Observable
final class AppSession {
    let catalog: ParkCatalog
    var state: GuestState
    var past: [GuestState] = []
    var future: [GuestState] = []
    var board: WaitBoard
    var recommendation: Recommendation?
    var forcedPick: ForcedPick?
    var busy = false
    var errorMessage: String?
    var selectedTab: Tab = .plan
    var showDisclaimer = false
    var preferredScheme: ColorScheme? = .dark

    enum Tab: String, CaseIterable, Identifiable {
        case plan, rides, me
        var id: String { rawValue }
        var title: String {
            switch self {
            case .plan: return "Plan"
            case .rides: return "Rides"
            case .me: return "Me"
            }
        }

        var index: Int {
            Self.allCases.firstIndex(of: self) ?? 0
        }

        static func from(index: Int) -> Tab {
            let cases = Self.allCases
            return cases[min(max(index, 0), cases.count - 1)]
        }
    }

    private let store = StateStore()
    private let waits: WaitTimeService
    private var recommender: OnnxRecommender?
    private var planTask: Task<Void, Never>?
    private let disclaimerKey = "omniqueue.companion.seenDisclaimer"
    private let themeKey = "omniqueue.companion.theme"

    init() throws {
        let catalog = try ParkCatalog.loadFromBundle()
        self.catalog = catalog
        let loaded = StateStore().load(catalog: catalog)
        self.state = loaded.0
        self.past = loaded.1
        self.future = loaded.2
        self.board = WaitBoard(
            rides: catalog.rides.map {
                RideWait(rideId: $0.id, name: $0.name, waitMin: nil, status: "UNKNOWN", open: false, entityId: $0.entityId)
            },
            fetchedAt: .distantPast,
            error: nil
        )
        self.waits = WaitTimeService(catalog: catalog)
        if let raw = UserDefaults.standard.string(forKey: themeKey) {
            preferredScheme = raw == "light" ? .light : .dark
        }
        showDisclaimer = !UserDefaults.standard.bool(forKey: disclaimerKey)
    }

    var canUndo: Bool { !past.isEmpty }
    var canRedo: Bool { !future.isEmpty }

    var locationName: String {
        catalog.location(for: state.location)?.name ?? "Entrance"
    }

    func boot() {
        Task { await loadModel() }
        Task { await refreshWaits(force: true) }
    }

    func acknowledgeDisclaimer() {
        UserDefaults.standard.set(true, forKey: disclaimerKey)
        showDisclaimer = false
    }

    func toggleTheme() {
        let next: ColorScheme = (preferredScheme == .light) ? .dark : .light
        preferredScheme = next
        UserDefaults.standard.set(next == .light ? "light" : "dark", forKey: themeKey)
    }

    func commit(_ next: GuestState) {
        past.append(state)
        if past.count > 50 { past.removeFirst(past.count - 50) }
        future.removeAll()
        state = next
        persist()
        schedulePlan()
    }

    func undo() {
        guard let prev = past.popLast() else { return }
        future.insert(state, at: 0)
        state = prev
        forcedPick = nil
        persist()
        schedulePlan()
    }

    func redo() {
        guard !future.isEmpty else { return }
        let next = future.removeFirst()
        past.append(state)
        state = next
        forcedPick = nil
        persist()
        schedulePlan()
    }

    func reset() {
        commit(GuestState.default(catalog: catalog))
        forcedPick = nil
    }

    func setForce(_ pick: ForcedPick) {
        if forcedPick == pick {
            forcedPick = nil
        } else {
            forcedPick = pick
        }
        schedulePlan()
    }

    func clearForce() {
        forcedPick = nil
        schedulePlan()
    }

    func refreshWaits(force: Bool) async {
        busy = true
        let next = await waits.board(force: force)
        board = next
        await recompute()
        busy = false
    }

    private func loadModel() async {
        busy = true
        do {
            let rec = try await Task.detached(priority: .userInitiated) { [catalog] in
                try OnnxRecommender(catalog: catalog)
            }.value
            recommender = rec
            await recompute()
        } catch {
            errorMessage = error.localizedDescription
        }
        busy = false
    }

    private func persist() {
        store.save(state: state, past: past, future: future)
    }

    private func schedulePlan() {
        planTask?.cancel()
        planTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(140))
            guard let self, !Task.isCancelled else { return }
            await self.recompute()
        }
    }

    private func recompute() async {
        guard let recommender else { return }
        busy = true
        errorMessage = nil
        do {
            let built = try ObservationBuilder.build(catalog: catalog, state: state, board: board)
            let rec = try recommender.recommend(built: built, board: board, force: forcedPick)
            recommendation = rec
        } catch {
            errorMessage = error.localizedDescription
        }
        busy = false
    }
}
