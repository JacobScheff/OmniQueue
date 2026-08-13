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
    var isBooting = true
    var isRefreshingWaits = false
    var isPlanning = false
    var errorMessage: String?
    var selectedTab: Tab = .plan
    var showDisclaimer = false
    /// Destination appearance. System chrome follows this immediately; `themeBlend` eases to it.
    var usesLightTheme = false
    /// 0 = dark, 1 = light. Animate this so palette colors ease between modes.
    var themeBlend: CGFloat = 0
    /// Basic = Skip/Low/Medium/High. Advanced = 0–100 per ride.
    var usesAdvancedPrefs = false

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
    private var didBoot = false
    private let disclaimerKey = "omniqueue.companion.seenDisclaimer"
    private let themeKey = "omniqueue.companion.theme"
    private let prefModeKey = "omniqueue.companion.prefMode"

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
        if UserDefaults.standard.string(forKey: themeKey) == "light" {
            usesLightTheme = true
            themeBlend = 1
        }
        usesAdvancedPrefs = UserDefaults.standard.string(forKey: prefModeKey) == "advanced"
        if usesAdvancedPrefs, state.fillMissingScores() {
            persist()
        }
        showDisclaimer = !UserDefaults.standard.bool(forKey: disclaimerKey)
    }

    var canUndo: Bool { !past.isEmpty }
    var canRedo: Bool { !future.isEmpty }
    var busy: Bool { isBooting || isRefreshingWaits || isPlanning }

    /// True once ThemeParks.wiki has given at least one real attraction status.
    var hasLiveWaits: Bool {
        board.rides.contains { $0.status.uppercased() != "UNKNOWN" }
    }

    var waitBoardSummary: String {
        if let raw = board.error, !raw.isEmpty {
            return Self.friendlyWaitError(raw)
        }
        return "ThemeParks.wiki didn’t send a wait board. Check the connection and try again."
    }

    var locationName: String {
        catalog.location(for: state.location)?.name ?? "Entrance"
    }

    func boot() {
        guard !didBoot else { return }
        didBoot = true
        Task { await bootOnce() }
    }

    private func bootOnce() async {
        isBooting = true
        async let model = loadModelOnly()
        async let fetched = waits.board(force: true)
        do {
            recommender = try await model
        } catch {
            errorMessage = error.localizedDescription
        }
        board = await fetched
        if hasLiveWaits {
            await recompute()
        } else {
            recommendation = nil
        }
        withAnimation(.easeInOut(duration: 0.45)) {
            isBooting = false
        }
    }

    private func loadModelOnly() async throws -> OnnxRecommender {
        try await Task.detached(priority: .userInitiated) { [catalog] in
            try OnnxRecommender(catalog: catalog)
        }.value
    }

    func acknowledgeDisclaimer() {
        UserDefaults.standard.set(true, forKey: disclaimerKey)
        showDisclaimer = false
    }

    var preferredScheme: ColorScheme {
        usesLightTheme ? .light : .dark
    }

    func toggleTheme() {
        usesLightTheme.toggle()
        UserDefaults.standard.set(usesLightTheme ? "light" : "dark", forKey: themeKey)
        withAnimation(.easeInOut(duration: 0.35)) {
            themeBlend = usesLightTheme ? 1 : 0
        }
    }

    func setAdvancedPrefs(_ on: Bool) {
        if on {
            var next = state
            if next.fillMissingScores() {
                state = next
                persist()
                schedulePlan()
            }
        }
        guard usesAdvancedPrefs != on else { return }
        UserDefaults.standard.set(on ? "advanced" : "basic", forKey: prefModeKey)
        withAnimation(.spring(duration: 0.55, bounce: 0.22)) {
            usesAdvancedPrefs = on
        }
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
        var next = GuestState.default(catalog: catalog)
        if usesAdvancedPrefs { next.fillMissingScores() }
        past.removeAll()
        future.removeAll()
        state = next
        forcedPick = nil
        persist()
        schedulePlan()
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
        isRefreshingWaits = true
        let started = ContinuousClock.now
        defer { isRefreshingWaits = false }
        let next = await waits.board(force: force)
        board = next
        if hasLiveWaits {
            await recompute()
        } else {
            recommendation = nil
        }
        // Keep the refresh glyph spinning for at least one turn so a fast
        // ThemeParks.wiki response doesn't look like a dead control.
        let minimum: Duration = .milliseconds(750)
        let elapsed = ContinuousClock.now - started
        if elapsed < minimum {
            try? await Task.sleep(for: minimum - elapsed)
        }
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
        guard hasLiveWaits else {
            recommendation = nil
            return
        }
        isPlanning = true
        errorMessage = nil
        do {
            let built = try ObservationBuilder.build(catalog: catalog, state: state, board: board)
            let rec = try recommender.recommend(built: built, board: board, force: forcedPick)
            recommendation = rec
        } catch {
            errorMessage = error.localizedDescription
        }
        isPlanning = false
    }

    private static func friendlyWaitError(_ raw: String) -> String {
        let lower = raw.lowercased()
        if lower.contains("offline") || lower.contains("notconnected") || lower.contains("network connection") {
            return "You’re offline, so wait times couldn’t load."
        }
        if lower.contains("timed out") || lower.contains("timeout") {
            return "The wait feed timed out. Try refresh in a moment."
        }
        return "Couldn’t load wait times from ThemeParks.wiki."
    }
}
