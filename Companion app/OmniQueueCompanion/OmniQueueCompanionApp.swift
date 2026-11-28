import SwiftUI

@main
struct OmniQueueCompanionApp: App {
    @State private var session: AppSession? = try? AppSession()

    var body: some Scene {
        WindowGroup {
            if let session {
                RootView()
                    .environment(session)
                    .preferredColorScheme(session.appearance.colorScheme)
                    .onAppear { session.boot() }
            } else {
                ContentUnavailableView("Planner unavailable", systemImage: "list.bullet.clipboard",
                                       description: Text("Couldn’t load the bundled Disneyland park data."))
            }
        }
    }
}
