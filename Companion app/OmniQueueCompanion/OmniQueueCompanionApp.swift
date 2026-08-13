import SwiftUI

@main
struct OmniQueueCompanionApp: App {
    @State private var session: AppSession?

    var body: some Scene {
        WindowGroup {
            Group {
                if let session {
                    ThemeBlendHost(blend: session.themeBlend) {
                        RootView()
                            .environment(session)
                            .preferredColorScheme(session.preferredScheme)
                    }
                } else {
                    BootFailureView()
                }
            }
            .onAppear {
                if session == nil {
                    session = try? AppSession()
                    session?.boot()
                }
            }
        }
    }
}

private struct BootFailureView: View {
    var body: some View {
        ZStack {
            TicketInk.paper(for: .light).ignoresSafeArea()
            VStack(spacing: 12) {
                Text("OmniQueue")
                    .font(TicketType.display)
                Text("Couldn’t load the bundled park data or model.")
                    .font(TicketType.body)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }
            .padding(28)
        }
    }
}
