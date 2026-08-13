import SwiftUI

@main
struct OmniQueueCompanionApp: App {
    @State private var session: AppSession? = try? AppSession()

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
                session?.boot()
            }
        }
    }
}

private struct BootFailureView: View {
    var body: some View {
        ZStack {
            TicketInk.paper(blend: 1).ignoresSafeArea()
            VStack(spacing: 18) {
                PaperclipMark()
                Text("OmniQueue")
                    .font(TicketType.display)
                    .foregroundStyle(TicketInk.ink(blend: 1))
                VStack(alignment: .leading, spacing: 8) {
                    Text("VOID")
                        .font(TicketType.mono)
                        .tracking(1.6)
                        .foregroundStyle(TicketInk.oxblood)
                    Text("Couldn’t load the bundled park data or model.")
                        .font(TicketType.body)
                        .foregroundStyle(TicketInk.ink(blend: 1))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.leading, TicketLayout.leading(16))
                .padding(.trailing, 16)
                .padding(.vertical, 16)
                .background { TicketStock(corner: 20, stubWidth: 16) }
            }
            .padding(28)
        }
        .environment(\.themeBlend, 1)
    }
}
