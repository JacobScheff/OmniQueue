import SwiftUI

struct DisclaimerSheet: View {
    @Environment(AppSession.self) private var session

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("OmniQueue is an independent app. Disney and the other park operators do not sponsor or endorse it.")
                    Text("Wait times and forecasts are unofficial estimates. Check the park’s own information for operating hours and attraction availability.")
                    Link("Powered by ThemeParks.wiki", destination: AppLinks.themeParksWiki)
                    Button {
                        session.acknowledgeDisclaimer()
                    } label: {
                        Text("Continue").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
            }
            .navigationTitle("Welcome to OmniQueue")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
