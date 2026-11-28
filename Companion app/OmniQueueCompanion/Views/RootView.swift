import SwiftUI

struct RootView: View {
    @Environment(AppSession.self) private var session

    var body: some View {
        TabView {
            NavigationStack {
                PlanView()
            }
            .tabItem { Label("Planner", systemImage: "list.bullet.clipboard") }

            ForecastsView()
                .tabItem { Label("Forecasts", systemImage: "chart.line.uptrend.xyaxis") }

            NavigationStack {
                MeView()
            }
            .tabItem { Label("Me", systemImage: "person.crop.circle") }
        }
        .sheet(isPresented: Binding(
            get: { session.showDisclaimer },
            set: { if !$0 { session.acknowledgeDisclaimer() } }
        )) {
            DisclaimerSheet()
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
    }
}
