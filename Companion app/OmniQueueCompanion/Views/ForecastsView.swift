import Charts
import MapKit
import SwiftUI

struct ForecastsView: View {
    private struct DayRequest: Equatable {
        let park: String
        let date: String?
        let refresh: Int
    }

    private struct PredictionRequest: Equatable {
        let park: String
        let date: String
        let start: Int?
        let revision: String?
        let reload: Int
    }

    @AppStorage("omniqueue.forecast.park") private var parkID = "disneyland"
    @State private var client = ForecastClient()
    @State private var parks: [ForecastPark] = []
    @State private var day: ForecastDay?
    @State private var etag: String?
    @State private var forecast: WaitForecast?
    @State private var date: String?
    @State private var start: Int?
    @State private var refresh = 0
    @State private var retryForecast = 0
    @State private var index = 0.0
    @State private var loadingDay = false
    @State private var loadingForecast = false
    @State private var dayError: String?
    @State private var forecastError: String?
    @State private var camera: MapCameraPosition = .automatic
    @State private var aerial = true
    @State private var showHistory = false
    @State private var showStarts = false
    @State private var showRides = false
    @State private var query = ""
    @State private var ride: ForecastRide?

    private var park: ForecastPark? { parks.first { $0.slug == parkID } }
    private var dayRequest: DayRequest { DayRequest(park: parkID, date: date, refresh: refresh) }

    private var predictionRequest: PredictionRequest? {
        guard let day, (start != nil || day.defaultStart != nil || !day.isToday), !day.snapshots.isEmpty else {
            return nil
        }
        return PredictionRequest(park: parkID, date: day.date, start: start, revision: etag, reload: retryForecast)
    }

    private var tick: Int? {
        guard let timeline = day?.timeline, timeline.indices.contains(Int(index.rounded())) else { return nil }
        return timeline[Int(index.rounded())]
    }

    private var observedFrame: ForecastFrame? { day?.snapshots.first { $0.timestamp == tick } }
    private var predictionFrame: ForecastFrame? {
        guard let day, let tick, day.scheduled.contains(tick) else { return nil }
        return forecast?.snapshots.first { $0.timestamp == tick }
    }

    var body: some View {
        NavigationStack {
            Group {
                if let park {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            controls(park)
                            if let dayError {
                                Label(dayError, systemImage: "exclamationmark.circle")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            parkMap(park)
                            if let day {
                                timeline(park, day)
                            } else if loadingDay {
                                ProgressView("Loading recorded waits")
                                    .frame(maxWidth: .infinity)
                            }
                            Link("Powered by ThemeParks.wiki", destination: AppLinks.themeParksWiki)
                                .font(.footnote)
                        }
                        .padding()
                    }
                } else if let dayError {
                    ContentUnavailableView {
                        Label("Forecasts unavailable", systemImage: "clock.badge.exclamationmark")
                    } description: {
                        Text(dayError)
                    } actions: {
                        Button("Try Again") { refresh += 1 }
                            .buttonStyle(.bordered)
                    }
                } else {
                    ProgressView("Loading parks")
                }
            }
            .navigationTitle("Forecasts")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !parks.isEmpty {
                        Menu {
                            ForEach(parks) { option in
                                Button {
                                    if option.slug != parkID {
                                        selectDay(nil)
                                        parkID = option.slug
                                        recenter(option)
                                    }
                                } label: {
                                    if option.slug == parkID {
                                        Label(option.name, systemImage: "checkmark")
                                    } else {
                                        Text(option.name)
                                    }
                                }
                            }
                        } label: {
                            Label("Change park", systemImage: "mappin.and.ellipse")
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if park != nil {
                        Button { showHistory = true } label: {
                            Label("Previous operating days", systemImage: "calendar")
                        }
                    }
                }
            }
            .sheet(isPresented: $showHistory) {
                if let park {
                    ForecastHistory(client: client, park: park, selected: date) { selectDay($0) }
                }
            }
            .sheet(isPresented: $showStarts) {
                if let day, let park { startPicker(day, park: park) }
            }
            .sheet(isPresented: $showRides, onDismiss: { query = "" }) {
                NavigationStack {
                    List {
                        ForEach(visibleRides) { item in
                            NavigationLink {
                                if let day, let park { rideDetail(item, park: park, day: day) }
                            } label: {
                                rideRow(item)
                            }
                        }
                    }
                    .searchable(text: $query, prompt: "Find an attraction")
                    .navigationTitle("Attractions")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar { Button("Done") { showRides = false } }
                }
            }
            .sheet(item: $ride) { item in
                NavigationStack {
                    if let day, let park { rideDetail(item, park: park, day: day) }
                }
            }
        }
        .task(id: dayRequest) { await reloadDay(dayRequest) }
        .task(id: predictionRequest) { await reloadForecast(predictionRequest) }
    }

    private func controls(_ park: ForecastPark) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(park.name)
                .font(.title2.bold())
                .accessibilityAddTraits(.isHeader)
            Text(day.map { $0.isToday ? "Today · \(dateLabel($0.date, park: park))" : dateLabel($0.date, park: park) } ?? "Today")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if date == nil {
                Button {
                    refresh += 1
                } label: {
                    HStack(spacing: 8) {
                        if loadingDay { ProgressView().tint(.white) }
                        Label("Refresh Today’s Waits", systemImage: loadingDay ? "clock" : "arrow.clockwise")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(loadingDay)
            } else {
                Button("Back to Today", systemImage: "clock") { selectDay(nil) }
                    .buttonStyle(.bordered)
            }
            if let latest = day?.latestObservation {
                Text("Last recorded \(park.clock(latest)) · park time")
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func parkMap(_ park: ForecastPark) -> some View {
        Map(position: $camera, interactionModes: .all) {
            ForEach(day?.rides ?? []) { item in
                if let point = item.coordinate {
                    let prediction = predictionFrame?.reading(item.id)
                    let reading = prediction ?? observedFrame?.reading(item.id)
                    if let reading, reading.isComparable {
                        Annotation(item.name, coordinate: CLLocationCoordinate2D(latitude: point.latitude,
                                                                                  longitude: point.longitude)) {
                            Button {
                                ride = item
                            } label: {
                                Text((reading.wait?.numericMinutes).map { String($0) } ?? reading.label)
                                    .font(.caption2.weight(.semibold).monospacedDigit())
                                    .lineLimit(1)
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 5)
                                    .background(.regularMaterial, in: Capsule())
                                    .overlay {
                                        Capsule().strokeBorder(prediction == nil ? Color.primary.opacity(0.6) : Color.purple,
                                                               style: StrokeStyle(lineWidth: 1.5, dash: prediction == nil ? [] : [4, 2]))
                                    }
                                    .frame(minHeight: 44)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("\(item.name), \(reading.label), \(pinSource(predicted: prediction != nil))")
                        }
                    }
                }
            }
        }
        .mapStyle(aerial ? .imagery(elevation: .realistic) : .standard(elevation: .flat))
        .mapControls {
            MapCompass()
            MapScaleView()
        }
        .overlay(alignment: .topLeading) {
            HStack(spacing: 6) {
                Button { aerial.toggle() } label: {
                    Label(aerial ? "Street map" : "3D aerial map", systemImage: aerial ? "map" : "globe.americas")
                        .labelStyle(.iconOnly)
                }
                Button { recenter(park) } label: {
                    Label("Recenter park", systemImage: "scope")
                        .labelStyle(.iconOnly)
                }
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.circle)
            .padding(8)
        }
        .frame(height: 330)
        .clipShape(.rect(cornerRadius: 16))
        .accessibilityLabel("\(park.name) wait map")
    }

    private func timeline(_ park: ForecastPark, _ day: ForecastDay) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let tick {
                LabeledContent {
                    Text(park.clock(tick)).monospacedDigit()
                } label: {
                    Label(frameLabel(day: day, tick: tick), systemImage: predictionFrame == nil ? "clock" : "chart.line.uptrend.xyaxis")
                }
                Slider(value: $index, in: 0...Double(max(1, day.timeline.count - 1)), step: 1)
                    .disabled(day.timeline.count < 2)
                    .accessibilityLabel("Operating day timeline")
                    .accessibilityValue("\(park.clock(tick)), \(frameLabel(day: day, tick: tick))")
            }
            if day.state == "closed" || day.state == "waiting" {
                Text(day.state == "closed" ? "The park has no scheduled hours today." : "Waiting for the first recorded waits.")
                    .foregroundStyle(.secondary)
            }
            ViewThatFits(in: .horizontal) {
                HStack {
                    startButton(day, park: park)
                    Spacer()
                    ridesButton(day)
                }
                VStack(alignment: .leading) {
                    startButton(day, park: park)
                    ridesButton(day)
                }
            }
            if loadingForecast {
                ProgressView("Generating forecast")
                    .font(.footnote)
            } else if let forecastError {
                Text(forecastError).font(.footnote).foregroundStyle(.secondary)
                Button("Retry Forecast") { retryForecast += 1 }
                    .font(.footnote)
            } else if let forecast {
                Text(forecastDescription(forecast, park: park))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if day.state == "recorded" {
                Text(day.modelAvailable ? "No forecast is available from the latest snapshot." : "Recorded waits are available. This park’s model is unpublished.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func frameLabel(day: ForecastDay, tick: Int) -> String {
        if predictionFrame != nil { return forecast?.kind == "replay" ? "Replay forecast" : "Forecast" }
        if observedFrame?.source == "observed" { return "Recorded" }
        if observedFrame?.source == "interpolated" { return "Interpolated" }
        return day.scheduled.contains(tick) ? "No observation" : "Scheduled break"
    }

    private func pinSource(predicted: Bool) -> String {
        if predicted { return "Forecast" }
        return observedFrame?.source == "interpolated" ? "Interpolated" : "Recorded"
    }

    private func startButton(_ day: ForecastDay, park: ForecastPark) -> some View {
        Button {
            showStarts = true
        } label: {
            Label((start ?? forecast?.start ?? day.defaultStart).map { "Forecast from \(park.clock($0))" } ?? "Forecast Start",
                  systemImage: "clock.arrow.circlepath")
        }
        .buttonStyle(.bordered)
        .disabled(day.forecastStarts.isEmpty)
    }

    private func ridesButton(_ day: ForecastDay) -> some View {
        Button("Attractions", systemImage: "list.bullet") { showRides = true }
            .buttonStyle(.bordered)
            .disabled(day.rides.isEmpty)
    }

    private func startPicker(_ day: ForecastDay, park: ForecastPark) -> some View {
        NavigationStack {
            List {
                Button {
                    chooseStart(nil)
                } label: {
                    Label("Automatic", systemImage: start == nil ? "checkmark" : "clock")
                }
                Section("Available times · park time") {
                    ForEach(day.forecastStarts.reversed(), id: \.self) { choice in
                        Button { chooseStart(choice) } label: {
                            HStack {
                                Text(park.clock(choice)).monospacedDigit()
                                Spacer()
                                if start == choice { Image(systemName: "checkmark") }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Forecast Start")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { showStarts = false } }
        }
    }

    private var visibleRides: [ForecastRide] {
        (day?.rides ?? []).filter { query.isEmpty || $0.name.localizedCaseInsensitiveContains(query) }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    private func rideRow(_ item: ForecastRide) -> some View {
        let prediction = predictionFrame?.reading(item.id)
        let actual = observedFrame?.reading(item.id)
        return HStack {
            Text(item.name)
            Spacer()
            if let prediction {
                VStack(alignment: .trailing) {
                    Text(prediction.label).foregroundStyle(.purple)
                    if observedFrame?.source == "observed", let actual, let error = prediction.errorLabel(comparedWith: actual) {
                        Text(error).font(.caption).foregroundStyle(.secondary)
                    }
                }
            } else {
                Text(actual?.label ?? "—").foregroundStyle(.secondary)
            }
        }
    }

    private func rideDetail(_ item: ForecastRide, park: ForecastPark, day: ForecastDay) -> some View {
        ForecastRideDetail(item: item, park: park, day: day, forecast: forecast, selected: tick)
    }

    private func recenter(_ park: ForecastPark) {
        let center = CLLocationCoordinate2D(latitude: park.latitude, longitude: park.longitude)
        camera = .camera(MapCamera(centerCoordinate: center, distance: 1250, heading: 0, pitch: aerial ? 35 : 0))
    }

    private func selectDay(_ date: String?) {
        guard self.date != date || day != nil else { return }
        day = nil
        etag = nil
        forecast = nil
        start = nil
        index = 0
        dayError = nil
        forecastError = nil
        self.date = date
        refresh += 1
    }

    private func chooseStart(_ timestamp: Int?) {
        if start != timestamp {
            start = timestamp
            forecast = nil
            forecastError = nil
        }
        if let timestamp, let newIndex = day?.timeline.firstIndex(of: timestamp) {
            index = Double(newIndex)
        }
        showStarts = false
    }

    @MainActor
    private func reloadDay(_ request: DayRequest) async {
        loadingDay = true
        dayError = nil
        defer { if dayRequest == request { loadingDay = false } }
        do {
            if parks.isEmpty {
                let options = try await client.parks()
                try Task.checkCancellation()
                guard dayRequest == request else { return }
                parks = options
                if let park { recenter(park) }
                else if let first = options.first {
                    parkID = first.slug
                    recenter(first)
                    return
                }
            }
            let followLatest = tick == nil || tick == day?.latestObservation
            let priorTick = tick
            let response = try await client.day(park: request.park, date: request.date, etag: etag)
            try Task.checkCancellation()
            guard dayRequest == request else { return }
            if let (newDay, newETag) = response {
                if day?.date != newDay.date {
                    forecast = nil
                    start = nil
                }
                day = newDay
                etag = newETag
                let target = followLatest ? newDay.latestObservation : priorTick
                index = Double(target.flatMap { newDay.timeline.firstIndex(of: $0) } ?? 0)
            } else if forecastError != nil {
                retryForecast += 1
            }
        } catch {
            guard !Task.isCancelled, dayRequest == request else { return }
            dayError = error.localizedDescription
        }
    }

    @MainActor
    private func reloadForecast(_ request: PredictionRequest?) async {
        loadingForecast = request != nil
        forecastError = nil
        guard let request else { forecast = nil; return }
        defer { if predictionRequest == request { loadingForecast = false } }
        do {
            let value = try await client.forecast(park: request.park, date: request.date, start: request.start)
            try Task.checkCancellation()
            guard request == predictionRequest else { return }
            forecast = value
        } catch {
            guard !Task.isCancelled, request == predictionRequest else { return }
            forecastError = error.localizedDescription
        }
    }
}

private func dateLabel(_ key: String, park: ForecastPark) -> String {
    let parser = DateFormatter()
    parser.locale = Locale(identifier: "en_US_POSIX")
    parser.timeZone = park.timeZone
    parser.dateFormat = "yyyy-MM-dd"
    guard let date = parser.date(from: key) else { return key }
    return date.formatted(Date.FormatStyle(date: .abbreviated, time: .omitted, timeZone: park.timeZone))
}

private func forecastDescription(_ forecast: WaitForecast, park: ForecastPark) -> String {
    var description = forecast.kind == "replay" ? "Replay forecast" : "Issued forecast"
    description += " · \(forecast.modelName)"
    if forecast.split == "train" { description += " · Model trained on this date" }
    if forecast.split == "validation" { description += " · Model validated on this date" }
    let formatter = ISO8601DateFormatter()
    if let inputTime = formatter.date(from: forecast.asOf) {
        description += " · Input through \(park.clock(Int(inputTime.timeIntervalSince1970)))"
    }
    if let issuedAt = forecast.issuedAt {
        if let time = formatter.date(from: issuedAt) {
            description += " · Issued \(park.clock(Int(time.timeIntervalSince1970)))"
        }
    }
    return description
}

private struct ForecastHistory: View {
    @Environment(\.dismiss) private var dismiss
    let client: ForecastClient
    let park: ForecastPark
    let selected: String?
    let select: (String?) -> Void
    @State private var dates: [ForecastDates.Day] = []
    @State private var next: String?
    @State private var loading = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List {
                Button { open(nil) } label: {
                    Label("Today", systemImage: selected == nil ? "checkmark" : "clock")
                }
                Section("Recorded operating days") {
                    ForEach(dates) { day in
                        Button { open(day.date) } label: {
                            HStack {
                                Text(dateLabel(day.date, park: park))
                                Spacer()
                                if !day.complete { Text("Partial").font(.caption).foregroundStyle(.secondary) }
                                if selected == day.date { Image(systemName: "checkmark") }
                            }
                        }
                    }
                    if loading {
                        ProgressView()
                    } else if let error {
                        Text(error).foregroundStyle(.secondary)
                        Button("Try Again") { Task { await load() } }
                    } else if next != nil {
                        Button("Earlier Dates") { Task { await load() } }
                    } else if dates.isEmpty {
                        Text("No earlier dates available").foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
        .task { await load() }
    }

    private func open(_ date: String?) {
        select(date)
        dismiss()
    }

    @MainActor
    private func load() async {
        guard !loading else { return }
        loading = true
        error = nil
        defer { loading = false }
        do {
            let page = try await client.dates(park: park.slug, before: next)
            let ids = Set(dates.map(\.id))
            dates.append(contentsOf: page.dates.filter { !ids.contains($0.id) })
            next = page.next
        } catch {
            guard !Task.isCancelled else { return }
            self.error = error.localizedDescription
        }
    }
}

private struct ForecastRideDetail: View {
    private struct Point {
        let tick: Int
        let reading: ForecastReading
        let segment: Int

        var date: Date { Date(timeIntervalSince1970: TimeInterval(tick)) }
    }

    let item: ForecastRide
    let park: ForecastPark
    let day: ForecastDay
    let forecast: WaitForecast?
    let selected: Int?
    @State private var inspected: Date?

    private var tick: Int? {
        if let inspected {
            return day.timeline.min { abs(Double($0) - inspected.timeIntervalSince1970) < abs(Double($1) - inspected.timeIntervalSince1970) }
        }
        return selected
    }

    var body: some View {
        let recorded = points(day.snapshots.filter { $0.source == "observed" })
        let predicted = points(forecast?.snapshots ?? [])
        List {
            Section("Wait time · park time") {
                if recorded.isEmpty && predicted.isEmpty {
                    Text("No numeric wait times available for this attraction.")
                        .foregroundStyle(.secondary)
                } else {
                    chart(recorded: recorded, predicted: predicted)
                    HStack(spacing: 16) {
                        Label("Recorded", systemImage: "line.diagonal")
                        if !predicted.isEmpty { Label("Forecast", systemImage: "waveform.path").foregroundStyle(.purple) }
                    }
                    .font(.caption)
                }
            }
            if let tick {
                comparison(at: tick)
            }
        }
        .navigationTitle(item.name)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func chart(recorded: [Point], predicted: [Point]) -> some View {
        Chart {
            ForEach(recorded, id: \.tick) { point in
                if let wait = point.reading.wait?.numericMinutes {
                    LineMark(x: .value("Time", point.date), y: .value("Recorded wait", wait),
                             series: .value("Series", "recorded-\(point.segment)"))
                        .foregroundStyle(Color.accentColor)
                        .symbol(.circle)
                }
            }
            ForEach(predicted, id: \.tick) { point in
                if let wait = point.reading.wait?.numericMinutes {
                    if let low = point.reading.p10?.numericMinutes, let high = point.reading.p90?.numericMinutes {
                        AreaMark(x: .value("Time", point.date), yStart: .value("Low", low), yEnd: .value("High", high),
                                 series: .value("Range", point.segment))
                            .foregroundStyle(Color.purple.opacity(0.13))
                    }
                    LineMark(x: .value("Time", point.date), y: .value("Forecast wait", wait),
                             series: .value("Series", "forecast-\(point.segment)"))
                        .foregroundStyle(Color.purple)
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 3]))
                        .symbol(.diamond)
                }
            }
            if let tick {
                RuleMark(x: .value("Selected time", Date(timeIntervalSince1970: TimeInterval(tick))))
                    .foregroundStyle(Color.secondary.opacity(0.5))
            }
        }
        .chartXAxis {
            AxisMarks { axis in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let date = axis.as(Date.self) { Text(park.clock(Int(date.timeIntervalSince1970))) }
                }
            }
        }
        .chartXSelection(value: $inspected)
        .frame(height: 220)
    }

    private func comparison(at tick: Int) -> some View {
        let actualFrame = day.snapshots.first { $0.timestamp == tick }
        let actual = actualFrame?.reading(item.id)
        let predicted = forecast?.snapshots.first { $0.timestamp == tick && day.scheduled.contains(tick) }?.reading(item.id)
        return Section(park.clock(tick)) {
            if let actual {
                LabeledContent(actualFrame?.source == "interpolated" ? "Interpolated" : "Recorded", value: actual.label)
            } else {
                LabeledContent("Recorded", value: "No observation")
            }
            if let predicted {
                LabeledContent(forecast?.kind == "replay" ? "Replay forecast" : "Forecast", value: predicted.label)
                if let low = predicted.p10?.numericMinutes, let high = predicted.p90?.numericMinutes {
                    LabeledContent("Forecast range", value: "\(low)–\(high) min")
                }
                if actualFrame?.source == "observed", let actual, let error = predicted.errorLabel(comparedWith: actual) {
                    LabeledContent("Comparison", value: error)
                }
            }
        }
    }

    private func points(_ frames: [ForecastFrame]) -> [Point] {
        let operating = Set(day.scheduled)
        var last: Int?
        var segment = 0
        return frames.compactMap { frame in
            guard operating.contains(frame.timestamp), let reading = frame.reading(item.id),
                  reading.wait?.numericMinutes != nil else { return nil }
            if last.map({ frame.timestamp - $0 != 300 }) == true { segment += 1 }
            last = frame.timestamp
            return Point(tick: frame.timestamp, reading: reading, segment: segment)
        }
    }
}
