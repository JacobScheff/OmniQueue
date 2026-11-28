import Foundation

struct ForecastPark: Decodable, Identifiable, Equatable, Sendable {
    let slug: String
    let name: String
    let timezone: String
    let latitude: Double
    let longitude: Double

    var id: String { slug }
    var timeZone: TimeZone { TimeZone(identifier: timezone) ?? .gmt }

    func clock(_ timestamp: Int) -> String {
        Date(timeIntervalSince1970: TimeInterval(timestamp))
            .formatted(Date.FormatStyle(date: .omitted, time: .shortened, timeZone: timeZone))
    }
}

struct ForecastRide: Decodable, Identifiable, Sendable {
    struct Coordinate: Decodable, Sendable {
        let latitude: Double
        let longitude: Double
    }

    let id: String
    let name: String
    let coordinate: Coordinate?
}

enum ForecastWait: Decodable, Equatable, Sendable {
    case minutes(Int)
    case overflow

    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let minutes = try? value.decode(Int.self) {
            self = .minutes(minutes)
        } else if try value.decode(String.self) == "300+" {
            self = .overflow
        } else {
            throw DecodingError.dataCorruptedError(in: value, debugDescription: "Unrecognized wait time")
        }
    }

    var numericMinutes: Int? {
        if case .minutes(let minutes) = self { return minutes }
        return nil
    }
}

struct ForecastReading: Decodable, Identifiable, Sendable {
    let id: String
    let name: String
    let status: String
    let wait: ForecastWait?
    let p10: ForecastWait?
    let p90: ForecastWait?

    var label: String {
        switch wait {
        case .some(.minutes(let minutes)): return "\(minutes) min"
        case .some(.overflow): return "300+ min"
        case .none:
            switch status {
            case "DOWN": return "Down"
            case "CLOSED": return "Closed"
            case "REFURBISHMENT": return "Refurbishment"
            default: return "No wait reported"
            }
        }
    }

    var isComparable: Bool {
        wait != nil || ["DOWN", "CLOSED", "REFURBISHMENT"].contains(status)
    }

    func errorLabel(comparedWith actual: Self) -> String? {
        guard isComparable, actual.isComparable else { return nil }
        if let estimate = wait?.numericMinutes, let observed = actual.wait?.numericMinutes {
            let delta = estimate - observed
            return "\(delta > 0 ? "+" : "")\(delta) min error"
        }
        return status == actual.status && wait == actual.wait ? "Status matches" : "Wait or status differs"
    }
}

struct ForecastFrame: Decodable, Sendable {
    let timestamp: Int
    let source: String?
    let rides: [ForecastReading]

    func reading(_ id: String) -> ForecastReading? {
        rides.first { $0.id == id }
    }
}

struct ForecastDay: Decodable, Sendable {
    let date: String
    let isToday: Bool
    let state: String
    let timeline: [Int]
    let scheduled: [Int]
    let snapshots: [ForecastFrame]
    let latestObservation: Int?
    let forecastStarts: [Int]
    let defaultStart: Int?
    let modelAvailable: Bool
    let rides: [ForecastRide]
}

struct WaitForecast: Decodable, Sendable {
    let date: String
    let kind: String
    let start: Int
    let asOf: String
    let issuedAt: String?
    let modelName: String
    let split: String?
    let snapshots: [ForecastFrame]
}

struct ForecastDates: Decodable, Sendable {
    struct Day: Decodable, Identifiable, Sendable {
        let date: String
        let complete: Bool

        var id: String { date }
    }

    let dates: [Day]
    let next: String?
}

actor ForecastClient {
    enum ClientError: LocalizedError {
        case unconfigured
        case requestFailed(String, Int)

        var errorDescription: String? {
            switch self {
            case .unconfigured: return "Forecasts are unavailable in this build."
            case .requestFailed(let message, _): return message
            }
        }
    }

    private struct Parks: Decodable {
        let parks: [ForecastPark]
    }

    private struct ServerError: Decodable {
        let error: String
    }

    private let baseURL: URL?
    private let session: URLSession
    private let decoder: JSONDecoder

    init() {
        let value = Bundle.main.object(forInfoDictionaryKey: "OmniQueueForecastURL") as? String
        let url = value.flatMap(URL.init(string:))
        baseURL = url?.scheme == "https" && url?.host != nil ? url : nil
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 35
        session = URLSession(configuration: configuration)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder
    }

    func parks() async throws -> [ForecastPark] {
        let result: Parks = try await get(["v1", "parks"])
        return result.parks
    }

    func day(park: String, date: String?, etag: String?) async throws -> (ForecastDay, String?)? {
        let (data, response) = try await request(["v1", "parks", park, "day"],
                                                 query: [("date", date)], etag: etag)
        guard response.statusCode != 304 else { return nil }
        return (try decoder.decode(ForecastDay.self, from: data), response.value(forHTTPHeaderField: "ETag"))
    }

    func dates(park: String, before: String?) async throws -> ForecastDates {
        try await get(["v1", "parks", park, "dates"], query: [("before", before)])
    }

    func forecast(park: String, date: String, start: Int?) async throws -> WaitForecast {
        let path = ["v1", "parks", park, "forecast"]
        let query = [("date", date as String?), ("start", start.map { String($0) })]
        for delay in [1, 2, 3] {
            do {
                return try await get(path, query: query)
            } catch ClientError.requestFailed(_, 503) {
                try await Task.sleep(for: .seconds(delay))
            }
        }
        return try await get(path, query: query)
    }

    private func get<Value: Decodable>(_ path: [String], query: [(String, String?)] = []) async throws -> Value {
        let (data, _) = try await request(path, query: query)
        return try decoder.decode(Value.self, from: data)
    }

    private func request(_ path: [String], query: [(String, String?)] = [],
                         etag: String? = nil) async throws -> (Data, HTTPURLResponse) {
        guard let baseURL else { throw ClientError.unconfigured }
        let endpoint = path.reduce(baseURL) { $0.appendingPathComponent($1) }
        guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            throw ClientError.unconfigured
        }
        components.queryItems = query.compactMap { name, value in
            value.map { URLQueryItem(name: name, value: $0) }
        }
        guard let url = components.url else { throw ClientError.unconfigured }
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(etag, forHTTPHeaderField: "If-None-Match")
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ClientError.requestFailed("Forecasts are temporarily unavailable.", 0)
        }
        guard (200..<300).contains(http.statusCode) || (http.statusCode == 304 && etag != nil) else {
            let message = (try? decoder.decode(ServerError.self, from: data))?.error
            throw ClientError.requestFailed(message ?? "Forecasts are temporarily unavailable.", http.statusCode)
        }
        return (data, http)
    }
}
