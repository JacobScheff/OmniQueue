import Foundation

actor WaitTimeService {
    private let catalog: ParkCatalog
    private var cached: WaitBoard?
    private var cachedAt: Date = .distantPast
    private let ttl: TimeInterval = 45

    init(catalog: ParkCatalog) {
        self.catalog = catalog
    }

    func board(force: Bool = false) async -> WaitBoard {
        if !force, let cached, Date().timeIntervalSince(cachedAt) < ttl, cached.error == nil {
            return cached
        }
        do {
            let fresh = try await fetch()
            cached = fresh
            cachedAt = Date()
            return fresh
        } catch {
            if let cached {
                var stale = cached
                stale.error = "stale cache; refresh failed: \(error.localizedDescription)"
                return stale
            }
            let empty = emptyBoard(error: error.localizedDescription)
            cached = empty
            cachedAt = Date()
            return empty
        }
    }

    private func emptyBoard(error: String?) -> WaitBoard {
        WaitBoard(
            rides: catalog.rides.map {
                RideWait(rideId: $0.id, name: $0.name, waitMin: nil, status: "UNKNOWN", open: false, entityId: $0.entityId)
            },
            fetchedAt: Date(),
            error: error
        )
    }

    private func fetch() async throws -> WaitBoard {
        let url = URL(string: "\(catalog.waitApiBase)/entity/\(catalog.parkEntityId)/live")!
        var request = URLRequest(url: url, timeoutInterval: 12)
        request.setValue("OmniQueueCompanion/1.0 (iOS)", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let liveRows = payload?["liveData"] as? [[String: Any]] ?? []

        let byEntity = Dictionary(uniqueKeysWithValues: catalog.rides.compactMap { ride in
            ride.entityId.map { ($0, ride.id) }
        })
        let nameIndex = buildNameIndex()

        var byId: [Int: RideWait] = [:]
        for row in liveRows {
            guard (row["entityType"] as? String) == "ATTRACTION" else { continue }
            let name = row["name"] as? String ?? ""
            let entityId = row["id"] as? String
            guard let rideId = resolve(entityId: entityId, name: name, byEntity: byEntity, nameIndex: nameIndex) else {
                continue
            }
            let status = ((row["status"] as? String) ?? "UNKNOWN").uppercased()
            let queue = row["queue"] as? [String: Any]
            let standby = queue?["STANDBY"] as? [String: Any]
            let wait: Double?
            if let value = standby?["waitTime"] as? NSNumber {
                wait = value.doubleValue
            } else {
                wait = nil
            }
            let open = status == "OPERATING"
            byId[rideId] = RideWait(
                rideId: rideId,
                name: catalog.rides[rideId].name,
                waitMin: open ? wait : nil,
                status: status,
                open: open,
                entityId: entityId
            )
        }

        let rides = catalog.rides.map { ride -> RideWait in
            if let live = byId[ride.id] { return live }
            return RideWait(rideId: ride.id, name: ride.name, waitMin: nil, status: "UNKNOWN", open: false, entityId: ride.entityId)
        }
        return WaitBoard(rides: rides, fetchedAt: Date(), error: nil)
    }

    private func buildNameIndex() -> [String: Int] {
        var index: [String: Int] = [:]
        for ride in catalog.rides {
            index[normalize(ride.name)] = ride.id
        }
        for (alias, idx) in catalog.nameAliases {
            index[normalize(alias)] = idx
        }
        return index
    }

    private func resolve(entityId: String?, name: String, byEntity: [String: Int], nameIndex: [String: Int]) -> Int? {
        if let entityId, let id = byEntity[entityId] { return id }
        let key = normalize(name)
        if let id = nameIndex[key] { return id }
        for (cand, idx) in nameIndex where cand.contains(key) || key.contains(cand) {
            return idx
        }
        return nil
    }

    private func normalize(_ name: String) -> String {
        var text = name.folding(options: .diacriticInsensitive, locale: .current).lowercased()
        text = text.replacingOccurrences(of: "&", with: " and ")
        text = text.replacingOccurrences(of: "™", with: "")
        text = text.replacingOccurrences(of: "®", with: "")
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        text = text.unicodeScalars.map { allowed.contains($0) ? Character($0) : " " }.map(String.init).joined()
        while text.contains("  ") { text = text.replacingOccurrences(of: "  ", with: " ") }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
