import Foundation

/// Forgiving search for ride and land names. Punctuation, word order, prefixes,
/// dropped letters, and small typos still match.
enum FuzzyMatch {
    static func matches(_ query: String, in texts: String...) -> Bool {
        let needle = fold(query)
        if needle.isEmpty { return true }
        let hay = fold(texts.joined(separator: " "))
        if hay.isEmpty { return false }
        if hay.contains(needle) { return true }

        let compactQ = needle.replacingOccurrences(of: " ", with: "")
        let compactH = hay.replacingOccurrences(of: " ", with: "")
        if compactH.contains(compactQ) { return true }

        let qTokens = needle.split(separator: " ").map(String.init)
        let hTokens = hay.split(separator: " ").map(String.init)
        if qTokens.isEmpty { return true }

        let initials = hTokens.compactMap(\.first).map(String.init).joined()
        let skipped: Set<String> = ["a", "an", "and", "of", "the", "in", "n"]
        let shortInitials = hTokens
            .filter { !skipped.contains($0) }
            .compactMap(\.first)
            .map(String.init)
            .joined()
        if compactQ.count >= 2, initials.contains(compactQ) || shortInitials.contains(compactQ) {
            return true
        }

        return qTokens.allSatisfy { q in
            hTokens.contains { h in tokenMatches(q, h) }
        }
    }

    private static func tokenMatches(_ query: String, _ hay: String) -> Bool {
        if hay.contains(query) { return true }
        if query.count >= 3, query.contains(hay), hay.count >= 3 { return true }
        if query.count >= 3, isSubsequence(query, of: hay) { return true }
        let tol = tolerance(for: query)
        guard tol > 0, abs(query.count - hay.count) <= tol else { return false }
        return distance(query, hay) <= tol
    }

    private static func tolerance(for query: String) -> Int {
        if query.count <= 3 { return 0 }
        if query.count <= 6 { return 1 }
        return 2
    }

    private static func isSubsequence(_ query: String, of hay: String) -> Bool {
        var rest = hay[...]
        for ch in query {
            guard let i = rest.firstIndex(of: ch) else { return false }
            rest = rest[rest.index(after: i)...]
        }
        return true
    }

    private static func distance(_ a: String, _ b: String) -> Int {
        let a = Array(a)
        let b = Array(b)
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var prev = Array(0...b.count)
        for (i, ca) in a.enumerated() {
            var cur = [i + 1]
            cur.reserveCapacity(b.count + 1)
            for (j, cb) in b.enumerated() {
                let cost = ca == cb ? 0 : 1
                cur.append(min(cur[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            }
            prev = cur
        }
        return prev[b.count]
    }

    private static func fold(_ text: String) -> String {
        var s = text.folding(options: .diacriticInsensitive, locale: .current).lowercased()
        s = s.replacingOccurrences(of: "&", with: " and ")
        s = s.replacingOccurrences(of: "™", with: "")
        s = s.replacingOccurrences(of: "®", with: "")
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        s = String(s.unicodeScalars.map { allowed.contains($0) ? Character($0) : " " })
        while s.contains("  ") {
            s = s.replacingOccurrences(of: "  ", with: " ")
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
