import Foundation
import MLX

private struct TokenBigram: Hashable {
    let first: String
    let second: String
}

enum TokenizerError: Error {
    case missingBPEFile(String)
    case malformedBPELine(String)
}

final class Tokenizer {
    let contextLength: Int
    let vocabSize: Int
    let sotTokenID: Int
    let eotTokenID: Int

    private let byteEncoder: [UInt8: String]
    private let byteDecoder: [String: UInt8]
    private let encoder: [String: Int]
    private let decoder: [Int: String]
    private let bpeRanks: [TokenBigram: Int]
    private let pattern: NSRegularExpression

    private var cache: [String: String]

    init(
        bpePath: String,
        contextLength: Int,
        clean: String = "lower"
    ) throws {
        self.contextLength = contextLength

        let bpeURL = URL(fileURLWithPath: bpePath)
        guard FileManager.default.fileExists(atPath: bpeURL.path) else {
            throw TokenizerError.missingBPEFile(bpeURL.path)
        }

        let bpeData = try Data(contentsOf: bpeURL)
        guard let bpeText = String(data: bpeData, encoding: .utf8) else {
            throw TokenizerError.malformedBPELine("Failed to decode BPE merges as UTF-8")
        }

        let byteTables = Tokenizer.makeByteUnicodeTables()
        byteEncoder = byteTables.encoder
        byteDecoder = byteTables.decoder

        let allLines = bpeText.split(separator: "\n").map(String.init)
        let mergeLimit = 49_152 - 256 - 2 + 1
        let merges = Array(allLines.dropFirst().prefix(mergeLimit))
            .compactMap { line -> TokenBigram? in
                let pieces = line.split(separator: " ").map(String.init)
                guard pieces.count == 2 else {
                    return nil
                }
                return TokenBigram(first: pieces[0], second: pieces[1])
            }

        var vocab = byteTables.orderedUnicode
        vocab += vocab.map { $0 + "</w>" }
        vocab += merges.map { $0.first + $0.second }

        let specialTokens = ["<start_of_text>", "<end_of_text>"]
        vocab += specialTokens

        var encoderMap: [String: Int] = [:]
        encoderMap.reserveCapacity(vocab.count)
        for (index, token) in vocab.enumerated() {
            encoderMap[token] = index
        }
        encoder = encoderMap

        var decoderMap: [Int: String] = [:]
        decoderMap.reserveCapacity(vocab.count)
        for (token, id) in encoderMap {
            decoderMap[id] = token
        }
        decoder = decoderMap

        var ranks: [TokenBigram: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (index, merge) in merges.enumerated() {
            ranks[merge] = index
        }
        bpeRanks = ranks

        cache = [:]
        for token in specialTokens {
            cache[token] = token
        }

        let special = specialTokens.map(NSRegularExpression.escapedPattern(for:)).joined(
            separator: "|")
        let regexPattern =
            special + "|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
        pattern = try NSRegularExpression(pattern: regexPattern, options: [.caseInsensitive])

        vocabSize = encoder.count
        sotTokenID = encoderMap["<start_of_text>"] ?? 0
        eotTokenID = encoderMap["<end_of_text>"] ?? 1

        _ = clean
    }

    static func fromPretrained(modelPath: String, contextLength: Int? = nil) throws -> Tokenizer {
        let modelURL = URL(fileURLWithPath: modelPath)

        let configURL = modelURL.appendingPathComponent("config.json")
        var resolvedContextLength = contextLength
        if resolvedContextLength == nil, FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let text = json["text"] as? [String: Any],
                let ctx = text["context_length"] as? NSNumber
            {
                resolvedContextLength = ctx.intValue
            }
        }

        let plainBPE = modelURL.appendingPathComponent("bpe_simple_vocab_16e6.txt")
        if FileManager.default.fileExists(atPath: plainBPE.path) {
            return try Tokenizer(
                bpePath: plainBPE.path,
                contextLength: resolvedContextLength ?? 77
            )
        }

        throw TokenizerError.missingBPEFile(plainBPE.path)
    }

    func encode(_ text: String) -> [Int] {
        let cleaned =
            text
            .lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let matches = pattern.matches(
            in: cleaned,
            range: NSRange(cleaned.startIndex ..< cleaned.endIndex, in: cleaned)
        )

        var tokenIDs: [Int] = []
        tokenIDs.reserveCapacity(cleaned.count)

        for match in matches {
            guard let range = Range(match.range, in: cleaned) else {
                continue
            }
            let token = String(cleaned[range])
            let encodedBytes = token.utf8.compactMap { byteEncoder[$0] }.joined()

            let bpePieces = bpe(encodedBytes).split(separator: " ").map(String.init)
            for piece in bpePieces {
                if let id = encoder[piece] {
                    tokenIDs.append(id)
                }
            }
        }

        return tokenIDs
    }

    func decode(_ tokens: [Int]) -> String {
        let text = tokens.compactMap { decoder[$0] }.joined()
        let bytes = text.compactMap { byteDecoder[String($0)] }
        let decoded = String(decoding: bytes, as: UTF8.self)
        return decoded.replacingOccurrences(of: "</w>", with: " ")
    }

    func encode(_ text: String, contextLength: Int? = nil) -> MLXArray {
        tokenize([text], contextLength: contextLength)
    }

    func tokenize(_ texts: [String], contextLength: Int? = nil) -> MLXArray {
        let finalContextLength = contextLength ?? self.contextLength
        var output = [Int32](repeating: 0, count: texts.count * finalContextLength)

        for (batchIndex, text) in texts.enumerated() {
            var tokens = [sotTokenID] + encode(text) + [eotTokenID]
            if tokens.count > finalContextLength {
                tokens = Array(tokens.prefix(finalContextLength))
                if let lastIndex = tokens.indices.last {
                    tokens[lastIndex] = eotTokenID
                }
            }

            let base = batchIndex * finalContextLength
            for (index, token) in tokens.enumerated() {
                output[base + index] = Int32(token)
            }
        }

        return MLXArray(output, [texts.count, finalContextLength])
    }

    private func bpe(_ token: String) -> String {
        if let cached = cache[token] {
            return cached
        }

        guard !token.isEmpty else {
            return ""
        }

        var symbols = token.map(String.init)
        if let last = symbols.popLast() {
            symbols.append(last + "</w>")
        }

        guard symbols.count > 1 else {
            let single = (symbols.first ?? token) + "</w>"
            cache[token] = single
            return single
        }

        while true {
            let pairs = Tokenizer.pairs(symbols)
            guard !pairs.isEmpty else {
                break
            }

            let ranked = pairs.map { ($0, bpeRanks[$0] ?? Int.max) }
            guard let best = ranked.min(by: { $0.1 < $1.1 })?.0,
                bpeRanks[best] != nil
            else {
                break
            }

            var newSymbols: [String] = []
            var index = 0
            while index < symbols.count {
                if index < symbols.count - 1,
                    symbols[index] == best.first,
                    symbols[index + 1] == best.second
                {
                    newSymbols.append(best.first + best.second)
                    index += 2
                } else {
                    newSymbols.append(symbols[index])
                    index += 1
                }
            }

            symbols = newSymbols
            if symbols.count == 1 {
                break
            }
        }

        let merged = symbols.joined(separator: " ")
        cache[token] = merged
        return merged
    }

    private static func pairs(_ symbols: [String]) -> Set<TokenBigram> {
        guard symbols.count > 1 else {
            return []
        }
        var pairs: Set<TokenBigram> = []
        pairs.reserveCapacity(symbols.count - 1)
        for idx in 0 ..< (symbols.count - 1) {
            pairs.insert(TokenBigram(first: symbols[idx], second: symbols[idx + 1]))
        }
        return pairs
    }

    private static func makeByteUnicodeTables() -> (
        encoder: [UInt8: String],
        decoder: [String: UInt8],
        orderedUnicode: [String]
    ) {
        var bs: [Int] = Array(33 ... 126)
        bs += Array(161 ... 172)
        bs += Array(174 ... 255)

        var cs = bs
        var n = 0

        for byte in 0 ..< 256 {
            if !bs.contains(byte) {
                bs.append(byte)
                cs.append(256 + n)
                n += 1
            }
        }

        let unicodeStrings = cs.compactMap { UnicodeScalar($0) }.map(String.init)

        var enc: [UInt8: String] = [:]
        var dec: [String: UInt8] = [:]

        for (byte, scalar) in zip(bs, unicodeStrings) {
            let key = UInt8(byte)
            enc[key] = scalar
            dec[scalar] = key
        }

        return (enc, dec, unicodeStrings)
    }
}
