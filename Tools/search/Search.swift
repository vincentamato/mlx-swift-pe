import ArgumentParser
import Foundation
import MLX
import MLXPE

@main
struct Search: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pe-search",
        abstract: "Search video segment embeddings with a text query."
    )

    @Option(help: "Text query to search for.")
    var query: String

    @Option(help: "Path to embeddings.safetensors file from pe-encode.")
    var embeddings: String

    @Option(help: "Path to converted PE-Core MLX model directory.")
    var model: String

    @Option(help: "Number of top results to show.")
    var topK: Int = 5

    @Option(help: "Duration of each segment in seconds.")
    var segmentDuration: Double = 10

    func run() throws {
        // Load model
        print("Loading model from \(model)...")
        let peModel = try PECore.load(from: model)

        // Encode query text -> [1, D], squeeze to [D]
        let queryVector = try peModel.encodeText(query).squeezed(axis: 0)
        eval(queryVector)

        // Load segment embeddings
        let embeddingsURL = URL(fileURLWithPath: embeddings)
        guard FileManager.default.fileExists(atPath: embeddingsURL.path) else {
            throw ValidationError("Embeddings file not found: \(embeddings)")
        }
        let segments = try MLX.loadArrays(url: embeddingsURL)

        let sortedKeys = segments.keys.sorted()
        print("\nQuery: \"\(query)\"")
        print("Searching \(sortedKeys.count) segments...\n")

        // Mean-pool each segment [N, D] -> [1, D] and stack into [S, D]
        var pooledVectors: [MLXArray] = []
        for key in sortedKeys {
            guard let frameEmbeddings = segments[key] else { continue }
            pooledVectors.append(mean(frameEmbeddings, axis: 0, keepDims: true))
        }
        let stacked = concatenated(pooledVectors, axis: 0)
        eval(stacked)

        // Cosine similarities: [S, D] × [D] -> [S]
        let scores = PECore.similarity(stacked, queryVector)
        eval(scores)

        // Collect and rank results
        let scoreValues: [Float] = scores.asArray(Float.self)
        var results: [(index: Int, key: String, score: Float)] = []
        for (index, key) in sortedKeys.enumerated() {
            results.append((index: index, key: key, score: scoreValues[index]))
        }
        results.sort { $0.score > $1.score }
        let topResults = results.prefix(topK)

        // Print ranked results
        print("Rank  Segment       Time Range           Score")
        for (rank, result) in topResults.enumerated() {
            let startTime = Double(result.index) * segmentDuration
            let endTime = startTime + segmentDuration
            let timeRange = String(format: "[%.1fs - %.1fs]", startTime, endTime)
            let segment = result.key.padding(toLength: 12, withPad: " ", startingAt: 0)
            let time = timeRange.padding(toLength: 19, withPad: " ", startingAt: 0)
            print(String(format: "%4d  \(segment)  \(time)  %.4f", rank + 1, result.score))
        }
    }
}
