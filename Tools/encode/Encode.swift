import AVFoundation
import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXPE

#if canImport(AppKit)
    import AppKit
#endif

@main
struct Encode: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pe-encode",
        abstract: "Encode a video into per-segment embeddings using a PE-Core model."
    )

    @Option(help: "Path to input video file.")
    var input: String

    @Option(help: "Path to converted PE-Core MLX model directory.")
    var model: String

    @Option(help: "Duration of each segment in seconds.")
    var segmentDuration: Double = 10

    @Option(help: "Number of frames to sample per segment.")
    var framesPerSegment: Int = 8

    @Option(help: "Output directory for .safetensors file.")
    var output: String = "./output"

    func run() throws {
        // Load model
        print("Loading model from \(model)...")
        let peModel = try PECore.load(from: model)
        print("Model loaded (imageSize=\(peModel.imageSize))")

        // Open video
        let videoURL = URL(fileURLWithPath: input)
        guard FileManager.default.fileExists(atPath: videoURL.path) else {
            throw ValidationError("Video file not found: \(input)")
        }
        let asset = AVAsset(url: videoURL)
        let duration = CMTimeGetSeconds(asset.duration)
        guard duration > 0 else {
            throw ValidationError("Could not determine video duration")
        }
        print("Video duration: \(String(format: "%.1f", duration))s")

        // Compute segments
        let segmentCount = Int(ceil(duration / segmentDuration))
        print(
            "Segments: \(segmentCount) (each \(segmentDuration)s, \(framesPerSegment) frames/segment)"
        )

        var allSegments: [String: MLXArray] = [:]

        for seg in 0 ..< segmentCount {
            let segStart = Double(seg) * segmentDuration
            let segEnd = min(segStart + segmentDuration, duration)
            let segLen = segEnd - segStart

            // Compute uniformly-spaced timestamps within this segment
            var timestamps: [CMTime] = []
            for f in 0 ..< framesPerSegment {
                let t = segStart + Double(f) * segLen / Double(framesPerSegment)
                timestamps.append(CMTime(seconds: t, preferredTimescale: 600))
            }

            // Extract frames
            let cgFrames = try Self.extractFrames(asset: asset, timestamps: timestamps)

            // Encode each frame and keep frame-level embeddings.
            var frameEmbeddings: [MLXArray] = []
            for cgImage in cgFrames {
                #if canImport(AppKit)
                    let nsImage = NSImage(
                        cgImage: cgImage,
                        size: NSSize(width: cgImage.width, height: cgImage.height))
                    let embedding = try peModel.encodeImage(nsImage)
                #else
                    let uiImage = UIImage(cgImage: cgImage)
                    let embedding = try peModel.encodeImage(uiImage)
                #endif
                frameEmbeddings.append(embedding)
            }

            // Stack into [N, D]
            let features = concatenated(frameEmbeddings, axis: 0)
            eval(features)

            let key = String(format: "segment_%03d", seg)
            allSegments[key] = features

            print(
                "  \(key): [\(String(format: "%.1f", segStart))s - \(String(format: "%.1f", segEnd))s] -> \(features.shape)"
            )
        }

        // Save to safetensors
        let outputDir = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let outputPath = outputDir.appendingPathComponent("embeddings.safetensors")
        try MLX.save(arrays: allSegments, url: outputPath)

        print("\nSaved \(allSegments.count) segments to \(outputPath.path)")
        print("Keys: \(allSegments.keys.sorted().joined(separator: ", "))")
    }

    private static func extractFrames(asset: AVAsset, timestamps: [CMTime]) throws -> [CGImage] {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var frames: [CGImage] = []
        for time in timestamps {
            let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
            frames.append(cgImage)
        }
        return frames
    }
}
