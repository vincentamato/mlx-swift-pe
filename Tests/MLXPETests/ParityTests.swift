import Foundation
import MLX
import XCTest

@testable import MLXPE

final class ParityTests: XCTestCase {
    private func assertClose(
        _ a: MLXArray,
        _ b: MLXArray,
        atol: Float = 1e-4,
        rtol: Float = 1e-4,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, "\(label): shape mismatch", file: file, line: line)

        let diff = abs(a.asType(.float32) - b.asType(.float32))
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = diff.mean().item(Float.self)
        let maxRef = abs(b.asType(.float32)).max().item(Float.self)

        let passes = maxDiff < atol || maxDiff < rtol * max(maxRef, 1e-12)
        XCTAssertTrue(
            passes,
            "\(label): maxDiff=\(maxDiff), meanDiff=\(meanDiff), maxRef=\(maxRef)",
            file: file,
            line: line
        )
    }

    private func resourceURL(_ path: String) -> URL? {
        Bundle.module.resourceURL?.appendingPathComponent("Resources").appendingPathComponent(path)
    }

    func testLayerwiseParityIfResourcesAvailable() throws {
        guard
            let modelDir = resourceURL("Model"),
            let referenceDir = resourceURL("ReferenceOutputs"),
            FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("config.json").path),
            FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("model.safetensors").path),
            FileManager.default.fileExists(
                atPath: referenceDir.appendingPathComponent("test_image.npy").path)
        else {
            throw XCTSkip(
                "Parity resources missing. Generate model + reference outputs first (see scripts/ and README)."
            )
        }

        let model = try loadPretrained(modelPath: modelDir.path)

        var image = try loadArray(url: referenceDir.appendingPathComponent("test_image.npy"))
        let tokens = try loadArray(url: referenceDir.appendingPathComponent("test_tokens.npy"))
            .asType(.int32)

        // Python reference stores image input as NCHW; MLX model expects NHWC.
        if image.ndim == 4, image.dim(1) == 3 {
            image = image.transposed(0, 2, 3, 1)
        }

        let debug = model.forwardWithDebug(image: image, text: tokens)

        let refPatch = try loadArray(
            url: referenceDir.appendingPathComponent("vision_patch_embed_output.npy"))
        let refBlock0 = try loadArray(
            url: referenceDir.appendingPathComponent("vision_block_0_output.npy"))
        let refBlockFinal = try loadArray(
            url: referenceDir.appendingPathComponent("vision_block_final_output.npy"))
        let refVisionPooled = try loadArray(
            url: referenceDir.appendingPathComponent("vision_pooled_output.npy"))
        let refVisionProjected = try loadArray(
            url: referenceDir.appendingPathComponent("vision_projected_output.npy"))

        let refTextEmbed = try loadArray(
            url: referenceDir.appendingPathComponent("text_embed_output.npy"))
        let refTextBlock0 = try loadArray(
            url: referenceDir.appendingPathComponent("text_block_0_output.npy"))
        let refImageFeatures = try loadArray(
            url: referenceDir.appendingPathComponent("image_features.npy"))
        let refTextFeatures = try loadArray(
            url: referenceDir.appendingPathComponent("text_features.npy"))
        let refSimilarity = try loadArray(
            url: referenceDir.appendingPathComponent("similarity.npy"))

        assertClose(debug.image.patchEmbedding, refPatch, label: "vision_patch_embed")

        if let block0 = debug.image.block0 {
            assertClose(block0, refBlock0, label: "vision_block_0")
        }

        assertClose(debug.image.finalBlock, refBlockFinal, label: "vision_block_final")
        assertClose(debug.image.pooled, refVisionPooled, label: "vision_pooled")
        assertClose(debug.image.projected, refVisionProjected, label: "vision_projected")

        assertClose(debug.text.tokenPlusPositionEmbedding, refTextEmbed, label: "text_embed")

        if let block0 = debug.text.block0 {
            assertClose(block0, refTextBlock0, label: "text_block_0")
        }

        assertClose(debug.imageFeatures, refImageFeatures, label: "image_features")
        assertClose(debug.textFeatures, refTextFeatures, label: "text_features")
        assertClose(debug.similarity, refSimilarity, label: "similarity")
    }
}
