import MLX
import XCTest

@testable import MLXPE

final class MLXPETests: XCTestCase {
    func testVariantRegistry() {
        XCTAssertEqual(PECoreVariant.allCases.count, 5)
        XCTAssertEqual(PECoreVariant.tiny.configuration.vision.patchSize, 16)
        XCTAssertEqual(PECoreVariant.large.configuration.vision.patchSize, 14)
        XCTAssertEqual(PECoreVariant.giant.configuration.vision.useClassToken, false)
    }

    // MARK: - Similarity

    func testSimilarityIdenticalVectors() {
        let v = MLXArray([1.0, 0.0, 0.0] as [Float])
        let score = PECore.similarity(v, v)
        XCTAssertEqual(score.ndim, 0, "scalar output for 1-D × 1-D")
        XCTAssertEqual(score.item(Float.self), 1.0, accuracy: 1e-5)
    }

    func testSimilarityOrthogonalVectors() {
        let a = MLXArray([1.0, 0.0, 0.0] as [Float])
        let b = MLXArray([0.0, 1.0, 0.0] as [Float])
        let score = PECore.similarity(a, b)
        XCTAssertEqual(score.item(Float.self), 0.0, accuracy: 1e-5)
    }

    func testSimilarityNonUnitNorm() {
        let a = MLXArray([3.0, 0.0, 0.0] as [Float])
        let b = MLXArray([5.0, 0.0, 0.0] as [Float])
        let score = PECore.similarity(a, b)
        XCTAssertEqual(
            score.item(Float.self), 1.0, accuracy: 1e-5,
            "Parallel vectors must have similarity 1 regardless of magnitude")
    }

    func testSimilarityBatchVector() {
        // [B, D] × [D] -> [B]
        let batch = MLXArray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0] as [Float]).reshaped(2, 3)
        let query = MLXArray([1.0, 0.0, 0.0] as [Float])
        let scores = PECore.similarity(batch, query)
        XCTAssertEqual(scores.shape, [2])
        let values: [Float] = scores.asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(values[1], 0.0, accuracy: 1e-5)
    }

    func testSimilarityBatchBatch() {
        // [B, D] × [B, D] -> [B, B]
        let a = MLXArray([1.0, 0.0, 0.0, 1.0] as [Float]).reshaped(2, 2)
        let b = MLXArray([1.0, 0.0, 0.0, 1.0] as [Float]).reshaped(2, 2)
        let scores = PECore.similarity(a, b)
        XCTAssertEqual(scores.shape, [2, 2])
        // Diagonal should be 1, off-diagonal 0
        XCTAssertEqual(scores[0, 0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(scores[0, 1].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(scores[1, 0].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(scores[1, 1].item(Float.self), 1.0, accuracy: 1e-5)
    }

    func testForwardShapesTiny() {
        let config = PECoreConfiguration.tiny
        let model = PECoreModel(config: config)

        let batch = 2
        let image = MLXRandom.normal([batch, config.vision.imageSize, config.vision.imageSize, 3])
        let tokens = MLXArray.zeros([batch, config.text.contextLength], type: Int32.self)

        let output = model(image: image, text: tokens)
        guard let imageFeatures = output.imageFeatures, let textFeatures = output.textFeatures
        else {
            XCTFail("Expected both image and text outputs")
            return
        }

        XCTAssertEqual(imageFeatures.shape, [batch, config.vision.outputDim])
        XCTAssertEqual(textFeatures.shape, [batch, config.text.outputDim])
        XCTAssertEqual(output.logitScale.ndim, 0)
    }
}
