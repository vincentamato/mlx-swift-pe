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
