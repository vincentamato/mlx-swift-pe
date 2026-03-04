import Foundation
import MLX
import MLXNN

struct PECoreOutput {
    let imageFeatures: MLXArray?
    let textFeatures: MLXArray?
    let logitScale: MLXArray
}

final class PECoreModel: Module {
    let config: PECoreConfiguration

    @ModuleInfo(key: "visual") var visual: VisionTransformer
    @ModuleInfo(key: "text") var text: TextTransformer
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    init(config: PECoreConfiguration) {
        self.config = config
        _visual.wrappedValue = VisionTransformer(config: config.vision)
        _text.wrappedValue = TextTransformer(config: config.text)
        _logitScale.wrappedValue = MLXArray(config.initialLogitScale)
        super.init()
    }

    private func l2Normalize(_ x: MLXArray, epsilon: Float = 1e-12) -> MLXArray {
        let squared = x * x
        let norm = sqrt(sum(squared, axis: -1, keepDims: true) + epsilon)
        return x / norm
    }

    func encodeImage(_ image: MLXArray, normalize: Bool = true) -> MLXArray {
        let features = visual(image)
        return normalize ? l2Normalize(features) : features
    }

    func encodeText(_ tokens: MLXArray, normalize: Bool = true) -> MLXArray {
        let features = text(tokens)
        return normalize ? l2Normalize(features) : features
    }

    func callAsFunction(image: MLXArray? = nil, text: MLXArray? = nil) -> PECoreOutput {
        let imageFeatures = image.map { encodeImage($0, normalize: true) }
        let textFeatures = text.map { encodeText($0, normalize: true) }
        return PECoreOutput(
            imageFeatures: imageFeatures,
            textFeatures: textFeatures,
            logitScale: exp(logitScale)
        )
    }

    func forwardWithDebug(image: MLXArray, text: MLXArray) -> (
        image: VisionDebugOutputs,
        text: TextDebugOutputs,
        imageFeatures: MLXArray,
        textFeatures: MLXArray,
        logitScale: MLXArray,
        similarity: MLXArray
    ) {
        let visionDebug = visual.forwardWithDebug(image)
        let textDebug = self.text.forwardWithDebug(text)

        let imageFeatures = l2Normalize(visionDebug.projected)
        let textFeatures = l2Normalize(textDebug.projected)
        let scale = exp(logitScale)
        let similarity = scale * matmul(imageFeatures, textFeatures.T)

        return (
            image: visionDebug,
            text: textDebug,
            imageFeatures: imageFeatures,
            textFeatures: textFeatures,
            logitScale: scale,
            similarity: similarity
        )
    }
}

/// Errors thrown by the PE Core high-level API.
public enum PECoreError: Error {
    /// Thrown when `encodeText(_:)` is called without an available tokenizer file.
    case tokenizerUnavailable(String)
}

/// The Perception Encoder (PE Core) model for visual and text embeddings.
///
/// This is the main public entry point for loading model weights and producing
/// normalized image and text embeddings.
public final class PECore {
    private var tokenizer: Tokenizer?
    private var model: PECoreModel
    private let imageProcessor: ImageProcessor

    /// The configuration currently used by this model.
    public private(set) var configuration: PECoreConfiguration

    /// Creates an unloaded PE Core model with the provided configuration.
    ///
    /// - Parameter configuration: The model configuration preset to initialize.
    public init(configuration: PECoreConfiguration = .large) {
        self.configuration = configuration
        model = PECoreModel(config: configuration)
        imageProcessor = ImageProcessor(imageSize: configuration.vision.imageSize)
        tokenizer = nil
    }

    private init(model: PECoreModel, tokenizer: Tokenizer?) {
        configuration = model.config
        self.model = model
        imageProcessor = ImageProcessor(imageSize: model.config.vision.imageSize)
        self.tokenizer = tokenizer
    }

    /// Loads a PE Core model from a directory containing `config.json` and `model.safetensors`.
    ///
    /// - Parameter path: Path to the model directory.
    /// - Returns: A loaded model ready for inference.
    /// - Throws: `WeightLoadingError` when model files are missing or invalid.
    public static func load(from path: String) throws -> PECore {
        let loadedModel = try loadPretrained(modelPath: path)
        let loadedTokenizer = try? Tokenizer.fromPretrained(
            modelPath: path,
            contextLength: loadedModel.config.text.contextLength
        )
        return PECore(model: loadedModel, tokenizer: loadedTokenizer)
    }

    /// Loads model weights from a directory into this instance.
    ///
    /// - Parameter path: Path to a directory containing `model.safetensors`.
    /// - Throws: `WeightLoadingError` when weights are missing or invalid.
    public func loadWeights(from path: String) throws {
        try model.loadWeights(from: path)
        tokenizer = try? Tokenizer.fromPretrained(
            modelPath: path, contextLength: model.config.text.contextLength)
    }

    /// The square image size expected by the vision encoder.
    public var imageSize: Int {
        configuration.vision.imageSize
    }

    /// Encodes a preprocessed image tensor into a normalized embedding.
    ///
    /// - Parameter image: Input tensor of shape `[B, H, W, C]`.
    /// - Returns: L2-normalized embedding tensor of shape `[B, D]`.
    public func encodeImage(_ image: MLXArray) -> MLXArray {
        model.encodeImage(image, normalize: true)
    }

    /// Encodes a platform image into a normalized embedding.
    ///
    /// - Parameter image: A platform image (`NSImage` on macOS, `UIImage` on iOS).
    /// - Returns: L2-normalized embedding tensor of shape `[1, D]`.
    public func encodeImage(_ image: PlatformImage) throws -> MLXArray {
        let pixels = try imageProcessor.process(image: image)
        return model.encodeImage(pixels, normalize: true)
    }

    /// Encodes token IDs into a normalized text embedding.
    ///
    /// - Parameter tokens: Token tensor of shape `[B, context_length]`.
    /// - Returns: L2-normalized embedding tensor of shape `[B, D]`.
    public func encodeText(_ tokens: MLXArray) -> MLXArray {
        model.encodeText(tokens, normalize: true)
    }

    /// Tokenizes and encodes a text prompt into a normalized embedding.
    ///
    /// - Parameter text: Input text prompt.
    /// - Returns: L2-normalized embedding tensor of shape `[1, D]`.
    /// - Throws: `PECoreError.tokenizerUnavailable` when tokenizer assets are missing.
    public func encodeText(_ text: String) throws -> MLXArray {
        guard let tokenizer else {
            throw PECoreError.tokenizerUnavailable(
                "Tokenizer assets are unavailable. Ensure bpe_simple_vocab_16e6.txt exists in the model directory."
            )
        }
        let tokenTensor = tokenizer.encode(text, contextLength: model.config.text.contextLength)
        return model.encodeText(tokenTensor, normalize: true)
    }

    func forwardWithDebug(image: MLXArray, text: MLXArray) -> (
        image: VisionDebugOutputs,
        text: TextDebugOutputs,
        imageFeatures: MLXArray,
        textFeatures: MLXArray,
        logitScale: MLXArray,
        similarity: MLXArray
    ) {
        model.forwardWithDebug(image: image, text: text)
    }
}
