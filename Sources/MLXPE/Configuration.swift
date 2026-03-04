import Foundation

struct VisionConfiguration: Sendable {
    let imageSize: Int
    let patchSize: Int
    let width: Int
    let layers: Int
    let heads: Int
    let mlpRatio: Float
    let outputDim: Int
    let useClassToken: Bool
    let useAbsolutePositionEmbedding: Bool
    let useRope2D: Bool
    let usePreLayerNorm: Bool
    let usePostLayerNorm: Bool
    let poolType: String
    let attentionPoolerHeads: Int

    init(
        imageSize: Int,
        patchSize: Int,
        width: Int,
        layers: Int,
        heads: Int,
        mlpRatio: Float,
        outputDim: Int,
        useClassToken: Bool,
        useAbsolutePositionEmbedding: Bool = true,
        useRope2D: Bool = true,
        usePreLayerNorm: Bool = true,
        usePostLayerNorm: Bool = true,
        poolType: String = "attn",
        attentionPoolerHeads: Int = 8
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlpRatio = mlpRatio
        self.outputDim = outputDim
        self.useClassToken = useClassToken
        self.useAbsolutePositionEmbedding = useAbsolutePositionEmbedding
        self.useRope2D = useRope2D
        self.usePreLayerNorm = usePreLayerNorm
        self.usePostLayerNorm = usePostLayerNorm
        self.poolType = poolType
        self.attentionPoolerHeads = attentionPoolerHeads
    }
}

struct TextConfiguration: Sendable {
    let contextLength: Int
    let vocabSize: Int
    let width: Int
    let heads: Int
    let layers: Int
    let mlpRatio: Float
    let outputDim: Int
    let poolType: String
    let padID: Int

    init(
        contextLength: Int,
        vocabSize: Int = 49_408,
        width: Int,
        heads: Int,
        layers: Int,
        mlpRatio: Float = 4.0,
        outputDim: Int,
        poolType: String = "argmax",
        padID: Int = 0
    ) {
        self.contextLength = contextLength
        self.vocabSize = vocabSize
        self.width = width
        self.heads = heads
        self.layers = layers
        self.mlpRatio = mlpRatio
        self.outputDim = outputDim
        self.poolType = poolType
        self.padID = padID
    }
}

/// Configuration preset for a PE Core model.
///
/// Use one of the built-in presets (`.tiny`, `.small`, `.base`, `.large`, `.giant`) or
/// construct from a `PECoreVariant`.
public struct PECoreConfiguration: Sendable {
    let modelName: String
    let vision: VisionConfiguration
    let text: TextConfiguration
    let initialLogitScale: Float

    init(
        modelName: String,
        vision: VisionConfiguration,
        text: TextConfiguration,
        initialLogitScale: Float = Float(log(1.0 / 0.07))
    ) {
        self.modelName = modelName
        self.vision = vision
        self.text = text
        self.initialLogitScale = initialLogitScale
    }

    /// Creates a preset configuration for the specified variant.
    ///
    /// - Parameter variant: The pretrained PE Core variant.
    public init(variant: PECoreVariant) {
        self = variant.configuration
    }

    /// Preset for `PE-Core-G14-448`.
    public static let giant = PECoreConfiguration(
        modelName: "PE-Core-G14-448",
        vision: VisionConfiguration(
            imageSize: 448,
            patchSize: 14,
            width: 1_536,
            layers: 50,
            heads: 16,
            mlpRatio: 8_960.0 / 1_536.0,
            outputDim: 1_280,
            useClassToken: false
        ),
        text: TextConfiguration(
            contextLength: 72,
            width: 1_280,
            heads: 20,
            layers: 24,
            outputDim: 1_280
        )
    )

    /// Preset for `PE-Core-L14-336`.
    public static let large = PECoreConfiguration(
        modelName: "PE-Core-L14-336",
        vision: VisionConfiguration(
            imageSize: 336,
            patchSize: 14,
            width: 1_024,
            layers: 24,
            heads: 16,
            mlpRatio: 4.0,
            outputDim: 1_024,
            useClassToken: true
        ),
        text: TextConfiguration(
            contextLength: 32,
            width: 1_024,
            heads: 16,
            layers: 24,
            outputDim: 1_024
        )
    )

    /// Preset for `PE-Core-B16-224`.
    public static let base = PECoreConfiguration(
        modelName: "PE-Core-B16-224",
        vision: VisionConfiguration(
            imageSize: 224,
            patchSize: 16,
            width: 768,
            layers: 12,
            heads: 12,
            mlpRatio: 4.0,
            outputDim: 1_024,
            useClassToken: true
        ),
        text: TextConfiguration(
            contextLength: 32,
            width: 1_024,
            heads: 16,
            layers: 24,
            outputDim: 1_024
        )
    )

    /// Preset for `PE-Core-S16-384`.
    public static let small = PECoreConfiguration(
        modelName: "PE-Core-S16-384",
        vision: VisionConfiguration(
            imageSize: 384,
            patchSize: 16,
            width: 384,
            layers: 12,
            heads: 6,
            mlpRatio: 4.0,
            outputDim: 512,
            useClassToken: true
        ),
        text: TextConfiguration(
            contextLength: 32,
            width: 512,
            heads: 8,
            layers: 12,
            outputDim: 512
        )
    )

    /// Preset for `PE-Core-T16-384`.
    public static let tiny = PECoreConfiguration(
        modelName: "PE-Core-T16-384",
        vision: VisionConfiguration(
            imageSize: 384,
            patchSize: 16,
            width: 192,
            layers: 12,
            heads: 3,
            mlpRatio: 4.0,
            outputDim: 512,
            useClassToken: true
        ),
        text: TextConfiguration(
            contextLength: 32,
            width: 512,
            heads: 8,
            layers: 12,
            outputDim: 512
        )
    )
}

/// Supported pretrained PE Core variants.
public enum PECoreVariant: String, CaseIterable, Sendable {
    case tiny = "PE-Core-T16-384"
    case small = "PE-Core-S16-384"
    case base = "PE-Core-B16-224"
    case large = "PE-Core-L14-336"
    case giant = "PE-Core-G14-448"

    /// Returns the default runtime configuration for this variant.
    public var configuration: PECoreConfiguration {
        switch self {
        case .tiny: .tiny
        case .small: .small
        case .base: .base
        case .large: .large
        case .giant: .giant
        }
    }

    var huggingFaceID: String {
        "facebook/\(rawValue)"
    }

    var checkpointFilename: String {
        "\(rawValue).pt"
    }
}
