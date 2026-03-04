import Foundation
import MLX
import MLXNN

private struct StoredConfiguration: Codable {
    let modelName: String
    let vision: VisionConfigurationCodable
    let text: TextConfigurationCodable
    let initialLogitScale: Float

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case vision
        case text
        case initialLogitScale = "initial_logit_scale"
    }
}

private struct VisionConfigurationCodable: Codable {
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

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case width
        case layers
        case heads
        case mlpRatio = "mlp_ratio"
        case outputDim = "output_dim"
        case useClassToken = "use_class_token"
        case useAbsolutePositionEmbedding = "use_abs_posemb"
        case useRope2D = "use_rope2d"
        case usePreLayerNorm = "use_ln_pre"
        case usePostLayerNorm = "use_ln_post"
        case poolType = "pool_type"
        case attentionPoolerHeads = "attn_pooler_heads"
    }

    func toRuntime() -> VisionConfiguration {
        VisionConfiguration(
            imageSize: imageSize,
            patchSize: patchSize,
            width: width,
            layers: layers,
            heads: heads,
            mlpRatio: mlpRatio,
            outputDim: outputDim,
            useClassToken: useClassToken,
            useAbsolutePositionEmbedding: useAbsolutePositionEmbedding,
            useRope2D: useRope2D,
            usePreLayerNorm: usePreLayerNorm,
            usePostLayerNorm: usePostLayerNorm,
            poolType: poolType,
            attentionPoolerHeads: attentionPoolerHeads
        )
    }
}

private struct TextConfigurationCodable: Codable {
    let contextLength: Int
    let vocabSize: Int
    let width: Int
    let heads: Int
    let layers: Int
    let mlpRatio: Float
    let outputDim: Int
    let poolType: String
    let padID: Int

    enum CodingKeys: String, CodingKey {
        case contextLength = "context_length"
        case vocabSize = "vocab_size"
        case width
        case heads
        case layers
        case mlpRatio = "mlp_ratio"
        case outputDim = "output_dim"
        case poolType = "pool_type"
        case padID = "pad_id"
    }

    func toRuntime() -> TextConfiguration {
        TextConfiguration(
            contextLength: contextLength,
            vocabSize: vocabSize,
            width: width,
            heads: heads,
            layers: layers,
            mlpRatio: mlpRatio,
            outputDim: outputDim,
            poolType: poolType,
            padID: padID
        )
    }
}

enum WeightLoadingError: Error {
    case missingConfig(String)
    case missingWeights(String)
}

func loadPretrained(modelPath: String) throws -> PECoreModel {
    let modelURL = URL(fileURLWithPath: modelPath)
    let configURL = modelURL.appendingPathComponent("config.json")
    let weightsURL = modelURL.appendingPathComponent("model.safetensors")

    guard FileManager.default.fileExists(atPath: configURL.path) else {
        throw WeightLoadingError.missingConfig(configURL.path)
    }
    guard FileManager.default.fileExists(atPath: weightsURL.path) else {
        throw WeightLoadingError.missingWeights(weightsURL.path)
    }

    let configData = try Data(contentsOf: configURL)
    let storedConfig = try JSONDecoder().decode(StoredConfiguration.self, from: configData)
    let runtimeConfig = PECoreConfiguration(
        modelName: storedConfig.modelName,
        vision: storedConfig.vision.toRuntime(),
        text: storedConfig.text.toRuntime(),
        initialLogitScale: storedConfig.initialLogitScale
    )

    let model = PECoreModel(config: runtimeConfig)
    try model.loadWeights(from: modelPath)
    return model
}

extension PECoreModel {
    func loadWeights(from modelPath: String) throws {
        let modelURL = URL(fileURLWithPath: modelPath)
        let weightsURL = modelURL.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: weightsURL)
        try update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
    }
}
