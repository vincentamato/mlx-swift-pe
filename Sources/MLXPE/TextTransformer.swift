import Foundation
import MLX
import MLXNN

struct TextDebugOutputs {
    let tokenPlusPositionEmbedding: MLXArray
    let block0: MLXArray?
    let finalHiddenState: MLXArray
    let pooled: MLXArray
    let projected: MLXArray
}

final class TextTransformer: Module {
    let contextLength: Int
    let vocabSize: Int
    let width: Int
    let heads: Int
    let layers: Int
    let outputDim: Int
    let poolType: String
    let padID: Int

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "ln_final") var lnFinal: LayerNorm
    @ParameterInfo(key: "text_projection") var textProjection: MLXArray?

    let causalMask: MLXArray?

    init(config: TextConfiguration, noCausalMask: Bool = false) {
        contextLength = config.contextLength
        vocabSize = config.vocabSize
        width = config.width
        heads = config.heads
        layers = config.layers
        outputDim = config.outputDim
        poolType = config.poolType
        padID = config.padID

        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: width)
        _positionalEmbedding.wrappedValue = MLXArray.zeros([contextLength, width])
        _transformer.wrappedValue = Transformer(
            width: width,
            layers: layers,
            heads: heads,
            mlpRatio: config.mlpRatio,
            layerScale: nil
        )
        _lnFinal.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _textProjection.wrappedValue = MLXArray.zeros([width, outputDim])

        if noCausalMask {
            causalMask = nil
        } else {
            let indices = MLXArray(0 ..< contextLength)
            var mask = indices[0..., .newAxis] .< indices[.newAxis, 0...]
            mask = mask.asType(.float32) * -1e9
            causalMask = mask
        }

        super.init()
    }

    private func textGlobalPool(_ x: MLXArray, tokens: MLXArray) -> MLXArray {
        switch poolType {
        case "first":
            return x[0..., 0, 0...]
        case "last":
            return x[0..., -1, 0...]
        case "argmax":
            let eosPositions = tokens.argMax(axis: -1)
            return x[MLXArray(0 ..< x.dim(0)), eosPositions]
        case "none":
            return x
        default:
            preconditionFailure("Unsupported text pool type: \(poolType)")
        }
    }

    func callAsFunction(_ text: MLXArray) -> MLXArray {
        var x = tokenEmbedding(text)
        let seqLen = text.dim(1)

        var mask: MLXArray?
        if let causalMask {
            mask = causalMask[..<seqLen, ..<seqLen]
        }

        x = x + positionalEmbedding[..<seqLen, 0...].asType(x.dtype)
        x = transformer(x, attentionMask: mask, rope: nil)
        x = lnFinal(x)

        var pooled = textGlobalPool(x, tokens: text)
        if let textProjection {
            pooled = matmul(pooled, textProjection)
        }
        return pooled
    }

    func forwardWithDebug(_ text: MLXArray) -> TextDebugOutputs {
        var x = tokenEmbedding(text)
        let seqLen = text.dim(1)

        var mask: MLXArray?
        if let causalMask {
            mask = causalMask[..<seqLen, ..<seqLen]
        }

        x = x + positionalEmbedding[..<seqLen, 0...].asType(x.dtype)
        let tokenAndPos = x

        var block0: MLXArray?
        if !transformer.resblocks.isEmpty {
            x = transformer.resblocks[0](x, attentionMask: mask, rope: nil)
            block0 = x
            if transformer.resblocks.count > 1 {
                for block in transformer.resblocks.dropFirst() {
                    x = block(x, attentionMask: mask, rope: nil)
                }
            }
        }

        x = lnFinal(x)

        let pooled = textGlobalPool(x, tokens: text)
        let projected: MLXArray
        if let textProjection {
            projected = matmul(pooled, textProjection)
        } else {
            projected = pooled
        }

        return TextDebugOutputs(
            tokenPlusPositionEmbedding: tokenAndPos,
            block0: block0,
            finalHiddenState: x,
            pooled: pooled,
            projected: projected
        )
    }
}
