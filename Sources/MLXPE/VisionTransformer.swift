import Foundation
import MLX
import MLXNN

struct VisionDebugOutputs {
    let patchEmbedding: MLXArray
    let block0: MLXArray?
    let finalBlock: MLXArray
    let pooled: MLXArray
    let projected: MLXArray
}

final class VisionTransformer: Module {
    let patchSize: Int
    let width: Int
    let heads: Int
    let layers: Int
    let outputDim: Int
    let poolType: String
    let useClassToken: Bool
    let useAbsolutePositionEmbedding: Bool
    let useRope2D: Bool
    let imageSize: Int

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "attn_pool") var attnPool: AttentionPooling?

    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray?
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray?
    @ParameterInfo(key: "proj") var projection: MLXArray?

    let rope2D: RotaryEmbedding2D?
    let positionGridSize: Int

    init(config: VisionConfiguration) {
        patchSize = config.patchSize
        width = config.width
        heads = config.heads
        layers = config.layers
        outputDim = config.outputDim
        poolType = config.poolType
        useClassToken = config.useClassToken
        useAbsolutePositionEmbedding = config.useAbsolutePositionEmbedding
        useRope2D = config.useRope2D
        imageSize = config.imageSize

        _conv1.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: width,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize),
            bias: false
        )
        _lnPre.wrappedValue =
            config.usePreLayerNorm
            ? LayerNorm(dimensions: width, eps: 1e-5)
            : LayerNorm(dimensions: width, eps: 1e-5, affine: false, bias: false)
        _lnPost.wrappedValue =
            config.usePostLayerNorm
            ? LayerNorm(dimensions: width, eps: 1e-5)
            : LayerNorm(dimensions: width, eps: 1e-5, affine: false, bias: false)
        _transformer.wrappedValue = Transformer(
            width: width,
            layers: layers,
            heads: heads,
            mlpRatio: config.mlpRatio,
            layerScale: nil
        )

        if poolType == "attn" {
            _attnPool.wrappedValue = AttentionPooling(
                embedDim: width,
                heads: config.attentionPoolerHeads,
                numProbe: 1,
                mlpRatio: 4.0
            )
        } else {
            _attnPool.wrappedValue = nil
        }

        let initScale = pow(Float(width), -0.5)
        if useClassToken {
            _classEmbedding.wrappedValue = MLXRandom.normal([width], scale: initScale)
        } else {
            _classEmbedding.wrappedValue = nil
        }

        positionGridSize = config.imageSize / config.patchSize
        if useAbsolutePositionEmbedding {
            let tokenCount = (useClassToken ? 1 : 0) + (positionGridSize * positionGridSize)
            _positionalEmbedding.wrappedValue = MLXRandom.normal(
                [tokenCount, width], scale: initScale)
        } else {
            _positionalEmbedding.wrappedValue = nil
        }

        _projection.wrappedValue = MLXRandom.normal([width, outputDim], scale: initScale)

        rope2D =
            useRope2D
            ? RotaryEmbedding2D(headDim: width / heads, useClassToken: useClassToken) : nil

        super.init()
    }

    private func sampleAbsolutePositionEmbedding(gridHeight: Int, gridWidth: Int) -> MLXArray? {
        guard let positionalEmbedding else {
            return nil
        }

        if positionGridSize == gridHeight, positionGridSize == gridWidth {
            return positionalEmbedding[.newAxis, 0..., 0...]
        }

        preconditionFailure(
            "Absolute positional embedding interpolation for dynamic image size is not implemented yet. "
                + "Expected grid \(positionGridSize)x\(positionGridSize), got \(gridHeight)x\(gridWidth)."
        )
    }

    private func pool(_ x: MLXArray) -> MLXArray {
        switch poolType {
        case "tok":
            return x[0..., 0, 0...]
        case "avg":
            return x.mean(axis: 1)
        case "attn":
            guard let attnPool else {
                preconditionFailure("attn pool requested but module is missing")
            }
            return attnPool(x).squeezed(axis: 1)
        case "none":
            return x
        default:
            preconditionFailure("Unsupported pool type: \(poolType)")
        }
    }

    func forwardFeatures(
        _ x: MLXArray,
        norm: Bool = false,
        layerIndex: Int = -1,
        stripClassToken: Bool = false,
        collectIntermediate: Bool = false
    ) -> (output: MLXArray, patchEmbedding: MLXArray?, block0: MLXArray?) {
        let batch = x.dim(0)
        let gridHeight = x.dim(1) / patchSize
        let gridWidth = x.dim(2) / patchSize

        var patchEmbedding = conv1(x)
        patchEmbedding = patchEmbedding.reshaped(batch, gridHeight * gridWidth, width)

        var hidden = patchEmbedding

        if useClassToken, let classEmbedding {
            let cls = classEmbedding.reshaped(1, 1, width)
            let clsExpanded = broadcast(cls, to: [batch, 1, width]).asType(hidden.dtype)
            hidden = concatenated([clsExpanded, hidden], axis: 1)
        }

        if useAbsolutePositionEmbedding,
            let sampled = sampleAbsolutePositionEmbedding(
                gridHeight: gridHeight, gridWidth: gridWidth)
        {
            hidden = hidden + sampled.asType(hidden.dtype)
        }

        if let rope2D {
            rope2D.updateGrid(gridHeight: gridHeight, gridWidth: gridWidth)
        }

        hidden = lnPre(hidden)

        var firstBlock: MLXArray?
        if collectIntermediate, !transformer.resblocks.isEmpty {
            hidden = transformer.resblocks[0](hidden, attentionMask: nil, rope: rope2D)
            firstBlock = hidden
            if transformer.resblocks.count > 1 {
                for block in transformer.resblocks.dropFirst() {
                    hidden = block(hidden, attentionMask: nil, rope: rope2D)
                }
            }
        } else {
            hidden = transformer(hidden, attentionMask: nil, rope: rope2D, layerIndex: layerIndex)
        }

        if norm {
            hidden = lnPost(hidden)
        }

        if stripClassToken, useClassToken {
            hidden = hidden[0..., 1..., 0...]
        }

        return (hidden, collectIntermediate ? patchEmbedding : nil, firstBlock)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var features = forwardFeatures(x, norm: true).output
        features = pool(features)

        if let projection {
            features = matmul(features, projection)
        }

        return features
    }

    func forwardWithDebug(_ x: MLXArray) -> VisionDebugOutputs {
        let outputs = forwardFeatures(x, norm: true, collectIntermediate: true)
        let pooled = pool(outputs.output)
        let projected: MLXArray
        if let projection {
            projected = matmul(pooled, projection)
        } else {
            projected = pooled
        }

        return VisionDebugOutputs(
            patchEmbedding: outputs.patchEmbedding ?? MLXArray.zeros([0]),
            block0: outputs.block0,
            finalBlock: outputs.output,
            pooled: pooled,
            projected: projected
        )
    }
}
