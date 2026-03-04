import Foundation
import MLX
import MLXNN

final class LayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int, initValue: Float = 1.0) {
        _gamma.wrappedValue = MLXArray.full([dim], values: MLXArray(initValue))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

final class FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var cFC: Linear
    @ModuleInfo(key: "gelu") var gelu: GELU
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(dim: Int, ratio: Float) {
        let hiddenDim = Int(Float(dim) * ratio)
        _cFC.wrappedValue = Linear(dim, hiddenDim, bias: true)
        _gelu.wrappedValue = GELU()
        _cProj.wrappedValue = Linear(hiddenDim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(gelu(cFC(x)))
    }
}

final class SelfAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(embedDim: Int, numHeads: Int) {
        precondition(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
        self.embedDim = embedDim
        self.numHeads = numHeads
        headDim = embedDim / numHeads
        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _kProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        super.init()
    }

    private func splitHeads(_ x: MLXArray) -> MLXArray {
        let (batch, sequence, _) = x.shape3
        return x.reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
    }

    private func mergeHeads(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(2)
        return x.transposed(0, 2, 1, 3).reshaped(batch, sequence, embedDim)
    }

    func callAsFunction(
        _ x: MLXArray, attentionMask: MLXArray? = nil, rope: RotaryEmbedding2D? = nil
    ) -> MLXArray {
        var q = splitHeads(qProj(x))
        var k = splitHeads(kProj(x))
        let v = splitHeads(vProj(x))

        if let rope {
            (q, k) = rope.apply(q, k)
        }

        let output = scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: attentionMask?.asType(q.dtype)
        )

        return outProj(mergeHeads(output))
    }
}

final class CrossAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(embedDim: Int, numHeads: Int) {
        precondition(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
        self.embedDim = embedDim
        self.numHeads = numHeads
        headDim = embedDim / numHeads
        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _kProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        super.init()
    }

    private func splitHeads(_ x: MLXArray) -> MLXArray {
        let (batch, sequence, _) = x.shape3
        return x.reshaped(batch, sequence, numHeads, headDim).transposed(0, 2, 1, 3)
    }

    private func mergeHeads(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(2)
        return x.transposed(0, 2, 1, 3).reshaped(batch, sequence, embedDim)
    }

    func callAsFunction(
        queries: MLXArray, keys: MLXArray, values: MLXArray, attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let q = splitHeads(qProj(queries))
        let k = splitHeads(kProj(keys))
        let v = splitHeads(vProj(values))

        let output = scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: attentionMask?.asType(q.dtype)
        )

        return outProj(mergeHeads(output))
    }
}

final class ResidualAttentionBlock: Module {
    @ModuleInfo(key: "attn") var attn: SelfAttention
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FeedForward
    @ModuleInfo(key: "ls_1") var ls1: LayerScale?
    @ModuleInfo(key: "ls_2") var ls2: LayerScale?

    init(dModel: Int, heads: Int, mlpRatio: Float = 4.0, layerScale: Float? = nil) {
        _attn.wrappedValue = SelfAttention(embedDim: dModel, numHeads: heads)
        _ln1.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        _ln2.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        _mlp.wrappedValue = FeedForward(dim: dModel, ratio: mlpRatio)

        if let layerScale {
            _ls1.wrappedValue = LayerScale(dim: dModel, initValue: layerScale)
            _ls2.wrappedValue = LayerScale(dim: dModel, initValue: layerScale)
        } else {
            _ls1.wrappedValue = nil
            _ls2.wrappedValue = nil
        }

        super.init()
    }

    private func applyScale(_ layer: LayerScale?, _ x: MLXArray) -> MLXArray {
        if let layer {
            return layer(x)
        }
        return x
    }

    func callAsFunction(
        _ x: MLXArray, attentionMask: MLXArray? = nil, rope: RotaryEmbedding2D? = nil
    ) -> MLXArray {
        var x = x
        x = x + applyScale(ls1, attn(ln1(x), attentionMask: attentionMask, rope: rope))
        x = x + applyScale(ls2, mlp(ln2(x)))
        return x
    }
}

final class Transformer: Module {
    @ModuleInfo(key: "resblocks") var resblocks: [ResidualAttentionBlock]

    init(width: Int, layers: Int, heads: Int, mlpRatio: Float, layerScale: Float? = nil) {
        var blocks: [ResidualAttentionBlock] = []
        blocks.reserveCapacity(layers)
        for _ in 0 ..< layers {
            blocks.append(
                ResidualAttentionBlock(
                    dModel: width,
                    heads: heads,
                    mlpRatio: mlpRatio,
                    layerScale: layerScale
                ))
        }
        _resblocks.wrappedValue = blocks
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, attentionMask: MLXArray? = nil, rope: RotaryEmbedding2D? = nil,
        layerIndex: Int = -1
    ) -> MLXArray {
        var x = x
        let stopIndex = (resblocks.count + layerIndex) % max(resblocks.count, 1)
        for (index, block) in resblocks.enumerated() {
            x = block(x, attentionMask: attentionMask, rope: rope)
            if index == stopIndex {
                break
            }
        }
        return x
    }
}

final class AttentionPooling: Module {
    @ParameterInfo(key: "probe") var probe: MLXArray
    @ModuleInfo(key: "attn") var attn: CrossAttention
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FeedForward

    init(embedDim: Int, heads: Int, numProbe: Int = 1, mlpRatio: Float = 4.0) {
        _probe.wrappedValue = MLXRandom.normal([1, numProbe, embedDim])
        _attn.wrappedValue = CrossAttention(embedDim: embedDim, numHeads: heads)
        _layerNorm.wrappedValue = LayerNorm(dimensions: embedDim, eps: 1e-5)
        _mlp.wrappedValue = FeedForward(dim: embedDim, ratio: mlpRatio)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let query = broadcast(probe, to: [batch, probe.dim(1), probe.dim(2)]).asType(x.dtype)
        let attended = attn(queries: query, keys: x, values: x)
        return attended + mlp(layerNorm(attended))
    }
}
