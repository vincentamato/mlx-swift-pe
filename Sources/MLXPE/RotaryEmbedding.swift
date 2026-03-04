import Foundation
import MLX

final class RotaryEmbedding1D {
    private let dim: Int
    private let theta: Float
    private var inverseFrequencies: MLXArray

    init(dim: Int, theta: Float = 10_000.0) {
        precondition(dim % 2 == 0, "RoPE dim must be even")
        self.dim = dim
        self.theta = theta

        let indices = MLXArray(stride(from: 0, to: Float(dim), by: 2)).asType(.float32)
        let exponent = indices / Float(dim)
        inverseFrequencies = pow(MLXArray(theta), -exponent)
    }

    func frequencies(_ positions: MLXArray) -> MLXArray {
        let angles = positions.asType(.float32)[0..., .newAxis] * inverseFrequencies[.newAxis, 0...]
        return repeated(angles, count: 2, axis: -1)
    }
}

final class RotaryEmbedding2D {
    private let headDim: Int
    private let useClassToken: Bool
    private let rope1D: RotaryEmbedding1D

    private var cachedGrid: (Int, Int)?
    private var cachedFrequencies: MLXArray?
    private var cachedCos: MLXArray?
    private var cachedSin: MLXArray?

    init(headDim: Int, useClassToken: Bool, theta: Float = 10_000.0) {
        precondition(headDim % 2 == 0, "Head dim must be even for 2D RoPE")
        self.headDim = headDim
        self.useClassToken = useClassToken
        rope1D = RotaryEmbedding1D(dim: headDim / 2, theta: theta)
    }

    func updateGrid(gridHeight: Int, gridWidth: Int) {
        if let cachedGrid, cachedGrid == (gridHeight, gridWidth), cachedFrequencies != nil {
            return
        }

        let offset: Float = useClassToken ? 1.0 : 0.0
        let yRange = MLXArray(stride(from: offset, to: Float(gridHeight) + offset, by: 1.0))
        let xRange = MLXArray(stride(from: offset, to: Float(gridWidth) + offset, by: 1.0))

        let freqsY = rope1D.frequencies(yRange)
        let freqsX = rope1D.frequencies(xRange)

        let expandedY = broadcast(
            freqsY[0..., .newAxis, 0...], to: [gridHeight, gridWidth, freqsY.dim(-1)])
        let expandedX = broadcast(
            freqsX[.newAxis, 0..., 0...], to: [gridHeight, gridWidth, freqsX.dim(-1)])

        var freq = concatenated([expandedX, expandedY], axis: -1).reshaped(
            gridHeight * gridWidth, headDim)

        if useClassToken {
            let cls = MLXArray.zeros([1, headDim], type: Float.self)
            freq = concatenated([cls, freq], axis: 0)
        }

        let freqBatch = expandedDimensions(freq, axis: 0)
        cachedGrid = (gridHeight, gridWidth)
        cachedFrequencies = freqBatch
        cachedCos = cos(freqBatch)
        cachedSin = sin(freqBatch)
    }

    func apply(_ q: MLXArray, _ k: MLXArray) -> (MLXArray, MLXArray) {
        guard cachedFrequencies != nil, let cosFreq = cachedCos, let sinFreq = cachedSin else {
            preconditionFailure("RotaryEmbedding2D.updateGrid must be called before apply")
        }

        let broadcastCos = cosFreq[0..., .newAxis, 0..., 0...]
        let broadcastSin = sinFreq[0..., .newAxis, 0..., 0...]

        let qRot = applyRotaryEmb(cosFreq: broadcastCos, sinFreq: broadcastSin, tensor: q)
        let kRot = applyRotaryEmb(cosFreq: broadcastCos, sinFreq: broadcastSin, tensor: k)

        return (qRot, kRot)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let reshaped = unflatten(x, axis: -1, shape: [-1, 2])
        let x1 = reshaped[.ellipsis, 0]
        let x2 = reshaped[.ellipsis, 1]
        return stacked([-x2, x1], axis: -1).flattened(start: -2, end: -1)
    }

    private func applyRotaryEmb(cosFreq: MLXArray, sinFreq: MLXArray, tensor: MLXArray) -> MLXArray
    {
        let dtype = tensor.dtype
        let rotDim = cosFreq.dim(-1)
        precondition(rotDim <= tensor.dim(-1), "RoPE dim exceeds tensor feature dimension")

        let middle = tensor[.ellipsis, ..<rotDim]
        let middle32 = middle.asType(.float32)
        let rotated = (middle32 * cosFreq) + (rotateHalf(middle32) * sinFreq)

        if rotDim == tensor.dim(-1) {
            return rotated.asType(dtype)
        }

        let right = tensor[.ellipsis, rotDim...]
        return concatenated([rotated.asType(dtype), right], axis: -1)
    }
}
