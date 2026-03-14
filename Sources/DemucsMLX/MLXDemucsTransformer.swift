import Foundation
import MLX
import MLXNN

private func createSinEmbedding(length: Int, dim: Int, shift: Int, maxPeriod: Float) -> MLXArray {
    precondition(dim % 2 == 0)

    let half = dim / 2
    let positions = MLXArray(0..<length).asType(.float32) + MLXArray(Float(shift))
    let adim = MLXArray(0..<half).asType(.float32)
    let denom = MLXArray(Float(half - 1))
    let invScale = adim / denom
    let base = MLXArray(maxPeriod)
    let phase = positions.expandedDimensions(axis: 1) / pow(base, invScale)

    let emb = concatenated([cos(phase), sin(phase)], axis: 1)
    return emb.expandedDimensions(axis: 0)
}

private func create2DSinEmbedding(dModel: Int, height: Int, width: Int, maxPeriod: Float) -> MLXArray {
    precondition(dModel % 4 == 0)

    let half = dModel / 2
    let step = -log(maxPeriod) / Float(half)
    let divTerm = exp(MLXArray(stride(from: 0, to: half, by: 2)).asType(.float32) * MLXArray(step))

    let posW = MLXArray(0..<width).asType(.float32).expandedDimensions(axis: 1)
    let posH = MLXArray(0..<height).asType(.float32).expandedDimensions(axis: 1)

    var sinW = sin(posW * divTerm).transposed(1, 0).reshaped([-1, 1, width])
    sinW = broadcast(sinW, to: [sinW.dim(0), height, width])

    var cosW = cos(posW * divTerm).transposed(1, 0).reshaped([-1, 1, width])
    cosW = broadcast(cosW, to: [cosW.dim(0), height, width])

    let peW = stacked([sinW, cosW], axis: 1).reshaped([-1, height, width])

    var sinH = sin(posH * divTerm).transposed(1, 0).reshaped([-1, height, 1])
    sinH = broadcast(sinH, to: [sinH.dim(0), height, width])

    var cosH = cos(posH * divTerm).transposed(1, 0).reshaped([-1, height, 1])
    cosH = broadcast(cosH, to: [cosH.dim(0), height, width])

    let peH = stacked([sinH, cosH], axis: 1).reshaped([-1, height, width])
    let pe = concatenated([peW, peH], axis: 0)

    return pe.expandedDimensions(axis: 0)
}

final class MyGroupNorm: Module, DemucsUnaryLayer {
    @ModuleInfo(key: "gn") var gn: GroupNorm

    init(groupCount: Int, channels: Int, eps: Float = 1e-5) {
        self._gn.wrappedValue = GroupNorm(
            groupCount: groupCount,
            dimensions: channels,
            eps: eps,
            affine: true,
            pytorchCompatible: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gn(x)
    }
}

final class TransformerMixLayer: Module {
    let isCross: Bool
    let normFirst: Bool
    let useNormOut: Bool

    @ModuleInfo(key: "attn") var attn: MultiHeadAttention?
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiHeadAttention?

    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var norm3: LayerNorm?

    @ModuleInfo(key: "norm_out") var normOut: MyGroupNorm?

    @ModuleInfo(key: "gamma_1") var gamma1: DemucsLayerScale?
    @ModuleInfo(key: "gamma_2") var gamma2: DemucsLayerScale?

    init(
        dModel: Int,
        nHead: Int,
        feedforward: Int,
        isCross: Bool,
        normFirst: Bool,
        useNormOut: Bool,
        layerScale: Bool
    ) {
        self.isCross = isCross
        self.normFirst = normFirst
        self.useNormOut = useNormOut

        if isCross {
            self._attn.wrappedValue = nil
            self._crossAttn.wrappedValue = MultiHeadAttention(dimensions: dModel, numHeads: nHead, bias: true)
        } else {
            self._attn.wrappedValue = MultiHeadAttention(dimensions: dModel, numHeads: nHead, bias: true)
            self._crossAttn.wrappedValue = nil
        }

        self._linear1.wrappedValue = Linear(dModel, feedforward, bias: true)
        self._linear2.wrappedValue = Linear(feedforward, dModel, bias: true)

        self._norm1.wrappedValue = LayerNorm(dimensions: dModel)
        self._norm2.wrappedValue = LayerNorm(dimensions: dModel)
        self._norm3.wrappedValue = isCross ? LayerNorm(dimensions: dModel) : nil

        self._normOut.wrappedValue = useNormOut ? MyGroupNorm(groupCount: 1, channels: dModel) : nil

        let initScale: Float = layerScale ? 1e-4 : 1.0
        if layerScale {
            self._gamma1.wrappedValue = DemucsLayerScale(channels: dModel, initial: initScale, channelLast: true)
            self._gamma2.wrappedValue = DemucsLayerScale(channels: dModel, initial: initScale, channelLast: true)
        } else {
            self._gamma1.wrappedValue = nil
            self._gamma2.wrappedValue = nil
        }

        super.init()
    }

    private func applyGamma(_ gamma: DemucsLayerScale?, _ x: MLXArray) -> MLXArray {
        guard let gamma else { return x }
        return gamma(x)
    }

    private func forwardSelf(_ x: MLXArray) -> MLXArray {
        guard let attn else { return x }
        var x = x
        if normFirst {
            let q = norm1(x)
            let y = attn(q, keys: q, values: q)
            x = x + applyGamma(gamma1, y)
            let ff = linear2(demucsGELU(linear1(norm2(x))))
            x = x + applyGamma(gamma2, ff)
            if useNormOut, let normOut {
                x = normOut(x)
            }
        } else {
            let y = attn(x, keys: x, values: x)
            x = norm1(x + applyGamma(gamma1, y))
            let ff = linear2(demucsGELU(linear1(x)))
            x = norm2(x + applyGamma(gamma2, ff))
        }
        return x
    }

    private func forwardCross(_ q: MLXArray, _ k: MLXArray) -> MLXArray {
        guard let crossAttn else { return q }
        var x = q
        if normFirst {
            let kNorm = norm2(k)
            let y = crossAttn(norm1(q), keys: kNorm, values: kNorm)
            x = q + applyGamma(gamma1, y)
            if let norm3 {
                let ff = linear2(demucsGELU(linear1(norm3(x))))
                x = x + applyGamma(gamma2, ff)
            }
            if useNormOut, let normOut {
                x = normOut(x)
            }
        } else {
            let y = crossAttn(q, keys: k, values: k)
            x = norm1(q + applyGamma(gamma1, y))
            let ff = linear2(demucsGELU(linear1(x)))
            x = norm2(x + applyGamma(gamma2, ff))
        }
        return x
    }

    func callAsFunction(_ x: MLXArray, other: MLXArray? = nil) -> MLXArray {
        if isCross {
            guard let other else { return x }
            return forwardCross(x, other)
        }
        return forwardSelf(x)
    }
}

final class CrossTransformerEncoder: Module {
    let numLayers: Int
    let classicParity: Int
    let maxPeriod: Float
    let weightPosEmbed: Float
    let sinRandomShift: Int

    @ModuleInfo(key: "norm_in") var normIn: LayerNorm
    @ModuleInfo(key: "norm_in_t") var normInT: LayerNorm

    @ModuleInfo(key: "layers") var layers: [TransformerMixLayer]
    @ModuleInfo(key: "layers_t") var layersT: [TransformerMixLayer]

    private var cache2D: [String: MLXArray] = [:]

    init(
        dim: Int,
        hiddenScale: Float,
        numHeads: Int,
        numLayers: Int,
        crossFirst: Bool,
        normFirst: Bool,
        normOut: Bool,
        maxPeriod: Float,
        weightPosEmbed: Float,
        layerScale: Bool,
        sinRandomShift: Int
    ) {
        self.numLayers = numLayers
        self.classicParity = crossFirst ? 1 : 0
        self.maxPeriod = maxPeriod
        self.weightPosEmbed = weightPosEmbed
        self.sinRandomShift = sinRandomShift

        self._normIn.wrappedValue = LayerNorm(dimensions: dim)
        self._normInT.wrappedValue = LayerNorm(dimensions: dim)

        let hidden = Int(Float(dim) * hiddenScale)
        self._layers.wrappedValue = (0..<numLayers).map { idx in
            TransformerMixLayer(
                dModel: dim,
                nHead: numHeads,
                feedforward: hidden,
                isCross: idx % 2 != (crossFirst ? 1 : 0),
                normFirst: normFirst,
                useNormOut: normFirst && normOut,
                layerScale: layerScale
            )
        }
        self._layersT.wrappedValue = (0..<numLayers).map { idx in
            TransformerMixLayer(
                dModel: dim,
                nHead: numHeads,
                feedforward: hidden,
                isCross: idx % 2 != (crossFirst ? 1 : 0),
                normFirst: normFirst,
                useNormOut: normFirst && normOut,
                layerScale: layerScale
            )
        }

        super.init()
    }

    private func pos2D(channels: Int, freq: Int, time: Int) -> MLXArray {
        let key = "\(channels)x\(freq)x\(time)x\(maxPeriod)"
        if let cached = cache2D[key] {
            return cached
        }
        let generated = create2DSinEmbedding(dModel: channels, height: freq, width: time, maxPeriod: maxPeriod)
        cache2D[key] = generated
        return generated
    }

    func callAsFunction(_ xIn: MLXArray, _ xtIn: MLXArray) -> (MLXArray, MLXArray) {
        try! forward(xIn, xtIn, monitor: nil)
    }

    func forward(_ xIn: MLXArray, _ xtIn: MLXArray, monitor: SeparationMonitor?) throws -> (MLXArray, MLXArray) {
        let b = xIn.dim(0)
        let c = xIn.dim(1)
        let f = xIn.dim(2)
        let t1 = xIn.dim(3)

        var x = xIn.transposed(0, 3, 2, 1).reshaped([b, t1 * f, c])

        var p2d = pos2D(channels: c, freq: f, time: t1)
        p2d = broadcast(p2d, to: [b, c, f, t1]).transposed(0, 3, 2, 1).reshaped([b, t1 * f, c])

        x = normIn(x) + MLXArray(weightPosEmbed) * p2d

        let t2 = xtIn.dim(2)
        var xt = xtIn.transposed(0, 2, 1)

        let shift = sinRandomShift > 0 ? Int.random(in: 0...sinRandomShift) : 0
        var p1d = createSinEmbedding(length: t2, dim: c, shift: shift, maxPeriod: maxPeriod)
        p1d = broadcast(p1d, to: [b, t2, c])

        xt = normInT(xt) + MLXArray(weightPosEmbed) * p1d

        for idx in 0..<numLayers {
            monitor?.reportProgress(Float(idx) / Float(numLayers), stage: "Transformer \(idx + 1)/\(numLayers)")
            try monitor?.checkCancellation()

            if idx % 2 == classicParity {
                x = layers[idx](x)
                xt = layersT[idx](xt)
            }
            else {
                let oldX = x
                x = layers[idx](x, other: xt)
                xt = layersT[idx](xt, other: oldX)
            }
        }

        let xOut = x.reshaped([b, t1, f, c]).transposed(0, 3, 2, 1)
        let xtOut = xt.transposed(0, 2, 1)
        return (xOut, xtOut)
    }
}
