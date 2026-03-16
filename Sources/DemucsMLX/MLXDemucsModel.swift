import Accelerate
import Foundation
import MLX
import MLXNN

// MARK: - Demucs v1/v2 Runtime Config

struct DemucsRuntimeConfig {
    let sources: [String]
    let audioChannels: Int
    let channels: Int
    let growth: Float
    let depth: Int
    let rewrite: Bool
    let lstmLayers: Int
    let kernelSize: Int
    let stride: Int
    let context: Int
    let normalize: Bool
    let resample: Bool
    let normStarts: Int
    let normGroups: Int
    let dconvMode: Int
    let dconvDepth: Int
    let dconvComp: Float
    let dconvAttn: Int
    let dconvLstm: Int
    let dconvInit: Float
    let samplerate: Int
    let segment: Float

    static func fromJSON(_ json: [String: Any]) throws -> DemucsRuntimeConfig {
        let kwargs = json["kwargs"] as? [String: Any] ?? json
        let i = ModelLoader.int
        let b = ModelLoader.bool
        let d = ModelLoader.double

        return DemucsRuntimeConfig(
            sources: ModelLoader.sources(kwargs),
            audioChannels: i(kwargs, "audio_channels", 2),
            channels: i(kwargs, "channels", 64),
            growth: Float(d(kwargs, "growth", 2.0)),
            depth: i(kwargs, "depth", 6),
            rewrite: b(kwargs, "rewrite", true),
            lstmLayers: i(kwargs, "lstm_layers", 0),
            kernelSize: i(kwargs, "kernel_size", 8),
            stride: i(kwargs, "stride", 4),
            context: i(kwargs, "context", 1),
            normalize: b(kwargs, "normalize", true),
            resample: b(kwargs, "resample", true),
            normStarts: i(kwargs, "norm_starts", 4),
            normGroups: i(kwargs, "norm_groups", 4),
            dconvMode: i(kwargs, "dconv_mode", 1),
            dconvDepth: i(kwargs, "dconv_depth", 2),
            dconvComp: Float(d(kwargs, "dconv_comp", 4.0)),
            dconvAttn: i(kwargs, "dconv_attn", 4),
            dconvLstm: i(kwargs, "dconv_lstm", 4),
            dconvInit: Float(d(kwargs, "dconv_init", 1e-4)),
            samplerate: i(kwargs, "samplerate", 44100),
            segment: Float(d(kwargs, "segment", 40.0))
        )
    }
}

// MARK: - Demucs Encoder Block (Sequential)

/// A single encoder block in the Demucs v1/v2 architecture.
/// Structure: Conv1d → [GroupNorm →] GELU → [DConv →] [Rewrite Conv1d → GroupNorm → GLU]
final class DemucsEncoderBlock: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        normEnabled: Bool,
        normGroups: Int,
        dconvEnabled: Bool,
        dconvDepth: Int,
        dconvComp: Float,
        dconvInit: Float,
        rewrite: Bool,
        context: Int,
        dconvLstm: Bool = false,
        dconvAttn: Bool = false
    ) {
        var mods: [Module] = []

        // Main conv
        mods.append(Conv1dNCL(inputChannels, outputChannels, kernelSize: kernelSize, stride: stride))

        // Norm
        if normEnabled {
            mods.append(GroupNormNCL(groupCount: normGroups, channels: outputChannels))
        } else {
            mods.append(DemucsIdentity())
        }

        // GELU activation (applied as identity slot - activation done in callAsFunction)
        mods.append(DemucsIdentity())

        // DConv
        if dconvEnabled {
            mods.append(DConv(
                channels: outputChannels,
                compress: dconvComp,
                depth: dconvDepth,
                initialScale: dconvInit,
                lstm: dconvLstm,
                attn: dconvAttn
            ))
        }

        // Rewrite
        if rewrite {
            mods.append(Conv1dNCL(outputChannels, 2 * outputChannels, kernelSize: 1))
            if normEnabled {
                mods.append(GroupNormNCL(groupCount: normGroups, channels: 2 * outputChannels))
            } else {
                mods.append(DemucsIdentity())
            }
            // GLU applied in callAsFunction
            mods.append(DemucsIdentity())
        }

        self._layers.wrappedValue = mods
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        var idx = 0

        // Conv
        y = applyModule(layers[idx], y)
        idx += 1

        // Norm
        y = applyModule(layers[idx], y)
        idx += 1

        // GELU activation
        y = demucsGELU(y)
        idx += 1

        // DConv (optional)
        if idx < layers.count, layers[idx] is DConv {
            y = applyModule(layers[idx], y)
            idx += 1
        }

        // Rewrite (optional): Conv + Norm + GLU
        if idx < layers.count {
            y = applyModule(layers[idx], y) // rewrite conv
            idx += 1
            if idx < layers.count {
                y = applyModule(layers[idx], y) // norm
                idx += 1
            }
            // GLU
            y = demucsGLU(y, axis: 1)
        }

        return y
    }

    private func applyModule(_ mod: Module, _ x: MLXArray) -> MLXArray {
        if let unary = mod as? any DemucsUnaryLayer {
            return unary.callAsFunction(x)
        }
        return x
    }
}

// MARK: - Demucs Decoder Block (Sequential)

/// A single decoder block in the Demucs v1/v2 architecture.
/// Structure: [Rewrite Conv1d → GroupNorm → GLU →] [DConv →] ConvTranspose1d [→ GroupNorm → GELU]
final class DemucsDecoderBlock: Module {
    let isLast: Bool
    @ModuleInfo(key: "layers") var layers: [Module]

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        isLast: Bool,
        normEnabled: Bool,
        normGroups: Int,
        dconvEnabled: Bool,
        dconvDepth: Int,
        dconvComp: Float,
        dconvInit: Float,
        rewrite: Bool,
        context: Int,
        dconvLstm: Bool = false,
        dconvAttn: Bool = false
    ) {
        self.isLast = isLast
        var mods: [Module] = []

        // Rewrite
        if rewrite {
            mods.append(Conv1dNCL(inputChannels, 2 * inputChannels, kernelSize: 2 * context + 1, padding: context))
            if normEnabled {
                mods.append(GroupNormNCL(groupCount: normGroups, channels: 2 * inputChannels))
            } else {
                mods.append(DemucsIdentity())
            }
            // GLU applied in callAsFunction
            mods.append(DemucsIdentity())
        }

        // DConv
        if dconvEnabled {
            mods.append(DConv(
                channels: inputChannels,
                compress: dconvComp,
                depth: dconvDepth,
                initialScale: dconvInit,
                lstm: dconvLstm,
                attn: dconvAttn
            ))
        }

        // Transpose conv
        mods.append(ConvTranspose1dNCL(inputChannels, outputChannels, kernelSize: kernelSize, stride: stride))

        // Output norm + activation (not for last layer)
        if !isLast {
            if normEnabled {
                mods.append(GroupNormNCL(groupCount: normGroups, channels: outputChannels))
            } else {
                mods.append(DemucsIdentity())
            }
            // GELU applied in callAsFunction
            mods.append(DemucsIdentity())
        }

        self._layers.wrappedValue = mods
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        var idx = 0

        // Check if rewrite is present (starts with Conv1dNCL for rewrite)
        let hasRewrite = layers.count > 0 && layers[0] is Conv1dNCL

        if hasRewrite {
            y = applyModule(layers[idx], y) // rewrite conv
            idx += 1
            y = applyModule(layers[idx], y) // norm
            idx += 1
            y = demucsGLU(y, axis: 1) // GLU
            idx += 1
        }

        // DConv (optional)
        if idx < layers.count, layers[idx] is DConv {
            y = applyModule(layers[idx], y)
            idx += 1
        }

        // ConvTranspose
        y = applyModule(layers[idx], y)
        idx += 1

        // Output norm + GELU (not last)
        if !isLast && idx < layers.count {
            y = applyModule(layers[idx], y) // norm
            idx += 1
            y = demucsGELU(y)
        }

        return y
    }

    private func applyModule(_ mod: Module, _ x: MLXArray) -> MLXArray {
        if let unary = mod as? any DemucsUnaryLayer {
            return unary.callAsFunction(x)
        }
        return x
    }
}

// MARK: - Demucs Graph (v1/v2 waveform-only)

final class DemucsGraph: Module {
    let config: DemucsRuntimeConfig

    @ModuleInfo(key: "encoder") var encoder: [DemucsEncoderBlock]
    @ModuleInfo(key: "decoder") var decoder: [DemucsDecoderBlock]
    @ModuleInfo(key: "lstm") var lstm: DemucsBLSTM?

    init(config: DemucsRuntimeConfig) {
        self.config = config

        var encoders: [DemucsEncoderBlock] = []
        var decoders: [DemucsDecoderBlock] = []

        var inChannels = config.audioChannels
        var channels = config.channels

        for index in 0..<config.depth {
            let norm = index >= config.normStarts
            let dconvEnabled = (config.dconvMode & 1) != 0
            let lstm = index >= config.dconvLstm
            let attn = index >= config.dconvAttn

            let enc = DemucsEncoderBlock(
                inputChannels: inChannels,
                outputChannels: channels,
                kernelSize: config.kernelSize,
                stride: config.stride,
                normEnabled: norm,
                normGroups: config.normGroups,
                dconvEnabled: dconvEnabled,
                dconvDepth: config.dconvDepth,
                dconvComp: config.dconvComp,
                dconvInit: config.dconvInit,
                rewrite: config.rewrite,
                context: config.context,
                dconvLstm: lstm,
                dconvAttn: attn
            )
            encoders.append(enc)

            let outChannels: Int
            if index > 0 {
                outChannels = inChannels
            } else {
                outChannels = config.sources.count * config.audioChannels
            }

            let dec = DemucsDecoderBlock(
                inputChannels: channels,
                outputChannels: outChannels,
                kernelSize: config.kernelSize,
                stride: config.stride,
                isLast: index == 0,
                normEnabled: norm,
                normGroups: config.normGroups,
                dconvEnabled: (config.dconvMode & 2) != 0,
                dconvDepth: config.dconvDepth,
                dconvComp: config.dconvComp,
                dconvInit: config.dconvInit,
                rewrite: config.rewrite,
                context: config.context,
                dconvLstm: lstm,
                dconvAttn: attn
            )
            decoders.insert(dec, at: 0)

            inChannels = channels
            channels = Int(config.growth * Float(channels))
        }

        self._encoder.wrappedValue = encoders
        self._decoder.wrappedValue = decoders

        // Bottleneck LSTM
        let bottleneckChannels = inChannels
        if config.lstmLayers > 0 {
            self._lstm.wrappedValue = DemucsBLSTM(
                dim: bottleneckChannels,
                layers: config.lstmLayers
            )
        } else {
            self._lstm.wrappedValue = nil
        }

        super.init()
    }

    // MARK: - Valid Length

    func validLength(_ length: Int) -> Int {
        var l = length
        if config.resample {
            l *= 2
        }
        for _ in 0..<config.depth {
            l = Int(ceil(Double(l - config.kernelSize) / Double(config.stride))) + 1
            l = max(1, l)
        }
        for _ in 0..<config.depth {
            l = (l - 1) * config.stride + config.kernelSize
        }
        if config.resample {
            l = Int(ceil(Double(l) / 2.0))
        }
        return l
    }

    // MARK: - Resampling (Accelerate/vDSP)

    /// Precomputed 63-tap Hann-windowed sinc lowpass FIR kernel (cutoff=0.25).
    private static let firKernel: [Float] = {
        let numtaps = 63
        let cutoff: Float = 0.25
        let half = (numtaps - 1) / 2
        var h = [Float](repeating: 0, count: numtaps)
        for n in 0..<numtaps {
            let tn = Float(n - half)
            let sinc: Float = tn == 0 ? 1.0 : sin(Float.pi * 2.0 * cutoff * tn) / (Float.pi * 2.0 * cutoff * tn)
            let window: Float = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(n) / Float(numtaps - 1))
            h[n] = 2.0 * cutoff * sinc * window
        }
        let sum = h.reduce(0, +)
        return h.map { $0 / sum }
    }()

    /// Upsample by 2 using zero-insertion + lowpass FIR (Accelerate-optimized).
    private func resample2x(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let samples = x.asArray(Float.self)

        let h = DemucsGraph.firKernel
        let numtaps = h.count
        let upT = 2 * t
        let pad = numtaps / 2
        let paddedLen = upT + 2 * pad

        var out = [Float](repeating: 0, count: b * c * upT)

        for bc in 0..<(b * c) {
            let inBase = bc * t

            // Zero-insertion upsample
            var up = [Float](repeating: 0, count: upT)
            for i in 0..<t {
                up[2 * i] = samples[inBase + i]
            }

            // Reflect-pad
            var padded = [Float](repeating: 0, count: paddedLen)
            for i in 0..<pad {
                padded[i] = up[min(upT - 1, max(0, pad - i))]
            }
            padded.withUnsafeMutableBufferPointer { dst in
                up.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: pad).update(from: src.baseAddress!, count: upT)
                }
            }
            for i in 0..<pad {
                padded[pad + upT + i] = up[min(upT - 1, max(0, upT - 2 - i))]
            }

            // FIR convolution via vDSP, then scale by 2.0
            let outBase = bc * upT
            padded.withUnsafeBufferPointer { pBuf in
                h.withUnsafeBufferPointer { hBuf in
                    out.withUnsafeMutableBufferPointer { oBuf in
                        vDSP_conv(pBuf.baseAddress!, 1,
                                  hBuf.baseAddress! + numtaps - 1, -1,
                                  oBuf.baseAddress! + outBase, 1,
                                  vDSP_Length(upT), vDSP_Length(numtaps))
                    }
                }
            }
            // Scale by upsample factor
            var scale: Float = 2.0
            out.withUnsafeMutableBufferPointer { oBuf in
                vDSP_vsmul(oBuf.baseAddress! + outBase, 1, &scale,
                           oBuf.baseAddress! + outBase, 1, vDSP_Length(upT))
            }
        }

        return MLXArray(out).reshaped([b, c, upT])
    }

    /// Downsample by 2 using lowpass FIR + decimation (Accelerate-optimized).
    private func resampleHalf(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let samples = x.asArray(Float.self)

        let h = DemucsGraph.firKernel
        let numtaps = h.count
        let pad = numtaps / 2
        let outT = (t + 1) / 2
        let paddedLen = t + 2 * pad

        var out = [Float](repeating: 0, count: b * c * outT)

        for bc in 0..<(b * c) {
            let inBase = bc * t

            // Reflect-pad
            var padded = [Float](repeating: 0, count: paddedLen)
            for i in 0..<pad {
                padded[i] = samples[inBase + min(t - 1, max(0, pad - i))]
            }
            padded.withUnsafeMutableBufferPointer { dst in
                samples.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: pad).update(from: src.baseAddress! + inBase, count: t)
                }
            }
            for i in 0..<pad {
                padded[pad + t + i] = samples[inBase + min(t - 1, max(0, t - 2 - i))]
            }

            // FIR convolution + decimation via vDSP_desamp
            let outBase = bc * outT
            padded.withUnsafeBufferPointer { pBuf in
                h.withUnsafeBufferPointer { hBuf in
                    out.withUnsafeMutableBufferPointer { oBuf in
                        vDSP_desamp(pBuf.baseAddress!, 2,
                                    hBuf.baseAddress!,
                                    oBuf.baseAddress! + outBase,
                                    vDSP_Length(outT), vDSP_Length(numtaps))
                    }
                }
            }
        }

        return MLXArray(out).reshaped([b, c, outT])
    }

    // MARK: - Forward Pass

    func callAsFunction(_ mix: MLXArray) -> MLXArray {
        try! forward(mix, monitor: nil)
    }

    func forward(_ mix: MLXArray, monitor: SeparationMonitor?) throws -> MLXArray {
        // Sub-step progress: encoder×depth + LSTM + decoder×depth
        let totalSteps = Float(config.depth + 1 + config.depth)
        var step: Float = 0

        var x = mix
        let length = x.dim(-1)

        // Normalize
        var meanVal = MLXArray(0.0)
        var stdVal = MLXArray(1.0)
        if config.normalize {
            let mono = mean(x, axis: 1, keepDims: true)
            meanVal = mean(mono, axis: -1, keepDims: true)
            stdVal = std(mono, axis: -1, keepDims: true)
            x = (x - meanVal) / (MLXArray(1e-5) + stdVal)
        }

        // Pad to valid length
        let delta = validLength(length) - length
        if delta > 0 {
            let padLeft = delta / 2
            let padRight = delta - padLeft
            var widths = [IntOrPair](repeating: 0, count: x.ndim)
            widths[widths.count - 1] = IntOrPair((padLeft, padRight))
            x = padded(x, widths: widths, mode: .constant)
        }

        // Resample 2x up
        if config.resample {
            x = resample2x(x)
        }
        // Encode
        var saved: [MLXArray] = []
        for (idx, enc) in encoder.enumerated() {
            monitor?.reportProgress(step / totalSteps, stage: "Encoder \(idx + 1)/\(config.depth)")
            try monitor?.checkCancellation()
            x = enc(x)
            saved.append(x)
            step += 1
        }

        monitor?.reportProgress(step / totalSteps, stage: "LSTM")
        try monitor?.checkCancellation()

        // Bottleneck LSTM
        if let lstmMod = lstm {
            x = lstmMod(x)
        }

        // Decode with skip connections
        step += 1
        for (idx, dec) in decoder.enumerated() {
            monitor?.reportProgress(step / totalSteps, stage: "Decoder \(idx + 1)/\(config.depth)")
            try monitor?.checkCancellation()
            var skip = saved.removeLast()
            skip = demucsCenterTrim(skip, referenceLength: x.dim(-1))
            x = dec(x + skip)
            step += 1
        }

        // Resample 2x down
        if config.resample {
            x = resampleHalf(x)
        }

        // Denormalize
        x = x * stdVal + meanVal

        // Center trim to original length
        x = demucsCenterTrim(x, referenceLength: length)

        // Reshape to (B, sources, channels, T)
        let b = x.dim(0)
        let s = config.sources.count
        x = x.reshaped([b, s, config.audioChannels, -1])

        return x
    }
}
