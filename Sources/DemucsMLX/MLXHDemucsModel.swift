import Foundation
import MLX
import MLXNN

// MARK: - HDemucs Runtime Config

struct HDemucsRuntimeConfig {
    let sources: [String]
    let audioChannels: Int
    let channels: Int
    let channelsTime: Int?
    let growth: Int
    let nFFT: Int
    let wienerIters: Int
    let cac: Bool
    let depth: Int
    let rewrite: Bool
    let hybrid: Bool
    let hybridOld: Bool
    let multiFreqs: [Float]
    let multiFreqsDepth: Int
    let freqEmb: Float
    let embScale: Float
    let embSmooth: Bool
    let kernelSize: Int
    let timeStride: Int
    let stride: Int
    let context: Int
    let contextEnc: Int
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

    static func fromJSON(_ json: [String: Any]) throws -> HDemucsRuntimeConfig {
        let kwargs = json["kwargs"] as? [String: Any] ?? json
        let i = ModelLoader.int
        let b = ModelLoader.bool
        let d = ModelLoader.double

        let multiFreqs: [Float]
        if let arr = kwargs["multi_freqs"] as? [Any] {
            multiFreqs = arr.compactMap { ($0 as? NSNumber)?.floatValue }
        } else {
            multiFreqs = []
        }

        return HDemucsRuntimeConfig(
            sources: ModelLoader.sources(kwargs),
            audioChannels: i(kwargs, "audio_channels", 2),
            channels: i(kwargs, "channels", 48),
            channelsTime: kwargs["channels_time"] as? Int,
            growth: i(kwargs, "growth", 2),
            nFFT: i(kwargs, "nfft", 4096),
            wienerIters: i(kwargs, "wiener_iters", 0),
            cac: b(kwargs, "cac", true),
            depth: i(kwargs, "depth", 6),
            rewrite: b(kwargs, "rewrite", true),
            hybrid: b(kwargs, "hybrid", true),
            hybridOld: b(kwargs, "hybrid_old", false),
            multiFreqs: multiFreqs,
            multiFreqsDepth: i(kwargs, "multi_freqs_depth", 2),
            freqEmb: Float(d(kwargs, "freq_emb", 0.2)),
            embScale: Float(d(kwargs, "emb_scale", 10.0)),
            embSmooth: b(kwargs, "emb_smooth", true),
            kernelSize: i(kwargs, "kernel_size", 8),
            timeStride: i(kwargs, "time_stride", 2),
            stride: i(kwargs, "stride", 4),
            context: i(kwargs, "context", 1),
            contextEnc: i(kwargs, "context_enc", 0),
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

// MARK: - HDemucs Graph (v3 Hybrid, no transformer)

final class HDemucsGraph: Module {
    let config: HDemucsRuntimeConfig
    let hopLength: Int
    let freqEmbScale: Float

    @ModuleInfo(key: "encoder") var encoder: [Module]
    @ModuleInfo(key: "decoder") var decoder: [Module]
    @ModuleInfo(key: "tencoder") var tencoder: [HEncLayer]
    @ModuleInfo(key: "tdecoder") var tdecoder: [HDecLayer]

    @ModuleInfo(key: "freq_emb") var freqEmb: ScaledEmbedding?

    let spectral: DemucsSpectralPair

    init(config: HDemucsRuntimeConfig) {
        self.config = config
        self.hopLength = config.nFFT / 4
        self.freqEmbScale = config.freqEmb

        var encoders: [Module] = []
        var decoders: [Module] = []
        var timeEncoders: [HEncLayer] = []
        var timeDecoders: [HDecLayer] = []
        let multiFreqs = config.multiFreqs

        var chin = config.audioChannels
        var chinZ = config.cac ? chin * 2 : chin
        var chout = config.channelsTime ?? config.channels
        var choutZ = config.channels
        var freqs = config.nFFT / 2

        for index in 0..<config.depth {
            let norm = index >= config.normStarts
            let freq = freqs > 1
            let lstm = index >= config.dconvLstm
            let attn = index >= config.dconvAttn

            var currentKernel = config.kernelSize
            var currentStride = config.stride
            if !freq {
                currentKernel = config.timeStride * 2
                currentStride = config.timeStride
            }

            var pad = true
            var lastFreq = false
            if freq && freqs <= config.kernelSize {
                currentKernel = freqs
                pad = false
                lastFreq = true
            }

            if lastFreq {
                choutZ = max(chout, choutZ)
                chout = choutZ
            }

            let encParams = HEncLayerParams(
                inputChannels: chinZ,
                outputChannels: choutZ,
                kernelSize: currentKernel,
                stride: currentStride,
                normGroups: config.normGroups,
                empty: false,
                freq: freq,
                dconvEnabled: (config.dconvMode & 1) != 0,
                normEnabled: norm,
                context: config.contextEnc,
                dconvDepth: config.dconvDepth,
                dconvComp: config.dconvComp,
                dconvInit: config.dconvInit,
                pad: pad,
                rewrite: config.rewrite,
                dconvLstm: lstm,
                dconvAttn: attn
            )
            let useMulti = !multiFreqs.isEmpty && index < config.multiFreqsDepth
            if useMulti {
                encoders.append(MultiWrapEnc(params: encParams, splitRatios: multiFreqs))
            } else {
                let enc = HEncLayer(
                    inputChannels: encParams.inputChannels,
                    outputChannels: encParams.outputChannels,
                    kernelSize: encParams.kernelSize,
                    stride: encParams.stride,
                    normGroups: encParams.normGroups,
                    empty: encParams.empty,
                    freq: encParams.freq,
                    dconvEnabled: encParams.dconvEnabled,
                    normEnabled: encParams.normEnabled,
                    context: encParams.context,
                    dconvDepth: encParams.dconvDepth,
                    dconvComp: encParams.dconvComp,
                    dconvInit: encParams.dconvInit,
                    pad: encParams.pad,
                    rewrite: encParams.rewrite,
                    dconvLstm: encParams.dconvLstm,
                    dconvAttn: encParams.dconvAttn
                )
                encoders.append(enc)
            }

            if config.hybrid && freq {
                let tenc = HEncLayer(
                    inputChannels: chin,
                    outputChannels: chout,
                    kernelSize: config.kernelSize,
                    stride: config.stride,
                    normGroups: config.normGroups,
                    empty: lastFreq,
                    freq: false,
                    dconvEnabled: (config.dconvMode & 1) != 0,
                    normEnabled: norm,
                    context: config.contextEnc,
                    dconvDepth: config.dconvDepth,
                    dconvComp: config.dconvComp,
                    dconvInit: config.dconvInit,
                    pad: true,
                    rewrite: config.rewrite,
                    dconvLstm: lstm,
                    dconvAttn: attn
                )
                timeEncoders.append(tenc)
            }

            if index == 0 {
                chin = config.audioChannels * config.sources.count
                chinZ = config.cac ? chin * 2 : chin
            }

            let decParams = HDecLayerParams(
                inputChannels: choutZ,
                outputChannels: chinZ,
                last: index == 0,
                kernelSize: currentKernel,
                stride: currentStride,
                normGroups: config.normGroups,
                empty: false,
                freq: freq,
                dconvEnabled: (config.dconvMode & 2) != 0,
                normEnabled: norm,
                context: config.context,
                dconvDepth: config.dconvDepth,
                dconvComp: config.dconvComp,
                dconvInit: config.dconvInit,
                pad: pad,
                contextFreq: !useMulti,
                rewrite: config.rewrite,
                dconvLstm: lstm,
                dconvAttn: attn
            )
            if useMulti {
                decoders.insert(MultiWrapDec(params: decParams, splitRatios: multiFreqs), at: 0)
            } else {
                let dec = HDecLayer(
                    inputChannels: decParams.inputChannels,
                    outputChannels: decParams.outputChannels,
                    last: decParams.last,
                    kernelSize: decParams.kernelSize,
                    stride: decParams.stride,
                    normGroups: decParams.normGroups,
                    empty: decParams.empty,
                    freq: decParams.freq,
                    dconvEnabled: decParams.dconvEnabled,
                    normEnabled: decParams.normEnabled,
                    context: decParams.context,
                    dconvDepth: decParams.dconvDepth,
                    dconvComp: decParams.dconvComp,
                    dconvInit: decParams.dconvInit,
                    pad: decParams.pad,
                    contextFreq: decParams.contextFreq,
                    rewrite: decParams.rewrite,
                    dconvLstm: decParams.dconvLstm,
                    dconvAttn: decParams.dconvAttn
                )
                decoders.insert(dec, at: 0)
            }

            if config.hybrid && freq {
                let tdec = HDecLayer(
                    inputChannels: chout,
                    outputChannels: chin,
                    last: index == 0,
                    kernelSize: config.kernelSize,
                    stride: config.stride,
                    normGroups: config.normGroups,
                    empty: lastFreq,
                    freq: false,
                    dconvEnabled: (config.dconvMode & 2) != 0,
                    normEnabled: norm,
                    context: config.context,
                    dconvDepth: config.dconvDepth,
                    dconvComp: config.dconvComp,
                    dconvInit: config.dconvInit,
                    pad: true,
                    contextFreq: true,
                    rewrite: config.rewrite,
                    dconvLstm: lstm,
                    dconvAttn: attn
                )
                timeDecoders.insert(tdec, at: 0)
            }

            chin = chout
            chinZ = choutZ
            chout *= config.growth
            choutZ *= config.growth

            if freq {
                if freqs <= config.kernelSize {
                    freqs = 1
                } else {
                    freqs /= max(1, config.stride)
                }
            }

            if index == 0 && config.freqEmb > 0 {
                self._freqEmb.wrappedValue = ScaledEmbedding(
                    numEmbeddings: freqs,
                    embeddingDim: chinZ,
                    scale: config.embScale
                )
            }
        }

        if config.freqEmb <= 0 {
            self._freqEmb.wrappedValue = nil
        }

        self._encoder.wrappedValue = encoders
        self._decoder.wrappedValue = decoders
        self._tencoder.wrappedValue = timeEncoders
        self._tdecoder.wrappedValue = timeDecoders

        self.spectral = DemucsSpectralPair(nFFT: config.nFFT, hopLength: hopLength, center: true)

        super.init()
    }

    // MARK: - Spectral Helpers

    /// GPU-native reflect padding along the temporal (last) axis of a [B, C, T] tensor.
    private func reflectPad1D3D(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        let t = x.dim(2)
        var indices = [Int32]()
        indices.reserveCapacity(t + left + right)
        for i in stride(from: left, through: 1, by: -1) {
            indices.append(Int32(min(i, t - 1)))
        }
        for i in 0..<t {
            indices.append(Int32(i))
        }
        for i in 0..<right {
            indices.append(Int32(max(0, t - 2 - i)))
        }
        return x.take(MLXArray(indices), axis: 2)
    }

    private func spec(_ x: MLXArray) -> DemucsComplexSpectrogram {
        let hl = hopLength
        let length = x.dim(-1)
        let le = Int(ceil(Double(length) / Double(hl)))
        let pad = (hl / 2) * 3

        let padded: MLXArray
        if !config.hybridOld {
            padded = reflectPad1D3D(x, left: pad, right: pad + le * hl - length)
        } else {
            // hybrid_old uses constant (zero) padding
            var widths = [IntOrPair](repeating: 0, count: x.ndim)
            widths[widths.count - 1] = IntOrPair((pad, pad + le * hl - length))
            padded = MLX.padded(x, widths: widths, mode: .constant)
        }
        var z = spectral.stft(padded)

        z = DemucsComplexSpectrogram(
            real: z.real[0..., 0..., 0..<(z.real.dim(2) - 1), 0...],
            imag: z.imag[0..., 0..., 0..<(z.imag.dim(2) - 1), 0...]
        )

        let start = 2
        let end = 2 + le
        return DemucsComplexSpectrogram(
            real: z.real[0..., 0..., 0..., start..<end],
            imag: z.imag[0..., 0..., 0..., start..<end]
        )
    }

    private func ispec(_ z: DemucsComplexSpectrogram, length: Int) -> MLXArray {
        let hl = hopLength
        var real = z.real
        var imag = z.imag

        if z.real.ndim == 5 {
            let widths: [IntOrPair] = [0, 0, 0, IntOrPair((0, 1)), IntOrPair((2, 2))]
            real = MLX.padded(real, widths: widths, mode: .constant)
            imag = MLX.padded(imag, widths: widths, mode: .constant)
        } else {
            let widths: [IntOrPair] = [0, 0, IntOrPair((0, 1)), IntOrPair((2, 2))]
            real = MLX.padded(real, widths: widths, mode: .constant)
            imag = MLX.padded(imag, widths: widths, mode: .constant)
        }

        let pad = (hl / 2) * 3
        let le = hl * Int(ceil(Double(length) / Double(hl))) + 2 * pad

        var x = spectral.istft(DemucsComplexSpectrogram(real: real, imag: imag), length: le)

        // Trim to original length - handle both 3D [B,C,T] and 4D [B,S,C,T] outputs
        // hybrid_old=True: zero padding → trim from (pad + hl) to align with original signal
        // hybrid_old=False: reflect padding → trim from pad (verified working)
        let trimStart = config.hybridOld ? (pad + hl) : pad
        if x.ndim == 4 {
            x = x[0..., 0..., 0..., trimStart..<(trimStart + length)]
        } else {
            x = x[0..., 0..., trimStart..<(trimStart + length)]
        }
        return x
    }

    private func magnitude(_ z: DemucsComplexSpectrogram) -> MLXArray {
        if config.cac {
            let b = z.real.dim(0)
            let c = z.real.dim(1)
            let f = z.real.dim(2)
            let t = z.real.dim(3)
            return stacked([z.real, z.imag], axis: 2).reshaped([b, c * 2, f, t])
        }
        return sqrt(z.real * z.real + z.imag * z.imag)
    }

    private func mask(_ z: DemucsComplexSpectrogram, m: MLXArray) -> DemucsComplexSpectrogram {
        if config.cac {
            let b = m.dim(0)
            let s = m.dim(1)
            let f = m.dim(3)
            let t = m.dim(4)
            let ri = m.reshaped([b, s, -1, 2, f, t]).transposed(0, 1, 2, 4, 5, 3)
            let parts = split(ri, parts: 2, axis: 5)
            return DemucsComplexSpectrogram(
                real: parts[0].squeezed(axis: 5),
                imag: parts[1].squeezed(axis: 5)
            )
        }
        // cac=False, wiener_iters=0, softmask=False: apply mix phase to predicted magnitudes.
        // m is [B, S, C, Fr, T] (predicted magnitudes), z is complex [B, C, Fr, T].
        // Matches OpenUnmix wiener() with iterations=0: y = magnitude * exp(i * angle(mix))
        let angle = MLX.atan2(z.imag, z.real).expandedDimensions(axis: 1)  // [B, 1, C, Fr, T]
        return DemucsComplexSpectrogram(
            real: m * cos(angle),  // [B, S, C, Fr, T]
            imag: m * sin(angle)
        )
    }

    // MARK: - Forward Pass

    func callAsFunction(_ mix: MLXArray) -> MLXArray {
        try! forward(mix, monitor: nil)
    }

    func forward(_ mix: MLXArray, monitor: SeparationMonitor?) throws -> MLXArray {
        // Sub-step progress: STFT + encoder×depth + decoder×depth + output
        let totalSteps = Float(1 + config.depth + config.depth + 1)
        var step: Float = 0

        let originalLength = mix.dim(-1)

        monitor?.reportProgress(0, stage: "STFT")
        try monitor?.checkCancellation()

        // Spectral analysis
        let z = spec(mix)

        var x = magnitude(z)

        let b = x.dim(0)
        let fq = x.dim(2)
        let tSpec = x.dim(3)

        // Normalize frequency domain
        let meanF = mean(x, axes: [1, 2, 3], keepDims: true)
        let stdF = std(x, axes: [1, 2, 3], keepDims: true)
        x = (x - meanF) / (MLXArray(1e-5) + stdF)

        // Normalize time domain
        var xt = mix
        var meant = MLXArray(0.0)
        var stdt = MLXArray(1.0)
        if config.hybrid {
            meant = mean(xt, axes: [1, 2], keepDims: true)
            stdt = std(xt, axes: [1, 2], keepDims: true)
            xt = (xt - meant) / (MLXArray(1e-5) + stdt)
        }

        var saved: [MLXArray] = []
        var savedT: [MLXArray] = []
        var lengths: [Int] = []
        var lengthsT: [Int] = []

        // Encoder
        step = 1
        for idx in 0..<encoder.count {
            monitor?.reportProgress(step / totalSteps, stage: "Encoder \(idx + 1)/\(config.depth)")
            try monitor?.checkCancellation()

            lengths.append(x.dim(-1))
            let enc = encoder[idx] as! HEncoderLayer

            var inject: MLXArray? = nil
            if config.hybrid && idx < tencoder.count {
                lengthsT.append(xt.dim(-1))
                let tenc = tencoder[idx]
                xt = tenc(xt, inject: nil)
                if !tenc.empty {
                    savedT.append(xt)
                }
                else {
                    inject = xt
                }
            }

            x = enc(x, inject: inject)

            if idx == 0, let freqEmbMod = freqEmb {
                let frs = MLXArray(0..<x.dim(-2)).asType(.int32)
                var emb = freqEmbMod(frs).transposed(1, 0)
                emb = emb.expandedDimensions(axis: 0)
                emb = emb.expandedDimensions(axis: 3)
                x = x + MLXArray(freqEmbScale) * emb
            }

            saved.append(x)
            step += 1
        }

        try monitor?.checkCancellation()

        // Bottleneck: zero-initialized (key difference from HTDemucs)
        x = MLXArray.zeros(like: x)
        if config.hybrid {
            xt = MLXArray.zeros(like: xt)
        }

        // Decoder
        let offset = config.depth - tdecoder.count
        for idx in 0..<decoder.count {
            monitor?.reportProgress(step / totalSteps, stage: "Decoder \(idx + 1)/\(config.depth)")
            try monitor?.checkCancellation()

            let skip = saved.removeLast()
            let length = lengths.removeLast()
            let dec = decoder[idx] as! HDecoderLayer
            let decoded = dec(x, skip: skip, length: length)
            x = decoded.0
            let pre = decoded.1

            if config.hybrid && idx >= offset {
                let tdec = tdecoder[idx - offset]
                let lengthT = lengthsT.removeLast()

                if tdec.empty {
                    // Collapse freq dimension for empty time decoder
                    let preFlat = pre[0..., 0..., 0, 0...]
                    let tdecoded = tdec(preFlat, skip: MLXArray.zeros([1]), length: lengthT)
                    xt = tdecoded.0
                }
                else {
                    let skipT = savedT.removeLast()
                    let tdecoded = tdec(xt, skip: skipT, length: lengthT)
                    xt = tdecoded.0
                }
            }
            step += 1
        }

        monitor?.reportProgress(step / totalSteps, stage: "Output")
        try monitor?.checkCancellation()

        // Reshape and denormalize frequency output
        let s = config.sources.count
        x = x.reshaped([b, s, -1, fq, tSpec])
        x = x * stdF.expandedDimensions(axis: 1) + meanF.expandedDimensions(axis: 1)

        // Apply mask and convert back to waveform
        let zout = mask(z, m: x)
        var xWave = ispec(zout, length: originalLength)

        // Combine with time domain
        if config.hybrid {
            let actualLength = xt.dim(-1)
            xt = xt.reshaped([b, s, -1, actualLength])
            xt = xt * stdt.expandedDimensions(axis: 1) + meant.expandedDimensions(axis: 1)

            // Align lengths
            if xWave.dim(-1) != xt.dim(-1) {
                if xWave.dim(-1) > xt.dim(-1) {
                    xWave = demucsCenterTrim(xWave, referenceLength: xt.dim(-1))
                }
                else {
                    xt = demucsCenterTrim(xt, referenceLength: xWave.dim(-1))
                }
            }
            xWave = xt + xWave
        }

        return xWave
    }
}
