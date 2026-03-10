import Foundation
import MLX
import MLXNN

struct HTDemucsRuntimeConfig {
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
    let dconvInit: Float
    let bottomChannels: Int
    let tLayers: Int
    let tHiddenScale: Float
    let tHeads: Int
    let tNormFirst: Bool
    let tNormOut: Bool
    let tCrossFirst: Bool
    let tLayerScale: Bool
    let tWeightPosEmbed: Float
    let tSinRandomShift: Int
    let tMaxPeriod: Float
    let samplerate: Int
    let segment: Float
    let useTrainSegment: Bool

    static func fromJSON(_ json: [String: Any]) throws -> HTDemucsRuntimeConfig {
        // Accept either { "kwargs": {...} } or direct kwargs dict
        let kwargs = json["kwargs"] as? [String: Any] ?? json

        func int(_ key: String, _ fallback: Int) -> Int {
            if let v = kwargs[key] as? Int { return v }
            if let v = kwargs[key] as? Double { return Int(v) }
            if let v = kwargs[key] as? String, let i = Int(v) { return i }
            return fallback
        }

        func bool(_ key: String, _ fallback: Bool) -> Bool {
            if let v = kwargs[key] as? Bool { return v }
            if let v = kwargs[key] as? Int { return v != 0 }
            if let v = kwargs[key] as? Double { return v != 0 }
            if let v = kwargs[key] as? String {
                return ["1", "true", "yes"].contains(v.lowercased())
            }
            return fallback
        }

        func double(_ key: String, _ fallback: Double) -> Double {
            if let v = kwargs[key] as? Double { return v }
            if let v = kwargs[key] as? Int { return Double(v) }
            if let v = kwargs[key] as? String {
                if let d = Double(v) { return d }
                let parts = v.split(separator: "/")
                if parts.count == 2,
                   let n = Double(parts[0]),
                   let d = Double(parts[1]),
                   d != 0 {
                    return n / d
                }
            }
            return fallback
        }

        let sources = (kwargs["sources"] as? [String]) ?? ["drums", "bass", "other", "vocals"]

        return HTDemucsRuntimeConfig(
            sources: sources,
            audioChannels: int("audio_channels", 2),
            channels: int("channels", 48),
            channelsTime: kwargs["channels_time"] as? Int,
            growth: int("growth", 2),
            nFFT: int("nfft", 4096),
            wienerIters: int("wiener_iters", 0),
            cac: bool("cac", true),
            depth: int("depth", 4),
            rewrite: bool("rewrite", true),
            freqEmb: Float(double("freq_emb", 0.2)),
            embScale: Float(double("emb_scale", 10.0)),
            embSmooth: bool("emb_smooth", true),
            kernelSize: int("kernel_size", 8),
            timeStride: int("time_stride", 2),
            stride: int("stride", 4),
            context: int("context", 1),
            contextEnc: int("context_enc", 0),
            normStarts: int("norm_starts", 4),
            normGroups: int("norm_groups", 4),
            dconvMode: int("dconv_mode", 3),
            dconvDepth: int("dconv_depth", 2),
            dconvComp: Float(double("dconv_comp", 8.0)),
            dconvInit: Float(double("dconv_init", 1e-3)),
            bottomChannels: int("bottom_channels", 0),
            tLayers: int("t_layers", 5),
            tHiddenScale: Float(double("t_hidden_scale", 4.0)),
            tHeads: int("t_heads", 8),
            tNormFirst: bool("t_norm_first", true),
            tNormOut: bool("t_norm_out", true),
            tCrossFirst: bool("t_cross_first", false),
            tLayerScale: bool("t_layer_scale", true),
            tWeightPosEmbed: Float(double("t_weight_pos_embed", 1.0)),
            tSinRandomShift: int("t_sin_random_shift", 0),
            tMaxPeriod: Float(double("t_max_period", 10_000.0)),
            samplerate: int("samplerate", 44_100),
            segment: Float(double("segment", 8.0)),
            useTrainSegment: bool("use_train_segment", true)
        )
    }
}

final class ScaledEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    let scale: Float

    init(numEmbeddings: Int, embeddingDim: Int, scale: Float) {
        self.scale = scale
        self._embedding.wrappedValue = Embedding(
            embeddingCount: numEmbeddings,
            dimensions: embeddingDim
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embedding(x) * MLXArray(scale)
    }
}

final class HTDemucsGraph: Module {
    let config: HTDemucsRuntimeConfig
    let hopLength: Int
    let freqEmbScale: Float

    @ModuleInfo(key: "encoder") var encoder: [HEncLayer]
    @ModuleInfo(key: "decoder") var decoder: [HDecLayer]
    @ModuleInfo(key: "tencoder") var tencoder: [HEncLayer]
    @ModuleInfo(key: "tdecoder") var tdecoder: [HDecLayer]

    @ModuleInfo(key: "freq_emb") var freqEmb: ScaledEmbedding

    @ModuleInfo(key: "channel_upsampler") var channelUpsampler: Conv1dNCL?
    @ModuleInfo(key: "channel_downsampler") var channelDownsampler: Conv1dNCL?
    @ModuleInfo(key: "channel_upsampler_t") var channelUpsamplerT: Conv1dNCL?
    @ModuleInfo(key: "channel_downsampler_t") var channelDownsamplerT: Conv1dNCL?

    @ModuleInfo(key: "crosstransformer") var crosstransformer: CrossTransformerEncoder

    let spectral: DemucsSpectralPair

    init(config: HTDemucsRuntimeConfig) {
        self.config = config
        self.hopLength = config.nFFT / 4
        self.freqEmbScale = config.freqEmb

        var encoders: [HEncLayer] = []
        var decoders: [HDecLayer] = []
        var timeEncoders: [HEncLayer] = []
        var timeDecoders: [HDecLayer] = []

        var chin = config.audioChannels
        var chinZ = config.cac ? chin * 2 : chin
        var chout = config.channelsTime ?? config.channels
        var choutZ = config.channels
        var freqs = config.nFFT / 2

        var transformerChannels = config.channels
        for _ in 1..<config.depth {
            transformerChannels *= config.growth
        }

        self._freqEmb.wrappedValue = ScaledEmbedding(
            numEmbeddings: max(1, freqs / max(1, config.stride)),
            embeddingDim: config.channels,
            scale: config.embScale
        )

        var crossDim = transformerChannels
        if config.bottomChannels > 0 {
            self._channelUpsampler.wrappedValue = Conv1dNCL(transformerChannels, config.bottomChannels, kernelSize: 1)
            self._channelDownsampler.wrappedValue = Conv1dNCL(config.bottomChannels, transformerChannels, kernelSize: 1)
            self._channelUpsamplerT.wrappedValue = Conv1dNCL(transformerChannels, config.bottomChannels, kernelSize: 1)
            self._channelDownsamplerT.wrappedValue = Conv1dNCL(config.bottomChannels, transformerChannels, kernelSize: 1)
            crossDim = config.bottomChannels
        } else {
            self._channelUpsampler.wrappedValue = nil
            self._channelDownsampler.wrappedValue = nil
            self._channelUpsamplerT.wrappedValue = nil
            self._channelDownsamplerT.wrappedValue = nil
        }

        self._crosstransformer.wrappedValue = CrossTransformerEncoder(
            dim: crossDim,
            hiddenScale: config.tHiddenScale,
            numHeads: config.tHeads,
            numLayers: config.tLayers,
            crossFirst: config.tCrossFirst,
            normFirst: config.tNormFirst,
            normOut: config.tNormOut,
            maxPeriod: config.tMaxPeriod,
            weightPosEmbed: config.tWeightPosEmbed,
            layerScale: config.tLayerScale,
            sinRandomShift: config.tSinRandomShift
        )

        for index in 0..<config.depth {
            let norm = index >= config.normStarts
            let freq = freqs > 1

            var currentKernel = config.kernelSize
            var currentStride = config.stride
            if !freq {
                currentKernel = config.timeStride * 2
                currentStride = config.timeStride
            }

            var pad = true
            if freq && freqs <= config.kernelSize {
                currentKernel = freqs
                pad = false
            }

            let enc = HEncLayer(
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
                rewrite: config.rewrite
            )
            encoders.append(enc)

            if freq {
                let tenc = HEncLayer(
                    inputChannels: chin,
                    outputChannels: chout,
                    kernelSize: config.kernelSize,
                    stride: config.stride,
                    normGroups: config.normGroups,
                    empty: false,
                    freq: false,
                    dconvEnabled: (config.dconvMode & 1) != 0,
                    normEnabled: norm,
                    context: config.contextEnc,
                    dconvDepth: config.dconvDepth,
                    dconvComp: config.dconvComp,
                    dconvInit: config.dconvInit,
                    pad: true,
                    rewrite: config.rewrite
                )
                timeEncoders.append(tenc)
            }

            if index == 0 {
                chin = config.audioChannels * config.sources.count
                chinZ = config.cac ? chin * 2 : chin
            }

            let dec = HDecLayer(
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
                contextFreq: true,
                rewrite: config.rewrite
            )
            decoders.insert(dec, at: 0)

            if freq {
                let tdec = HDecLayer(
                    inputChannels: chout,
                    outputChannels: chin,
                    last: index == 0,
                    kernelSize: config.kernelSize,
                    stride: config.stride,
                    normGroups: config.normGroups,
                    empty: false,
                    freq: false,
                    dconvEnabled: (config.dconvMode & 2) != 0,
                    normEnabled: norm,
                    context: config.context,
                    dconvDepth: config.dconvDepth,
                    dconvComp: config.dconvComp,
                    dconvInit: config.dconvInit,
                    pad: true,
                    contextFreq: true,
                    rewrite: config.rewrite
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
        }

        self._encoder.wrappedValue = encoders
        self._decoder.wrappedValue = decoders
        self._tencoder.wrappedValue = timeEncoders
        self._tdecoder.wrappedValue = timeDecoders

        self.spectral = DemucsSpectralPair(nFFT: config.nFFT, hopLength: hopLength, center: true)

        super.init()
    }

    private func reflectPad1D3D(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let src = x.asArray(Float.self)

        let outT = t + left + right
        var out = [Float](repeating: 0, count: b * c * outT)

        for bc in 0..<(b * c) {
            let inBase = bc * t
            let outBase = bc * outT

            let signal = Array(src[inBase..<(inBase + t)])
            var padded = [Float]()
            padded.reserveCapacity(outT)

            if left > 0 {
                for i in 0..<left {
                    let idx = max(0, min(signal.count - 1, left - i))
                    padded.append(signal[idx])
                }
            }
            padded.append(contentsOf: signal)
            if right > 0 {
                for i in 0..<right {
                    let idx = max(0, min(signal.count - 1, signal.count - 2 - i))
                    padded.append(signal[idx])
                }
            }

            out.replaceSubrange(outBase..<(outBase + outT), with: padded)
        }

        return MLXArray(out).reshaped([b, c, outT])
    }

    private func spec(_ x: MLXArray) -> DemucsComplexSpectrogram {
        let hl = hopLength
        let length = x.dim(-1)
        let le = Int(ceil(Double(length) / Double(hl)))
        let pad = (hl / 2) * 3

        let padded = reflectPad1D3D(x, left: pad, right: pad + le * hl - length)
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
            real = padded(real, widths: widths, mode: .constant)
            imag = padded(imag, widths: widths, mode: .constant)
        } else {
            let widths: [IntOrPair] = [0, 0, IntOrPair((0, 1)), IntOrPair((2, 2))]
            real = padded(real, widths: widths, mode: .constant)
            imag = padded(imag, widths: widths, mode: .constant)
        }

        let pad = (hl / 2) * 3
        let le = hl * Int(ceil(Double(length) / Double(hl))) + 2 * pad

        var x = spectral.istft(DemucsComplexSpectrogram(real: real, imag: imag), length: le)
        x = x[0..., 0..., 0..., pad..<(pad + length)]
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
        return DemucsComplexSpectrogram(real: z.real, imag: z.imag)
    }

    func callAsFunction(_ mix: MLXArray) -> MLXArray {
        let profilingEnabled = ProcessInfo.processInfo.environment["DEMUCS_MLX_SWIFT_PROFILE"] == "1"
        let tStart = profilingEnabled ? CFAbsoluteTimeGetCurrent() : 0

        var mix = mix
        let originalLength = mix.dim(-1)

        var lengthPrePad: Int? = nil
        let trainingLength = Int(config.segment * Float(config.samplerate))
        if config.useTrainSegment, mix.dim(-1) < trainingLength {
            lengthPrePad = mix.dim(-1)
            let padRight = trainingLength - lengthPrePad!
            let widths: [IntOrPair] = [0, 0, IntOrPair((0, padRight))]
            mix = padded(mix, widths: widths, mode: .constant)
        }

        let z = spec(mix)
        let tAfterSpec = profilingEnabled ? CFAbsoluteTimeGetCurrent() : 0
        var x = magnitude(z)

        let meanF = mean(x, axes: [1, 2, 3], keepDims: true)
        let stdv = std(x, axes: [1, 2, 3], keepDims: true)
        x = (x - meanF) / (MLXArray(1e-5) + stdv)

        var xt = mix
        let meant = mean(xt, axes: [1, 2], keepDims: true)
        let stdt = std(xt, axes: [1, 2], keepDims: true)
        xt = (xt - meant) / (MLXArray(1e-5) + stdt)

        var saved: [MLXArray] = []
        var savedT: [MLXArray] = []
        var lengths: [Int] = []
        var lengthsT: [Int] = []

        for idx in 0..<encoder.count {
            lengths.append(x.dim(-1))
            lengthsT.append(xt.dim(-1))

            xt = tencoder[idx](xt, inject: nil)
            savedT.append(xt)

            x = encoder[idx](x, inject: nil)
            if idx == 0 {
                let frs = MLXArray(0..<x.dim(-2)).asType(.int32)
                var emb = freqEmb(frs).transposed(1, 0)
                emb = emb.expandedDimensions(axis: 0)
                emb = emb.expandedDimensions(axis: 3)
                x = x + MLXArray(freqEmbScale) * emb
            }

            saved.append(x)
        }

        if config.tLayers > 0 {
            if config.bottomChannels > 0 {
                let b = x.dim(0)
                let c = x.dim(1)
                let f = x.dim(2)
                let t = x.dim(3)

                x = x.reshaped([b, c, f * t])
                x = channelUpsampler!(x)
                x = x.reshaped([b, config.bottomChannels, f, t])
                xt = channelUpsamplerT!(xt)

                let out = crosstransformer(x, xt)
                x = out.0
                xt = out.1

                x = x.reshaped([b, config.bottomChannels, f * t])
                x = channelDownsampler!(x)
                x = x.reshaped([b, c, f, t])
                xt = channelDownsamplerT!(xt)
            } else {
                let out = crosstransformer(x, xt)
                x = out.0
                xt = out.1
            }
        }
        let tAfterCore = profilingEnabled ? CFAbsoluteTimeGetCurrent() : 0

        let offset = config.depth - tdecoder.count
        for idx in 0..<decoder.count {
            let skip = saved.removeLast()
            let length = lengths.removeLast()
            let decoded = decoder[idx](x, skip: skip, length: length)
            x = decoded.0

            if idx >= offset {
                let lengthT = lengthsT.removeLast()
                let skipT = savedT.removeLast()
                let tdecoded = tdecoder[idx - offset](xt, skip: skipT, length: lengthT)
                xt = tdecoded.0
            }
        }

        let b = x.dim(0)
        let s = config.sources.count
        let f = x.dim(2)
        let t = x.dim(3)

        x = x.reshaped([b, s, -1, f, t])
        x = x * stdv.expandedDimensions(axis: 1) + meanF.expandedDimensions(axis: 1)

        let zout = mask(z, m: x)

        let targetLength = (config.useTrainSegment && lengthPrePad != nil) ? trainingLength : originalLength
        var xWave = ispec(zout, length: targetLength)
        let tAfterISpec = profilingEnabled ? CFAbsoluteTimeGetCurrent() : 0

        let actualLength = xt.dim(-1)
        xt = xt.reshaped([b, s, -1, actualLength])
        xt = xt * stdt.expandedDimensions(axis: 1) + meant.expandedDimensions(axis: 1)

        if xWave.dim(-1) != xt.dim(-1) {
            if xWave.dim(-1) > xt.dim(-1) {
                xWave = demucsCenterTrim(xWave, referenceLength: xt.dim(-1))
            } else {
                xt = demucsCenterTrim(xt, referenceLength: xWave.dim(-1))
            }
        }

        xWave = xt + xWave

        xWave = xWave[0..., 0..., 0..., 0..<targetLength]

        if let lengthPrePad {
            xWave = xWave[0..., 0..., 0..., 0..<lengthPrePad]
        }

        if profilingEnabled {
            let tDone = CFAbsoluteTimeGetCurrent()
            let specMs = (tAfterSpec - tStart) * 1000
            let coreMs = (tAfterCore - tAfterSpec) * 1000
            let ispecMs = (tAfterISpec - tAfterCore) * 1000
            let tailMs = (tDone - tAfterISpec) * 1000
            let totalMs = (tDone - tStart) * 1000
            fputs(
                String(
                    format: "HTDemucs profile ms: spec=%.2f core=%.2f ispec=%.2f tail=%.2f total=%.2f\n",
                    specMs, coreMs, ispecMs, tailMs, totalMs
                ),
                stderr
            )
        }

        return xWave
    }
}

/// Legacy entry point that creates an HTDemucs model using the shared ModelLoader.
/// Kept for backward compatibility; new code should use DemucsModelFactory.
final class NativeHTDemucsModel: StemSeparationModel {
    let descriptor: DemucsModelDescriptor
    private let graph: HTDemucsGraph

    init(descriptor: DemucsModelDescriptor, modelDirectory: URL?) throws {
        let directory = try ModelLoader.resolveModelDirectory(modelName: descriptor.name, preferred: modelDirectory)
        let config = try ModelLoader.loadConfig(from: directory, modelName: descriptor.name)

        let cfg = try HTDemucsRuntimeConfig.fromJSON(config)
        self.descriptor = DemucsModelDescriptor(
            name: descriptor.name,
            sourceNames: cfg.sources,
            sampleRate: cfg.samplerate,
            audioChannels: cfg.audioChannels,
            defaultSegmentSeconds: Double(cfg.segment)
        )
        self.graph = HTDemucsGraph(config: cfg)

        let loaded = try ModelLoader.loadWeights(from: directory, modelName: descriptor.name)
        var stripped: [String: MLXArray] = [:]
        stripped.reserveCapacity(loaded.count)

        for (key, value) in loaded {
            if key.hasPrefix("model_0.") {
                stripped[String(key.dropFirst("model_0.".count))] = value
            }
        }

        // If no model_0. prefix found, use weights as-is
        if stripped.isEmpty {
            stripped = loaded
        }

        try graph.update(parameters: ModuleParameters.unflattened(stripped), verify: .all)
        MLX.eval(graph.parameters())
    }

    func predict(
        batchData: [Float],
        batchSize: Int,
        channels: Int,
        frames: Int
    ) throws -> [Float] {
        let input = MLXArray(batchData).reshaped([batchSize, channels, frames])
        let output = graph(input)
        MLX.eval(output)
        return output.asArray(Float.self)
    }
}
