@preconcurrency import AVFoundation
import Foundation

// MARK: - Audio Output Format

/// Describes the desired output audio encoding.
public enum AudioOutputFormat: Sendable {
    /// Linear PCM inside a WAV container.
    case wav(bitDepth: WAVBitDepth)
    /// Apple Lossless Audio Codec (ALAC), typically in an .m4a or .caf container.
    case alac
    /// FLAC lossless compression.
    case flac
    /// AAC lossy compression (Apple's preferred lossy format), typically in .m4a.
    case aac(bitRate: Int = 256_000)

    /// Default format: 16-bit WAV.
    public static let `default` = AudioOutputFormat.wav(bitDepth: .int16)
}

/// Bit depths supported for WAV output.
public enum WAVBitDepth: Int, Sendable {
    case int16 = 16
    case int24 = 24
    case int32 = 32
    case float32 = 0 // sentinel; we use 32-bit float PCM

    public var bitsPerSample: Int {
        switch self {
        case .int16: return 16
        case .int24: return 24
        case .int32: return 32
        case .float32: return 32
        }
    }
}

// MARK: - AudioIO

public enum AudioIO {

    // MARK: Load

    public static func loadAudio(from url: URL) throws -> DemucsAudio {
        do {
            let file = try AVAudioFile(forReading: url)
            let sourceFormat = file.processingFormat
            let channels = Int(sourceFormat.channelCount)
            guard channels > 0 else {
                throw DemucsError.audioIO("Input file has zero channels")
            }

            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sourceFormat.sampleRate,
                channels: sourceFormat.channelCount,
                interleaved: false
            )
            guard let outputFormat else {
                throw DemucsError.audioIO("Failed to create float32 output format")
            }

            let frameCapacity = AVAudioFrameCount(file.length)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: frameCapacity) else {
                throw DemucsError.audioIO("Failed to allocate audio buffer")
            }

            try file.read(into: buffer)
            guard let channelData = buffer.floatChannelData else {
                throw DemucsError.audioIO("No float channel data available")
            }

            let frames = Int(buffer.frameLength)
            var channelMajor = [Float](repeating: 0, count: channels * frames)
            for c in 0..<channels {
                let src = channelData[c]
                let base = c * frames
                for t in 0..<frames {
                    channelMajor[base + t] = src[t]
                }
            }

            return try DemucsAudio(
                channelMajor: channelMajor,
                channels: channels,
                sampleRate: Int(sourceFormat.sampleRate)
            )
        } catch let err as DemucsError {
            throw err
        } catch {
            throw DemucsError.audioIO(error.localizedDescription)
        }
    }

    // MARK: Write (legacy convenience)

    /// Write audio as a 16-bit WAV file. Kept for backward compatibility.
    public static func writeWAV(
        _ audio: DemucsAudio,
        to url: URL,
        bitsPerSample: UInt16 = 16
    ) throws {
        let bitDepth: WAVBitDepth
        switch bitsPerSample {
        case 16: bitDepth = .int16
        case 24: bitDepth = .int24
        case 32: bitDepth = .int32
        default:
            throw DemucsError.audioIO("Unsupported WAV bit depth: \(bitsPerSample). Use 16, 24, or 32.")
        }
        try writeAudio(audio, to: url, format: .wav(bitDepth: bitDepth))
    }

    // MARK: Write (general)

    /// Write audio to `url` in the given format.
    ///
    /// The container is chosen by the file extension of `url`:
    /// - `.wav`  -> WAV (Linear PCM only)
    /// - `.m4a`  -> MPEG-4 Audio (AAC or ALAC)
    /// - `.flac` -> FLAC
    /// - `.caf`  -> Core Audio Format (supports all codecs)
    ///
    /// If the format and container are incompatible (e.g., AAC in a .wav file)
    /// the call will throw.
    public static func writeAudio(
        _ audio: DemucsAudio,
        to url: URL,
        format: AudioOutputFormat = .default
    ) throws {
        do {
            switch format {
            case .wav(let bitDepth):
                try writeLinearPCM(audio, to: url, bitDepth: bitDepth)
            case .alac:
                try writeCompressed(audio, to: url, formatID: kAudioFormatAppleLossless)
            case .flac:
                try writeCompressed(audio, to: url, formatID: kAudioFormatFLAC)
            case .aac(let bitRate):
                try writeCompressed(audio, to: url, formatID: kAudioFormatMPEG4AAC, bitRate: bitRate)
            }
        } catch let err as DemucsError {
            throw err
        } catch {
            throw DemucsError.audioIO(error.localizedDescription)
        }
    }

    // MARK: - Internal: Linear PCM via AVAudioFile

    private static func writeLinearPCM(
        _ audio: DemucsAudio,
        to url: URL,
        bitDepth: WAVBitDepth
    ) throws {
        let channels = AVAudioChannelCount(audio.channels)
        let sampleRate = Double(audio.sampleRate)
        let frames = audio.frameCount

        // Build the on-disk (file) format.
        let fileFormat: AVAudioFormat
        switch bitDepth {
        case .int24:
            // For 24-bit WAV we must build the AudioStreamBasicDescription
            // manually because AVAudioFormat's convenience initializers don't
            // expose a 24-bit integer mode directly.
            var asbd = AudioStreamBasicDescription(
                mSampleRate: sampleRate,
                mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
                mBytesPerPacket: UInt32(channels) * 3,
                mFramesPerPacket: 1,
                mBytesPerFrame: UInt32(channels) * 3,
                mChannelsPerFrame: UInt32(channels),
                mBitsPerChannel: 24,
                mReserved: 0
            )
            guard let fmt = AVAudioFormat(streamDescription: &asbd) else {
                throw DemucsError.audioIO("Failed to create 24-bit PCM format description")
            }
            fileFormat = fmt

        case .int16:
            guard let fmt = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: true
            ) else {
                throw DemucsError.audioIO("Failed to create 16-bit PCM format description")
            }
            fileFormat = fmt

        case .int32:
            guard let fmt = AVAudioFormat(
                commonFormat: .pcmFormatInt32,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: true
            ) else {
                throw DemucsError.audioIO("Failed to create 32-bit PCM format description")
            }
            fileFormat = fmt

        case .float32:
            guard let fmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: true
            ) else {
                throw DemucsError.audioIO("Failed to create float32 PCM format description")
            }
            fileFormat = fmt
        }

        // Processing (source) format: non-interleaved Float32 -- matches our
        // in-memory layout.
        guard let processingFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: false
        ) else {
            throw DemucsError.audioIO("Failed to create processing format")
        }

        let file = try AVAudioFile(
            forWriting: url,
            settings: fileFormat.settings,
            commonFormat: processingFormat.commonFormat,
            interleaved: false
        )

        // Fill a buffer from our channel-major samples.
        let buffer = try fillBuffer(from: audio, format: processingFormat, frames: frames)

        try file.write(from: buffer)
    }

    // MARK: - Internal: Compressed via AVAudioFile + AVAudioConverter

    private static func writeCompressed(
        _ audio: DemucsAudio,
        to url: URL,
        formatID: AudioFormatID,
        bitRate: Int? = nil
    ) throws {
        let channels = AVAudioChannelCount(audio.channels)
        let sampleRate = Double(audio.sampleRate)
        let frames = audio.frameCount

        // Source format: non-interleaved Float32.
        guard let srcFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: false
        ) else {
            throw DemucsError.audioIO("Failed to create source format for compressed output")
        }

        // Build the destination (compressed) AudioStreamBasicDescription.
        let dstFormat: AVAudioFormat
        switch formatID {
        case kAudioFormatAppleLossless:
            var asbd = AudioStreamBasicDescription(
                mSampleRate: sampleRate,
                mFormatID: kAudioFormatAppleLossless,
                mFormatFlags: 0,
                mBytesPerPacket: 0,
                mFramesPerPacket: 4096,
                mBytesPerFrame: 0,
                mChannelsPerFrame: UInt32(channels),
                mBitsPerChannel: 16,
                mReserved: 0
            )
            guard let fmt = AVAudioFormat(streamDescription: &asbd) else {
                throw DemucsError.audioIO("Failed to create ALAC output format")
            }
            dstFormat = fmt

        case kAudioFormatFLAC:
            var asbd = AudioStreamBasicDescription(
                mSampleRate: sampleRate,
                mFormatID: kAudioFormatFLAC,
                mFormatFlags: 0,
                mBytesPerPacket: 0,
                mFramesPerPacket: 0,
                mBytesPerFrame: 0,
                mChannelsPerFrame: UInt32(channels),
                mBitsPerChannel: 16,
                mReserved: 0
            )
            guard let fmt = AVAudioFormat(streamDescription: &asbd) else {
                throw DemucsError.audioIO("Failed to create FLAC output format")
            }
            dstFormat = fmt

        default:
            // AAC and others.
            var asbd = AudioStreamBasicDescription(
                mSampleRate: sampleRate,
                mFormatID: formatID,
                mFormatFlags: 0,
                mBytesPerPacket: 0,
                mFramesPerPacket: 1024,
                mBytesPerFrame: 0,
                mChannelsPerFrame: UInt32(channels),
                mBitsPerChannel: 0,
                mReserved: 0
            )
            guard let fmt = AVAudioFormat(streamDescription: &asbd) else {
                throw DemucsError.audioIO("Failed to create compressed output format (\(fourCCString(formatID)))")
            }
            dstFormat = fmt
        }

        // Build the settings dictionary for the output file.
        // AVAudioFile(forWriting:settings:commonFormat:interleaved:) accepts
        // compressed settings and PCM commonFormat -- it performs the encoding
        // internally when we write PCM buffers.
        var settings = dstFormat.settings
        if let bitRate = bitRate {
            settings[AVEncoderBitRateKey] = bitRate
        }

        // Fill source PCM buffer.
        let srcBuffer = try fillBuffer(from: audio, format: srcFormat, frames: frames)

        // Open the output file. AVAudioFile handles encoding from PCM to the
        // compressed format specified in `settings`.
        let outputFile = try AVAudioFile(
            forWriting: url,
            settings: settings,
            commonFormat: srcFormat.commonFormat,
            interleaved: false
        )

        try outputFile.write(from: srcBuffer)
    }

    // MARK: - Helpers

    /// Fill an AVAudioPCMBuffer from a DemucsAudio's channel-major sample array.
    private static func fillBuffer(
        from audio: DemucsAudio,
        format: AVAudioFormat,
        frames: Int
    ) throws -> AVAudioPCMBuffer {
        let channels = audio.channels
        let frameCount = AVAudioFrameCount(frames)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw DemucsError.audioIO("Failed to allocate PCM buffer (\(frames) frames, \(channels) channels)")
        }
        buffer.frameLength = frameCount

        guard let channelData = buffer.floatChannelData else {
            throw DemucsError.audioIO("Buffer has no float channel data")
        }

        let samples = audio.channelMajorSamples
        for c in 0..<channels {
            let dst = channelData[c]
            let srcBase = c * frames
            for t in 0..<frames {
                dst[t] = AudioDSP.clampAudio(samples[srcBase + t])
            }
        }

        return buffer
    }

    /// Convert a FourCC AudioFormatID to a readable string for error messages.
    private static func fourCCString(_ code: AudioFormatID) -> String {
        let bytes = [
            UInt8((code >> 24) & 0xFF),
            UInt8((code >> 16) & 0xFF),
            UInt8((code >> 8) & 0xFF),
            UInt8(code & 0xFF),
        ]
        if let s = String(bytes: bytes, encoding: .ascii) {
            return s
        }
        return String(format: "0x%08X", code)
    }

    /// Infer the best `AudioOutputFormat` from a file extension and optional parameters.
    ///
    /// This is a convenience for CLI or other callers that want to map an extension
    /// to a sensible default format.
    public static func inferFormat(
        forExtension ext: String,
        bitDepth: WAVBitDepth = .int16,
        useALAC: Bool = false,
        aacBitRate: Int = 256_000
    ) -> AudioOutputFormat {
        switch ext.lowercased() {
        case "wav", "wave":
            return .wav(bitDepth: bitDepth)
        case "m4a", "mp4":
            return useALAC ? .alac : .aac(bitRate: aacBitRate)
        case "flac":
            return .flac
        case "caf":
            // CAF can hold anything; default to ALAC if lossless requested, else WAV PCM.
            return useALAC ? .alac : .wav(bitDepth: bitDepth)
        default:
            return .wav(bitDepth: bitDepth)
        }
    }
}
