import Foundation

#if canImport(MLX)
import MLX
#endif

public struct DemucsAudio: Sendable {
    public let channelMajorSamples: [Float]
    public let channels: Int
    public let sampleRate: Int

    public init(channelMajor: [Float], channels: Int, sampleRate: Int) throws {
        guard channels > 0 else {
            throw DemucsError.invalidAudioData("channels must be > 0")
        }
        guard sampleRate > 0 else {
            throw DemucsError.invalidAudioData("sampleRate must be > 0")
        }
        guard channelMajor.count % channels == 0 else {
            throw DemucsError.invalidAudioData("channelMajor.count must be divisible by channels")
        }

        self.channelMajorSamples = channelMajor
        self.channels = channels
        self.sampleRate = sampleRate
    }

    public var frameCount: Int {
        channelMajorSamples.count / channels
    }
}

#if canImport(MLX)
extension DemucsAudio {
    public init(tensor: MLXArray, sampleRate: Int) throws {
        guard tensor.ndim == 2 else {
            throw DemucsError.invalidAudioShape(tensor.shape)
        }
        let channels = tensor.shape[0]
        let samples = tensor.asArray(Float.self)
        try self.init(channelMajor: samples, channels: channels, sampleRate: sampleRate)
    }

    public func asMLXArray() -> MLXArray {
        MLXArray(channelMajorSamples, [channels, frameCount])
    }
}
#endif

public struct DemucsSeparationResult: Sendable {
    public let input: DemucsAudio
    public let stems: [String: DemucsAudio]
}

public struct DemucsSeparationParameters: Sendable {
    public var shifts: Int
    public var overlap: Float
    public var split: Bool
    public var segmentSeconds: Double?
    public var batchSize: Int
    public var seed: Int?

    public init(
        shifts: Int = 1,
        overlap: Float = 0.25,
        split: Bool = true,
        segmentSeconds: Double? = nil,
        batchSize: Int = 8,
        seed: Int? = nil
    ) {
        self.shifts = shifts
        self.overlap = overlap
        self.split = split
        self.segmentSeconds = segmentSeconds
        self.batchSize = batchSize
        self.seed = seed
    }

    func validated() throws -> DemucsSeparationParameters {
        guard shifts >= 0 else {
            throw DemucsError.invalidParameter("shifts must be >= 0")
        }
        guard overlap >= 0 && overlap < 1 else {
            throw DemucsError.invalidParameter("overlap must be in [0, 1)")
        }
        if let segmentSeconds {
            guard segmentSeconds > 0 else {
                throw DemucsError.invalidParameter("segmentSeconds must be > 0 when provided")
            }
        }
        guard batchSize > 0 else {
            throw DemucsError.invalidParameter("batchSize must be > 0")
        }
        return self
    }
}
