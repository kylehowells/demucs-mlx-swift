import Foundation

public final class DemucsSeparator: @unchecked Sendable {
    public let modelName: String
    public let descriptor: DemucsModelDescriptor

    public var parameters: DemucsSeparationParameters

    private let model: StemSeparationModel

    public init(
        modelName: String = "htdemucs",
        parameters: DemucsSeparationParameters = DemucsSeparationParameters(),
        modelDirectory: URL? = nil
    ) throws {
        self.modelName = modelName
        self.descriptor = try DemucsModelRegistry.descriptor(for: modelName)
        self.parameters = try parameters.validated()
        self.model = try DemucsModelFactory.makeModel(for: descriptor, modelDirectory: modelDirectory)
    }

    public var sampleRate: Int {
        descriptor.sampleRate
    }

    public var audioChannels: Int {
        descriptor.audioChannels
    }

    public var sources: [String] {
        descriptor.sourceNames
    }

    public func updateParameters(_ parameters: DemucsSeparationParameters) throws {
        self.parameters = try parameters.validated()
    }

    public func separate(fileAt url: URL) throws -> DemucsSeparationResult {
        let audio = try AudioIO.loadAudio(from: url)
        return try separate(audio: audio)
    }

    public func separate(audio: DemucsAudio) throws -> DemucsSeparationResult {
        return try self.separate(audio: audio, monitor: nil)
    }

    // MARK: - Closure-Based Async API

    /// The internal serial queue used for background separation work.
    private static let separationQueue = DispatchQueue(label: "com.demucs.separation", qos: .userInitiated)

    /// Separate an audio file into stems on a background queue.
    ///
    /// - Parameters:
    ///   - url: Path to the audio file to separate.
    ///   - cancelToken: Optional token to request cancellation. Call `cancel()` on it to stop.
    ///   - progress: Optional progress callback. Called on the **main queue**.
    ///   - completion: Called on the **main queue** with the result or error.
    public func separate(
        fileAt url: URL,
        cancelToken: DemucsCancelToken?,
        progress: (@Sendable (_ progress: DemucsSeparationProgress) -> Void)?,
        completion: @escaping @Sendable (_ result: Result<DemucsSeparationResult, Error>) -> Void
    ) {
        let progressCopy = progress
        let completionCopy = completion
        DemucsSeparator.separationQueue.async(execute: { [self] in
            let result: Result<DemucsSeparationResult, Error>

            // Create interpolator for smooth progress during GPU batch gaps
            let interpolator: ProgressInterpolator?
            if let progressCopy {
                interpolator = ProgressInterpolator(callback: progressCopy)
            } else {
                interpolator = nil
            }

            do {
                let audio = try AudioIO.loadAudio(from: url)
                let monitor = SeparationMonitor(
                    cancelToken: cancelToken,
                    progressHandler: { fraction, stage in
                        interpolator?.onProgress(fraction, stage: stage)
                    }
                )
                let separationResult = try self.separate(audio: audio, monitor: monitor)
                result = .success(separationResult)
            }
            catch {
                result = .failure(error)
            }
            interpolator?.stop()
            DispatchQueue.main.async(execute: {
                completionCopy(result)
            })
        })
    }

    /// Separate audio into stems on a background queue.
    ///
    /// - Parameters:
    ///   - audio: The audio data to separate.
    ///   - cancelToken: Optional token to request cancellation.
    ///   - progress: Optional progress callback. Called on the **main queue**.
    ///   - completion: Called on the **main queue** with the result or error.
    public func separate(
        audio: DemucsAudio,
        cancelToken: DemucsCancelToken?,
        progress: (@Sendable (_ progress: DemucsSeparationProgress) -> Void)?,
        completion: @escaping @Sendable (_ result: Result<DemucsSeparationResult, Error>) -> Void
    ) {
        let progressCopy = progress
        let completionCopy = completion
        DemucsSeparator.separationQueue.async(execute: { [self] in
            let result: Result<DemucsSeparationResult, Error>

            let interpolator: ProgressInterpolator?
            if let progressCopy {
                interpolator = ProgressInterpolator(callback: progressCopy)
            } else {
                interpolator = nil
            }

            do {
                let monitor = SeparationMonitor(
                    cancelToken: cancelToken,
                    progressHandler: { fraction, stage in
                        interpolator?.onProgress(fraction, stage: stage)
                    }
                )
                let separationResult = try self.separate(audio: audio, monitor: monitor)
                result = .success(separationResult)
            }
            catch {
                result = .failure(error)
            }
            interpolator?.stop()
            DispatchQueue.main.async(execute: {
                completionCopy(result)
            })
        })
    }

    // MARK: - Internal

    private func separate(audio: DemucsAudio, monitor: SeparationMonitor?) throws -> DemucsSeparationResult {
        let validated = try parameters.validated()

        try monitor?.checkCancellation()

        let input = audio.channelMajorSamples
        let remixed = AudioDSP.remixChannels(
            channelMajor: input,
            inputChannels: audio.channels,
            targetChannels: descriptor.audioChannels,
            frames: audio.frameCount
        )

        let resampled = AudioDSP.resampleChannelMajor(
            remixed,
            channels: descriptor.audioChannels,
            inputSampleRate: audio.sampleRate,
            targetSampleRate: descriptor.sampleRate,
            frames: audio.frameCount
        )

        let normalizedAudio = try DemucsAudio(
            channelMajor: resampled.samples,
            channels: descriptor.audioChannels,
            sampleRate: descriptor.sampleRate
        )

        try monitor?.checkCancellation()
        monitor?.reportProgress(0.0, stage: "Starting separation")

        let engine = SeparationEngine(model: model, parameters: validated, monitor: monitor)
        let stemsFlat = try engine.separate(
            mix: resampled.samples,
            channels: descriptor.audioChannels,
            frames: resampled.frames,
            sampleRate: descriptor.sampleRate
        )

        try monitor?.checkCancellation()
        monitor?.reportProgress(1.0, stage: "Complete")

        var stems: [String: DemucsAudio] = [:]
        for (sourceIndex, sourceName) in descriptor.sourceNames.enumerated() {
            var sourceSamples = [Float](repeating: 0, count: descriptor.audioChannels * resampled.frames)

            for c in 0..<descriptor.audioChannels {
                let sourceBase = (sourceIndex * descriptor.audioChannels + c) * resampled.frames
                let dstBase = c * resampled.frames
                for t in 0..<resampled.frames {
                    sourceSamples[dstBase + t] = stemsFlat[sourceBase + t]
                }
            }

            stems[sourceName] = try DemucsAudio(
                channelMajor: sourceSamples,
                channels: descriptor.audioChannels,
                sampleRate: descriptor.sampleRate
            )
        }

        return DemucsSeparationResult(input: normalizedAudio, stems: stems)
    }
}
