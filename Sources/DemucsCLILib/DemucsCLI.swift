import ArgumentParser
import DemucsMLX
import Foundation

public struct DemucsCLI: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "demucs-mlx-swift",
        abstract: "Demucs-style audio stem separation with Swift + MLX",
        discussion: "Separates input audio files into drums, bass, other, and vocals stems."
    )

    @Argument(help: "Input audio files")
    public var tracks: [String] = []

    @Option(name: [.short, .long], help: "Model name")
    public var name: String = "htdemucs"

    @Option(name: [.short, .long], help: "Output directory")
    public var out: String = "separated"

    @Option(name: .customLong("model-dir"), help: "Directory containing model files (.safetensors + _config.json)")
    public var modelDir: String?

    @Option(name: .long, help: "Segment length in seconds")
    public var segment: Double?

    @Option(name: .long, help: "Overlap ratio [0, 1)")
    public var overlap: Float = 0.25

    @Option(name: .long, help: "Number of shift augmentations")
    public var shifts: Int = 1

    @Option(name: .long, help: "Optional random seed for deterministic shifts")
    public var seed: Int?

    @Option(name: [.short, .long], help: "Chunk batch size")
    public var batchSize: Int = 8

    @Flag(name: .customLong("no-split"), help: "Disable chunked overlap-add inference")
    public var noSplit: Bool = false

    @Option(name: .customLong("two-stems"), help: "Only output the given stem and its complement (e.g. vocals produces vocals.wav and no_vocals.wav)")
    public var twoStems: String?

    @Flag(name: .customLong("list-models"), help: "List available models")
    public var listModels: Bool = false

    @Flag(name: .customLong("async"), help: "Use the closure-based async API with progress reporting")
    public var useAsync: Bool = false

    @Option(name: .customLong("cancel-after"), help: "Cancel separation after N seconds (for testing cancellation)")
    public var cancelAfter: Double?

    // MARK: Output format options

    @Flag(name: .customLong("mp3"), help: "Output as AAC in .m4a (Apple's lossy equivalent of MP3)")
    public var mp3: Bool = false

    @Flag(name: .customLong("flac"), help: "Output as FLAC lossless")
    public var flac: Bool = false

    @Flag(name: .customLong("alac"), help: "Output as Apple Lossless (ALAC) in .m4a")
    public var alac: Bool = false

    @Flag(name: .customLong("int24"), help: "Output 24-bit integer WAV")
    public var int24: Bool = false

    @Flag(name: .customLong("float32"), help: "Output 32-bit float WAV")
    public var float32: Bool = false

    public init() {}

    public mutating func run() throws {
        if listModels {
            for model in listAvailableDemucsModels() {
                print(model)
            }
            return
        }

        guard !tracks.isEmpty else {
            throw ValidationError("Please provide at least one input track or use --list-models")
        }

        // Determine output format and file extension from flags.
        let (outputFormat, fileExtension) = try resolveOutputFormat()

        let params = DemucsSeparationParameters(
            shifts: shifts,
            overlap: overlap,
            split: !noSplit,
            segmentSeconds: segment,
            batchSize: batchSize,
            seed: seed
        )

        print("Loading model '\(name)'...")
        let modelDirectoryURL = modelDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let separator = try DemucsSeparator(modelName: name, parameters: params, modelDirectory: modelDirectoryURL)
        print("Model loaded. Sources: \(separator.sources.joined(separator: ", "))")

        let outputRoot = URL(fileURLWithPath: out, isDirectory: true)
        try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)

        // Validate --two-stems value against the model's source names
        if let stem = twoStems {
            guard separator.sources.contains(stem) else {
                throw ValidationError(
                    "Stem \"\(stem)\" is not in the selected model. "
                    + "Must be one of: \(separator.sources.joined(separator: ", "))"
                )
            }
        }

        for track in tracks {
            let inputURL = URL(fileURLWithPath: track)
            print("\nSeparating: \(inputURL.path)")

            let start = CFAbsoluteTimeGetCurrent()

            let result: DemucsSeparationResult
            if useAsync || cancelAfter != nil {
                result = try Self.separateAsync(separator: separator, inputURL: inputURL, cancelAfterSeconds: cancelAfter)
            }
            else {
                result = try separator.separate(fileAt: inputURL)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            print(String(format: "Separation complete (%.2fs)", elapsed))

            let trackDir = outputRoot.appendingPathComponent(inputURL.deletingPathExtension().lastPathComponent, isDirectory: true)
            try FileManager.default.createDirectory(at: trackDir, withIntermediateDirectories: true)

            if let stem = twoStems {
                // Two-stem mode: write the selected stem and its complement
                guard let selectedAudio = result.stems[stem]
                else { continue }

                // Write the selected stem
                let stemURL = trackDir.appendingPathComponent("\(stem).\(fileExtension)", isDirectory: false)
                try AudioIO.writeAudio(selectedAudio, to: stemURL, format: outputFormat)
                print("  wrote \(stemURL.path)")

                // Compute the complement: original mix minus the selected stem
                let mixSamples = result.input.channelMajorSamples
                let stemSamples = selectedAudio.channelMajorSamples
                var complementSamples = [Float](repeating: 0, count: mixSamples.count)
                for i in 0 ..< mixSamples.count {
                    complementSamples[i] = mixSamples[i] - stemSamples[i]
                }

                let complementAudio = try DemucsAudio(
                    channelMajor: complementSamples,
                    channels: selectedAudio.channels,
                    sampleRate: selectedAudio.sampleRate
                )

                let complementURL = trackDir.appendingPathComponent("no_\(stem).\(fileExtension)", isDirectory: false)
                try AudioIO.writeAudio(complementAudio, to: complementURL, format: outputFormat)
                print("  wrote \(complementURL.path)")
            }
            else {
                // Normal mode: write all stems
                for source in separator.sources {
                    guard let stemAudio = result.stems[source]
                    else { continue }

                    let stemURL = trackDir.appendingPathComponent("\(source).\(fileExtension)", isDirectory: false)
                    try AudioIO.writeAudio(stemAudio, to: stemURL, format: outputFormat)
                    print("  wrote \(stemURL.path)")
                }
            }
        }
    }

    // MARK: - Async Separation

    /// Thread-safe box for mutable state shared across closures.
    private final class AsyncState: @unchecked Sendable {
        private let lock = NSLock()
        var result: Result<DemucsSeparationResult, Error>?
        var lastProgressLineLength: Int = 0

        func setResult(_ value: Result<DemucsSeparationResult, Error>) {
            self.lock.lock()
            self.result = value
            self.lock.unlock()
        }

        func getResult() -> Result<DemucsSeparationResult, Error>? {
            self.lock.lock()
            defer { self.lock.unlock() }
            return self.result
        }
    }

    /// Use the closure-based async API with progress reporting.
    /// Blocks the calling thread until the separation completes.
    private static func separateAsync(separator: DemucsSeparator, inputURL: URL, cancelAfterSeconds: Double? = nil) throws -> DemucsSeparationResult {
        let semaphore = DispatchSemaphore(value: 0)
        let state = AsyncState()
        let cancelToken = DemucsCancelToken()

        // Schedule cancellation after a delay if requested
        if let delay = cancelAfterSeconds {
            print("  Will cancel after \(delay)s...")
            DispatchQueue.global().asyncAfter(deadline: .now() + delay, execute: {
                print("\n  Cancelling separation...")
                cancelToken.cancel()
            })
        }

        separator.separate(
            fileAt: inputURL,
            cancelToken: cancelToken,
            progress: { progress in
                // Called on main queue - print progress
                let percent = Int(progress.fraction * 100)
                let bar = progressBar(fraction: progress.fraction, width: 30)
                let etaStr: String
                if let eta = progress.estimatedTimeRemaining, eta > 0 && progress.fraction < 1.0 {
                    let mins = Int(eta) / 60
                    let secs = Int(eta) % 60
                    etaStr = mins > 0 ? " ETA \(mins)m\(String(format: "%02d", secs))s" : " ETA \(secs)s"
                } else {
                    etaStr = ""
                }
                let line = "\r  [\(bar)] \(percent)% - \(progress.stage)\(etaStr)"
                let padded = line.padding(toLength: max(line.count, state.lastProgressLineLength), withPad: " ", startingAt: 0)
                print(padded, terminator: "")
                fflush(stdout)
                state.lastProgressLineLength = line.count

                if ProcessInfo.processInfo.environment["DEMUCS_BENCH"] != nil {
                    fputs("PROGRESS_TS \(CFAbsoluteTimeGetCurrent()) \(progress.fraction) \(progress.stage)\n", stderr)
                }
            },
            completion: { result in
                // Called on main queue
                state.setResult(result)
                semaphore.signal()
            }
        )

        // Run the main run loop so that main-queue callbacks can fire
        while semaphore.wait(timeout: .now() + 0.05) == .timedOut {
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.05))
        }

        // Clear the progress line
        print("")

        guard let result = state.getResult()
        else {
            throw DemucsError.cancelled
        }

        return try result.get()
    }

    /// Render a simple ASCII progress bar.
    private static func progressBar(fraction: Float, width: Int) -> String {
        let filled = Int(fraction * Float(width))
        let empty = width - filled
        return String(repeating: "=", count: filled) + String(repeating: " ", count: empty)
    }

    // MARK: - Format resolution

    private func resolveOutputFormat() throws -> (AudioOutputFormat, String) {
        // Count how many exclusive format flags were set.
        let formatFlags = [mp3, flac, alac].filter { $0 }
        if formatFlags.count > 1 {
            throw ValidationError("Only one of --mp3, --flac, --alac may be specified")
        }

        // Bit depth flags are only relevant for WAV output.
        let bitDepthFlags = [int24, float32].filter { $0 }
        if bitDepthFlags.count > 1 {
            throw ValidationError("Only one of --int24, --float32 may be specified")
        }

        if mp3 {
            if !bitDepthFlags.isEmpty {
                throw ValidationError("Bit depth flags (--int24, --float32) are not applicable to AAC output")
            }
            return (.aac(bitRate: 256_000), "m4a")
        }

        if flac {
            if !bitDepthFlags.isEmpty {
                throw ValidationError("Bit depth flags (--int24, --float32) are not applicable to FLAC output")
            }
            return (.flac, "flac")
        }

        if alac {
            if !bitDepthFlags.isEmpty {
                throw ValidationError("Bit depth flags (--int24, --float32) are not applicable to ALAC output")
            }
            return (.alac, "m4a")
        }

        // WAV output (default).
        let bitDepth: WAVBitDepth
        if int24 {
            bitDepth = .int24
        }
        else if float32 {
            bitDepth = .float32
        }
        else {
            bitDepth = .int16
        }
        return (.wav(bitDepth: bitDepth), "wav")
    }
}
