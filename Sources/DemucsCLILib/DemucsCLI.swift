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

        let modelDirectoryURL = modelDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let separator = try DemucsSeparator(modelName: name, parameters: params, modelDirectory: modelDirectoryURL)
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
            print("Separating: \(inputURL.path)")

            let result = try separator.separate(fileAt: inputURL)
            let trackDir = outputRoot.appendingPathComponent(inputURL.deletingPathExtension().lastPathComponent, isDirectory: true)
            try FileManager.default.createDirectory(at: trackDir, withIntermediateDirectories: true)

            if let stem = twoStems {
                // Two-stem mode: write the selected stem and its complement
                guard let selectedAudio = result.stems[stem] else { continue }

                // Write the selected stem
                let stemURL = trackDir.appendingPathComponent("\(stem).\(fileExtension)", isDirectory: false)
                try AudioIO.writeAudio(selectedAudio, to: stemURL, format: outputFormat)
                print("  wrote \(stemURL.path)")

                // Compute the complement: original mix minus the selected stem
                let mixSamples = result.input.channelMajorSamples
                let stemSamples = selectedAudio.channelMajorSamples
                var complementSamples = [Float](repeating: 0, count: mixSamples.count)
                for i in 0..<mixSamples.count {
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
            } else {
                // Normal mode: write all stems
                for source in separator.sources {
                    guard let stemAudio = result.stems[source] else { continue }
                    let stemURL = trackDir.appendingPathComponent("\(source).\(fileExtension)", isDirectory: false)
                    try AudioIO.writeAudio(stemAudio, to: stemURL, format: outputFormat)
                    print("  wrote \(stemURL.path)")
                }
            }
        }
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
        } else if float32 {
            bitDepth = .float32
        } else {
            bitDepth = .int16
        }
        return (.wav(bitDepth: bitDepth), "wav")
    }
}
