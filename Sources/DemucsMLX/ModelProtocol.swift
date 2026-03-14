import Foundation
import MLX
import MLXNN

protocol StemSeparationModel {
    var descriptor: DemucsModelDescriptor { get }

    /// Input shape: [batch, channels, frames] (flattened channel-major per batch item).
    /// Output shape: [batch, sources, channels, frames] (flattened).
    func predict(
        batchData: [Float],
        batchSize: Int,
        channels: Int,
        frames: Int,
        monitor: SeparationMonitor?
    ) throws -> [Float]
}

enum DemucsModelFactory {
    /// Create the appropriate model (or ensemble) for the given descriptor.
    /// Reads the config JSON to determine model_class and routes accordingly.
    static func makeModel(
        for descriptor: DemucsModelDescriptor,
        modelDirectory: URL? = nil
    ) throws -> StemSeparationModel {
        let directory = try ModelLoader.resolveModelDirectory(
            modelName: descriptor.name, preferred: modelDirectory
        )
        let config = try ModelLoader.loadConfig(from: directory, modelName: descriptor.name)

        let modelClass = config["model_class"] as? String ?? ""

        if modelClass == "BagOfModelsMLX" || modelClass == "BagOfModels" {
            return try makeBagOfModels(
                descriptor: descriptor,
                config: config,
                directory: directory
            )
        }

        return try makeSingleModel(
            descriptor: descriptor,
            config: config,
            directory: directory,
            weightPrefix: nil
        )
    }

    // MARK: - Bag of Models

    private static func makeBagOfModels(
        descriptor: DemucsModelDescriptor,
        config: [String: Any],
        directory: URL
    ) throws -> StemSeparationModel {
        let numModels = ModelLoader.int(config, "num_models", 1)
        let bagWeights = config["weights"] as? [Any]
        let defaultSubModelClass = config["sub_model_class"] as? String ?? "HTDemucsMLX"

        // Per-model class array for heterogeneous bags (e.g., mdx = 2×Demucs + 2×HDemucs)
        let subModelClasses = config["sub_model_classes"] as? [String]
        // Per-model config with kwargs
        let modelConfigs = config["model_configs"] as? [[String: Any]]

        let allWeights = try ModelLoader.loadWeights(from: directory, modelName: descriptor.name)

        var subModels: [StemSeparationModel] = []
        var weightVectors: [[Float]] = []

        for i in 0..<numModels {
            let prefix = "model_\(i)."
            var subWeights: [String: MLXArray] = [:]
            subWeights.reserveCapacity(allWeights.count / numModels)
            for (key, value) in allWeights {
                if key.hasPrefix(prefix) {
                    subWeights[String(key.dropFirst(prefix.count))] = value
                }
            }

            // Determine model class for this sub-model
            let modelClass: String
            if let classes = subModelClasses, i < classes.count {
                modelClass = classes[i]
            } else {
                modelClass = defaultSubModelClass
            }

            // Build per-model config: use model_configs[i].kwargs if available, else fall back to bag-level kwargs
            let subConfig: [String: Any]
            if let configs = modelConfigs, i < configs.count {
                subConfig = configs[i]["kwargs"] as? [String: Any] ?? config
            } else {
                subConfig = config["kwargs"] as? [String: Any] ?? config
            }

            let model = try buildModelGraph(
                descriptor: descriptor,
                modelClass: modelClass,
                config: subConfig,
                weights: subWeights
            )
            subModels.append(model)

            if let bagWeights, i < bagWeights.count {
                let sourceCount = descriptor.sourceNames.count
                if let wArray = bagWeights[i] as? [Any] {
                    weightVectors.append(wArray.map { ($0 as? NSNumber)?.floatValue ?? 1.0 })
                } else if let w = (bagWeights[i] as? NSNumber)?.floatValue {
                    weightVectors.append([Float](repeating: w, count: sourceCount))
                } else {
                    weightVectors.append([Float](repeating: 1.0, count: sourceCount))
                }
            }
        }

        return BagOfModels(
            descriptor: descriptor,
            models: subModels,
            weights: weightVectors.isEmpty ? nil : weightVectors
        )
    }

    // MARK: - Single Model Builder

    private static func makeSingleModel(
        descriptor: DemucsModelDescriptor,
        config: [String: Any],
        directory: URL,
        weightPrefix: String?
    ) throws -> StemSeparationModel {
        let allWeights = try ModelLoader.loadWeights(from: directory, modelName: descriptor.name)
        var weights = allWeights

        // Strip prefix if present (e.g., "model_0." for single-model bags)
        if let prefix = weightPrefix {
            var stripped: [String: MLXArray] = [:]
            for (key, value) in allWeights {
                if key.hasPrefix(prefix) {
                    stripped[String(key.dropFirst(prefix.count))] = value
                }
            }
            weights = stripped
        } else {
            // Auto-detect "model_0." prefix for single-model bags
            let hasModelPrefix = allWeights.keys.contains { $0.hasPrefix("model_0.") }
            if hasModelPrefix {
                var stripped: [String: MLXArray] = [:]
                for (key, value) in allWeights {
                    if key.hasPrefix("model_0.") {
                        stripped[String(key.dropFirst("model_0.".count))] = value
                    }
                }
                weights = stripped
            }
        }

        let modelClass = config["model_class"] as? String
            ?? config["sub_model_class"] as? String
            ?? detectModelClass(from: config)

        return try buildModelGraph(
            descriptor: descriptor,
            modelClass: modelClass,
            config: config,
            weights: weights
        )
    }

    /// Build and load weights into a single model graph.
    private static func buildModelGraph(
        descriptor: DemucsModelDescriptor,
        modelClass: String,
        config: [String: Any],
        weights: [String: MLXArray]
    ) throws -> StemSeparationModel {
        switch modelClass {
        case "HTDemucsMLX", "HTDemucs":
            let cfg = try HTDemucsRuntimeConfig.fromJSON(config)
            let graph = HTDemucsGraph(config: cfg)
            try graph.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            MLX.eval(graph.parameters())
            let updatedDescriptor = DemucsModelDescriptor(
                name: descriptor.name,
                sourceNames: cfg.sources,
                sampleRate: cfg.samplerate,
                audioChannels: cfg.audioChannels,
                defaultSegmentSeconds: Double(cfg.segment)
            )
            return HTDemucsModelWrapper(descriptor: updatedDescriptor, graph: graph)

        case "HDemucsMLX", "HDemucs":
            let cfg = try HDemucsRuntimeConfig.fromJSON(config)
            let graph = HDemucsGraph(config: cfg)
            try graph.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            MLX.eval(graph.parameters())
            let updatedDescriptor = DemucsModelDescriptor(
                name: descriptor.name,
                sourceNames: cfg.sources,
                sampleRate: cfg.samplerate,
                audioChannels: cfg.audioChannels,
                defaultSegmentSeconds: Double(cfg.segment)
            )
            return HDemucsModelWrapper(descriptor: updatedDescriptor, graph: graph)

        case "DemucsMLX", "Demucs":
            let cfg = try DemucsRuntimeConfig.fromJSON(config)
            let graph = DemucsGraph(config: cfg)
            try graph.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            MLX.eval(graph.parameters())
            let updatedDescriptor = DemucsModelDescriptor(
                name: descriptor.name,
                sourceNames: cfg.sources,
                sampleRate: cfg.samplerate,
                audioChannels: cfg.audioChannels,
                defaultSegmentSeconds: Double(cfg.segment)
            )
            return DemucsModelWrapper(descriptor: updatedDescriptor, graph: graph)

        default:
            throw DemucsError.unsupportedModelBackend(
                "Unknown model class '\(modelClass)'. Supported: HTDemucsMLX, HDemucsMLX, DemucsMLX"
            )
        }
    }

    /// Detect model class from config kwargs when not explicitly specified.
    private static func detectModelClass(from config: [String: Any]) -> String {
        let kwargs = config["kwargs"] as? [String: Any] ?? config
        if kwargs["nfft"] != nil || kwargs["nFFT"] != nil {
            // Has spectral params → hybrid model
            if kwargs["t_layers"] != nil {
                return "HTDemucsMLX"
            }
            return "HDemucsMLX"
        }
        return "DemucsMLX"
    }
}

// MARK: - GPU-native predict protocol

/// Optional GPU-native predict for models that can stay on GPU.
/// Used by BagOfModels to avoid CPU↔GPU roundtrips between sub-models.
protocol GPUPredictable {
    func predictGPU(input: MLXArray, monitor: SeparationMonitor?) throws -> MLXArray
}

// MARK: - Lightweight Wrappers

/// Wraps an HTDemucsGraph as a StemSeparationModel.
final class HTDemucsModelWrapper: StemSeparationModel, GPUPredictable {
    let descriptor: DemucsModelDescriptor
    private let graph: HTDemucsGraph

    init(descriptor: DemucsModelDescriptor, graph: HTDemucsGraph) {
        self.descriptor = descriptor
        self.graph = graph
    }

    func predictGPU(input: MLXArray, monitor: SeparationMonitor? = nil) throws -> MLXArray {
        try graph.forward(input, monitor: monitor)
    }

    func predict(batchData: [Float], batchSize: Int, channels: Int, frames: Int, monitor: SeparationMonitor? = nil) throws -> [Float] {
        let input = MLXArray(batchData).reshaped([batchSize, channels, frames])
        let output = try predictGPU(input: input, monitor: monitor)
        MLX.eval(output)
        return output.asArray(Float.self)
    }
}

/// Wraps an HDemucsGraph as a StemSeparationModel.
final class HDemucsModelWrapper: StemSeparationModel, GPUPredictable {
    let descriptor: DemucsModelDescriptor
    private let graph: HDemucsGraph

    init(descriptor: DemucsModelDescriptor, graph: HDemucsGraph) {
        self.descriptor = descriptor
        self.graph = graph
    }

    func predictGPU(input: MLXArray, monitor: SeparationMonitor? = nil) throws -> MLXArray {
        try graph.forward(input, monitor: monitor)
    }

    func predict(batchData: [Float], batchSize: Int, channels: Int, frames: Int, monitor: SeparationMonitor? = nil) throws -> [Float] {
        let input = MLXArray(batchData).reshaped([batchSize, channels, frames])
        let output = try predictGPU(input: input, monitor: monitor)
        MLX.eval(output)
        return output.asArray(Float.self)
    }
}

/// Wraps a DemucsGraph as a StemSeparationModel.
final class DemucsModelWrapper: StemSeparationModel, GPUPredictable {
    let descriptor: DemucsModelDescriptor
    private let graph: DemucsGraph

    init(descriptor: DemucsModelDescriptor, graph: DemucsGraph) {
        self.descriptor = descriptor
        self.graph = graph
    }

    func predictGPU(input: MLXArray, monitor: SeparationMonitor? = nil) throws -> MLXArray {
        try graph.forward(input, monitor: monitor)
    }

    func predict(batchData: [Float], batchSize: Int, channels: Int, frames: Int, monitor: SeparationMonitor? = nil) throws -> [Float] {
        let input = MLXArray(batchData).reshaped([batchSize, channels, frames])
        let output = try predictGPU(input: input, monitor: monitor)
        MLX.eval(output)
        return output.asArray(Float.self)
    }
}
