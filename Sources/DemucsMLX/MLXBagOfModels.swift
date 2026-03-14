import Foundation
import MLX

/// Ensemble model that runs multiple sub-models and averages their outputs
/// with per-source weights.
///
/// Matches the Python MLX BagOfModelsMLX behavior.
final class BagOfModels: StemSeparationModel {
    let descriptor: DemucsModelDescriptor
    private let models: [StemSeparationModel]
    /// Per-model, per-source weights: weights[modelIdx][sourceIdx]
    private let weights: [[Float]]
    /// Sum of weights per source for normalization
    private let totals: [Float]

    init(
        descriptor: DemucsModelDescriptor,
        models: [StemSeparationModel],
        weights: [[Float]]?
    ) {
        self.descriptor = descriptor
        self.models = models

        let sourceCount = descriptor.sourceNames.count
        let defaultWeight = [Float](repeating: 1.0, count: sourceCount)

        if let weights, !weights.isEmpty {
            self.weights = weights.map { w in
                w.count == sourceCount ? w : defaultWeight
            }
        } else {
            self.weights = [[Float]](repeating: defaultWeight, count: models.count)
        }

        // Compute per-source totals for normalization
        var sums = [Float](repeating: 0.0, count: sourceCount)
        for modelWeights in self.weights {
            for s in 0..<sourceCount {
                sums[s] += modelWeights[s]
            }
        }
        self.totals = sums
    }

    func predict(
        batchData: [Float],
        batchSize: Int,
        channels: Int,
        frames: Int,
        monitor: SeparationMonitor? = nil
    ) throws -> [Float] {
        let sourceCount = descriptor.sourceNames.count
        let input = MLXArray(batchData).reshaped([batchSize, channels, frames])

        // Accumulate on GPU to avoid CPU↔GPU transfers between sub-models
        var accumulated: MLXArray? = nil

        for (modelIdx, model) in models.enumerated() {
            try monitor?.checkCancellation()

            let w = weights[modelIdx]

            // Create weight tensor: [1, S, 1, 1] for broadcasting
            let weightArray = MLXArray(w).reshaped([1, sourceCount, 1, 1])

            let output: MLXArray
            if let gpuModel = model as? GPUPredictable {
                output = try gpuModel.predictGPU(input: input, monitor: monitor)
            }
            else {
                // Fallback: go through [Float]
                let floatOutput = try model.predict(
                    batchData: batchData,
                    batchSize: batchSize,
                    channels: channels,
                    frames: frames,
                    monitor: monitor
                )
                output = MLXArray(floatOutput).reshaped([batchSize, sourceCount, channels, frames])
            }

            // output shape: [B, S, C, T]
            let weighted = output * weightArray

            if let acc = accumulated {
                accumulated = acc + weighted
            }
            else {
                accumulated = weighted
            }
        }

        guard let result = accumulated
        else {
            return [Float](repeating: 0, count: batchSize * sourceCount * channels * frames)
        }

        // Normalize by total weight per source: [1, S, 1, 1]
        let totalArray = MLXArray(totals).reshaped([1, sourceCount, 1, 1])
        let normalized = result / MLX.maximum(totalArray, MLXArray(Float(1e-8)))

        MLX.eval(normalized)
        return normalized.asArray(Float.self)
    }
}
