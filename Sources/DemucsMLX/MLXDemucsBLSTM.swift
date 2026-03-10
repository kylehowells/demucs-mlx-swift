import Foundation
import MLX
import MLXNN

/// Bidirectional LSTM matching the Python MLX BLSTM implementation.
/// Input/output shape: (B, C, T) in NCL format.
final class DemucsBLSTM: Module, DemucsUnaryLayer {
    let maxSteps: Int?
    let skip: Bool
    let lstmLayers: Int

    @ModuleInfo(key: "forward_lstms") var forwardLstms: [LSTM]
    @ModuleInfo(key: "backward_lstms") var backwardLstms: [LSTM]
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int, layers: Int = 1, maxSteps: Int? = nil, skip: Bool = false) {
        self.maxSteps = maxSteps
        self.skip = skip
        self.lstmLayers = layers

        // Each layer takes dim input (first layer) or 2*dim (subsequent layers,
        // because forward and backward are concatenated).
        self._forwardLstms.wrappedValue = (0..<layers).map { i in
            LSTM(inputSize: i == 0 ? dim : 2 * dim, hiddenSize: dim)
        }
        self._backwardLstms.wrappedValue = (0..<layers).map { i in
            LSTM(inputSize: i == 0 ? dim : 2 * dim, hiddenSize: dim)
        }
        self._linear.wrappedValue = Linear(2 * dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let y = x

        var framed = false
        var nframes = 0
        var seq: MLXArray

        if let maxSteps, t > maxSteps {
            // Frame-based processing for long sequences
            let width = maxSteps
            let stride = width / 2
            let frames = unfold(x, width: width, stride: stride)
            nframes = frames.dim(2)
            framed = true
            // frames: (B, C, nframes, width) → (B*nframes, width, C)
            seq = frames.transposed(0, 2, 3, 1).reshaped([b * nframes, width, c])
        } else {
            // Direct: (B, C, T) → (B, T, C)
            seq = x.transposed(0, 2, 1)
        }

        // Run forward and backward LSTMs
        for i in 0..<lstmLayers {
            let (fOut, _) = forwardLstms[i](seq)
            let bIn = reversed(seq, axis: 1)
            let (bOut, _) = backwardLstms[i](bIn)
            let bOutReversed = reversed(bOut, axis: 1)
            seq = concatenated([fOut, bOutReversed], axis: -1)
        }

        var result = linear(seq)
        // (B, T, C) → (B, C, T)
        result = result.transposed(0, 2, 1)

        if framed {
            // Reassemble frames: (B*nframes, C, width) → (B, nframes, C, width)
            let width = maxSteps!
            let stride = width / 2
            let limit = stride / 2
            let frames = result.reshaped([b, nframes, c, width])
            var out: [MLXArray] = []
            for k in 0..<nframes {
                let frame = frames[0..., k, 0..., 0...]
                if k == 0 {
                    out.append(frame[0..., 0..., 0..<(width - limit)])
                } else if k == nframes - 1 {
                    out.append(frame[0..., 0..., limit...])
                } else {
                    out.append(frame[0..., 0..., limit..<(width - limit)])
                }
            }
            result = concatenated(out, axis: -1)
            result = result[0..., 0..., 0..<t]
        }

        if skip {
            result = result + y
        }
        return result
    }

    /// Extract overlapping frames from input (with padding to cover the full sequence).
    /// Input: (B, C, T), output: (B, C, nframes, width)
    private func unfold(_ x: MLXArray, width: Int, stride: Int) -> MLXArray {
        let t = x.dim(2)
        let nframes = Int(ceil(Double(t) / Double(stride)))
        let targetLength = (nframes - 1) * stride + width
        let pad = targetLength - t

        var padded = x
        if pad > 0 {
            let widths: [IntOrPair] = [0, 0, IntOrPair((0, pad))]
            padded = MLX.padded(x, widths: widths, mode: .constant)
        }

        var frames: [MLXArray] = []
        frames.reserveCapacity(nframes)
        for i in 0..<nframes {
            let start = i * stride
            let end = start + width
            frames.append(padded[0..., 0..., start..<end])
        }
        return stacked(frames, axis: 2)
    }

    /// Reverse along an axis.
    private func reversed(_ x: MLXArray, axis: Int) -> MLXArray {
        let n = x.dim(axis)
        let indices = MLXArray((0..<n).reversed())
        return x.take(indices, axis: axis)
    }
}
