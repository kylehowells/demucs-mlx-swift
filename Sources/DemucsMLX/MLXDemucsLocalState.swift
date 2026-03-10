import Foundation
import MLX
import MLXNN

/// Local attention module matching the Python MLX LocalState implementation.
/// Input/output shape: (B, C, T) in NCL format.
final class DemucsLocalState: Module, DemucsUnaryLayer {
    let heads: Int
    let nfreqs: Int
    let ndecay: Int

    @ModuleInfo(key: "content") var content: Conv1dNCL
    @ModuleInfo(key: "query") var query: Conv1dNCL
    @ModuleInfo(key: "key") var key_: Conv1dNCL
    @ModuleInfo(key: "proj") var proj: Conv1dNCL

    @ModuleInfo(key: "query_freqs") var queryFreqs: Conv1dNCL?
    @ModuleInfo(key: "query_decay") var queryDecay: Conv1dNCL?

    init(channels: Int, heads: Int = 4, nfreqs: Int = 0, ndecay: Int = 4) {
        precondition(channels % heads == 0, "channels must be divisible by heads")
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay

        self._content.wrappedValue = Conv1dNCL(channels, channels, kernelSize: 1)
        self._query.wrappedValue = Conv1dNCL(channels, channels, kernelSize: 1)
        self._key_.wrappedValue = Conv1dNCL(channels, channels, kernelSize: 1)

        let projInChannels = channels + heads * nfreqs
        self._proj.wrappedValue = Conv1dNCL(projInChannels, channels, kernelSize: 1)

        if nfreqs > 0 {
            self._queryFreqs.wrappedValue = Conv1dNCL(channels, heads * nfreqs, kernelSize: 1)
        } else {
            self._queryFreqs.wrappedValue = nil
        }

        if ndecay > 0 {
            self._queryDecay.wrappedValue = Conv1dNCL(channels, heads * ndecay, kernelSize: 1)
        } else {
            self._queryDecay.wrappedValue = nil
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let perHead = c / heads

        let (delta, eye) = deltaEye(t: t, dtype: x.dtype)

        // Multi-head reshape: (B, C, T) → (B, H, C/H, T)
        let queries = query(x).reshaped([b, heads, perHead, t])
        let keys = key_(x).reshaped([b, heads, perHead, t])

        // Attention: (B, H, T, C/H) @ (B, H, C/H, T) → (B, H, T, T)
        let keysT = keys.transposed(0, 1, 3, 2)
        var dots = matmul(keysT, queries) * MLXArray(Float(1.0 / sqrt(Float(perHead))))

        // Optional frequency-based attention
        if nfreqs > 0, let queryFreqs {
            let periods = MLXArray(Array(1...nfreqs)).asType(x.dtype)
            let freqKernel = cos(
                MLXArray(2.0 * Float.pi) * delta / periods.reshaped([nfreqs, 1, 1])
            )
            let freqQ = queryFreqs(x).reshaped([b, heads, nfreqs, t])
            let freqScale = Float(1.0 / sqrt(Float(nfreqs)))
            dots = dots + MLX.einsum("fts,bhfs->bhts", freqKernel, freqQ * MLXArray(freqScale))
        }

        // Optional decay term
        if ndecay > 0, let queryDecay {
            let decays = MLXArray(Array(1...ndecay)).asType(x.dtype)
            var decayQ = queryDecay(x).reshaped([b, heads, ndecay, t])
            decayQ = sigmoid(decayQ) * MLXArray(Float(0.5))

            // coeff = sum_f(decay_q * decays)
            let coeff = (decayQ * decays.reshaped([1, 1, ndecay, 1])).sum(axis: 2)
            let absDelta = abs(delta)
            let decayScale = Float(1.0 / sqrt(Float(ndecay)))
            dots = dots - (
                absDelta.reshaped([1, 1, t, t])
                    * coeff.reshaped([b, heads, 1, t])
                    * MLXArray(decayScale)
            )
        }

        // Mask diagonal and softmax
        dots = MLX.where(eye, MLXArray(Float(-100.0)).asType(dots.dtype), dots)
        let weights = softmax(dots, axis: 2)

        // Apply attention to content
        // PyTorch: result[b,h,c,s] = sum_t(weights[b,h,t,s] * content[b,h,c,t])
        // matmul(content, weights): (B,H,C/H,T) @ (B,H,T_key,T_query) = (B,H,C/H,T_query)
        let contentVal = content(x).reshaped([b, heads, perHead, t])
        var result = matmul(contentVal, weights)

        // Optional frequency signatures
        if nfreqs > 0 {
            let periods = MLXArray(Array(1...nfreqs)).asType(x.dtype)
            let freqKernel = cos(
                MLXArray(2.0 * Float.pi) * delta / periods.reshaped([nfreqs, 1, 1])
            )
            let timeSig = MLX.einsum("bhts,fts->bhfs", weights, freqKernel)
            result = concatenated([result, timeSig], axis: 2)
        }

        result = result.reshaped([b, -1, t])
        return x + proj(result)
    }

    /// Create delta (signed distance) and eye (diagonal mask) matrices.
    private func deltaEye(t: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let indices = MLXArray(0..<t).asType(dtype)
        let delta = indices.expandedDimensions(axis: 0) - indices.expandedDimensions(axis: 1)
        let eye = MLX.eye(t).asType(.bool)
        return (delta, eye)
    }
}
