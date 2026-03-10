import Accelerate
import Dispatch
import Foundation
import MLX

struct DemucsComplexSpectrogram {
    let real: MLXArray
    let imag: MLXArray
}

final class DemucsSpectralPair {
    let nFFT: Int
    let hopLength: Int
    let freqBins: Int
    let center: Bool
    let windowArray: MLXArray
    let windowSquared: [Float]

    init(nFFT: Int, hopLength: Int, center: Bool = true) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.freqBins = nFFT / 2 + 1
        self.center = center

        let window = (0..<nFFT).map { n in
            0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(nFFT)))
        }
        self.windowArray = MLXArray(window)
        self.windowSquared = window.map { $0 * $0 }
    }

    // MARK: - GPU STFT using MLX rfft

    func stft(_ x: MLXArray) -> DemucsComplexSpectrogram {
        precondition(x.ndim == 3)  // [B, C, T]

        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)

        // Reflect pad along time axis
        var padded = x
        if center {
            let pad = nFFT / 2
            padded = reflectPadGPU(x, pad: pad, axis: 2, length: t)
        }

        let paddedT = padded.dim(2)
        let frameCount = max(0, 1 + (paddedT - nFFT) / hopLength)

        // Flatten B*C for frame extraction
        let flat = padded.reshaped([b * c, paddedT])  // [BC, paddedT]

        // Extract overlapping frames using asStrided
        let frames = asStrided(
            flat, [b * c, frameCount, nFFT],
            strides: [paddedT, hopLength, 1]
        )  // [BC, frames, nFFT]

        // Apply Hann window
        let windowed = frames * windowArray  // [BC, frames, nFFT]

        // Batch rfft on GPU
        let spectrum = MLXFFT.rfft(windowed)  // [BC, frames, freqBins] complex

        // Extract real and imaginary parts
        let real = spectrum.realPart()  // [BC, frames, freqBins]
        let imag = spectrum.imaginaryPart()  // [BC, frames, freqBins]

        // Reshape to [B, C, freqBins, frames] — transpose freq and time dims
        let realOut = real.reshaped([b, c, frameCount, freqBins]).transposed(0, 1, 3, 2)
        let imagOut = imag.reshaped([b, c, frameCount, freqBins]).transposed(0, 1, 3, 2)

        return DemucsComplexSpectrogram(real: realOut, imag: imagOut)
    }

    // MARK: - iSTFT: GPU irfft + CPU overlap-add

    func istft(_ z: DemucsComplexSpectrogram, length: Int) -> MLXArray {
        precondition(z.real.shape == z.imag.shape)
        precondition(z.real.ndim == 4 || z.real.ndim == 5)

        let ndim = z.real.ndim
        let outer: Int
        let finalShapePrefix: [Int]
        let frames: Int

        if ndim == 4 {
            // [B, C, freqBins, frames]
            let b = z.real.dim(0)
            let c = z.real.dim(1)
            frames = z.real.dim(3)
            outer = b * c
            finalShapePrefix = [b, c]
        } else {
            // [B, S, C, freqBins, frames]
            let b = z.real.dim(0)
            let s = z.real.dim(1)
            let c = z.real.dim(2)
            frames = z.real.dim(4)
            outer = b * s * c
            finalShapePrefix = [b, s, c]
        }

        // Reshape to [outer, freqBins, frames] then transpose to [outer, frames, freqBins]
        let realT = z.real.reshaped([outer, freqBins, frames]).transposed(0, 2, 1)
        let imagT = z.imag.reshaped([outer, freqBins, frames]).transposed(0, 2, 1)

        // Create complex array and batch irfft on GPU
        let complex = realT + imagT.asImaginary()  // [outer, frames, freqBins] complex
        let timeFrames = MLXFFT.irfft(complex, n: nFFT)  // [outer, frames, nFFT]

        // Apply window on GPU
        let windowed = timeFrames * windowArray  // [outer, frames, nFFT]

        // Evaluate to bring to CPU for overlap-add
        MLX.eval(windowed)

        // Overlap-add on CPU (parallelized across channels)
        let windowVals = windowArray.asArray(Float.self)
        let frameData = windowed.asArray(Float.self)

        let rawLength = nFFT + max(0, frames - 1) * hopLength
        let eps: Float = 1e-8

        // Precompute window squared denominator (shared across all channels)
        var windowDenom = [Float](repeating: 0, count: rawLength)
        windowDenom.withUnsafeMutableBufferPointer { denom in
            windowSquared.withUnsafeBufferPointer { wsq in
                for fi in 0..<frames {
                    let start = fi * hopLength
                    vDSP_vadd(
                        denom.baseAddress! + start, 1,
                        wsq.baseAddress!, 1,
                        denom.baseAddress! + start, 1,
                        vDSP_Length(nFFT)
                    )
                }
            }
        }
        // Apply epsilon floor and invert for multiplication
        let invDenom: [Float] = (0..<rawLength).map { 1.0 / max(windowDenom[$0], eps) }

        let centerOffset = center ? (nFFT / 2) : 0
        let outAll = UnsafeMutableBufferPointer<Float>.allocate(capacity: outer * length)
        outAll.initialize(repeating: 0)
        let outPtr = outAll.baseAddress!

        // Parallel overlap-add: each outer channel is independent
        let capturedNFFT = nFFT
        let capturedHopLength = hopLength
        frameData.withUnsafeBufferPointer { frameBuf in
            invDenom.withUnsafeBufferPointer { invBuf in
                DispatchQueue.concurrentPerform(iterations: outer) { o in
                    var signal = [Float](repeating: 0, count: rawLength)
                    signal.withUnsafeMutableBufferPointer { sig in
                        for fi in 0..<frames {
                            let srcBase = (o * frames + fi) * capturedNFFT
                            let start = fi * capturedHopLength
                            vDSP_vadd(
                                sig.baseAddress! + start, 1,
                                frameBuf.baseAddress! + srcBase, 1,
                                sig.baseAddress! + start, 1,
                                vDSP_Length(capturedNFFT)
                            )
                        }

                        // Normalize by inverse window squared sum
                        vDSP_vmul(
                            sig.baseAddress!, 1,
                            invBuf.baseAddress!, 1,
                            sig.baseAddress!, 1,
                            vDSP_Length(rawLength)
                        )

                        // Copy trimmed result
                        let copyLen = min(length, max(0, rawLength - 2 * centerOffset))
                        let base = o * length
                        memcpy(outPtr + base, sig.baseAddress! + centerOffset, copyLen * MemoryLayout<Float>.size)
                    }
                }
            }
        }

        let result = MLXArray(Array(outAll)).reshaped(finalShapePrefix + [length])
        outAll.deallocate()
        return result
    }

    // MARK: - Reflect padding on GPU

    private func reflectPadGPU(_ x: MLXArray, pad: Int, axis: Int, length: Int) -> MLXArray {
        guard pad > 0 else { return x }

        // Build reflect-pad indices: [pad, pad-1, ..., 1, 0, 1, ..., T-1, T-2, ..., T-1-pad+1]
        var indices = [Int32]()
        indices.reserveCapacity(length + 2 * pad)

        // Left reflection: indices pad, pad-1, ..., 1
        for i in stride(from: pad, through: 1, by: -1) {
            indices.append(Int32(i))
        }
        // Original signal
        for i in 0..<length {
            indices.append(Int32(i))
        }
        // Right reflection: indices T-2, T-3, ..., T-1-pad
        for i in 0..<pad {
            indices.append(Int32(length - 2 - i))
        }

        let idxArray = MLXArray(indices)
        return x.take(idxArray, axis: axis)
    }
}
