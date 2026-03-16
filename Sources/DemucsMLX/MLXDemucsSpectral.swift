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

    // MARK: - iSTFT: GPU irfft + GPU overlap-add (Metal kernel)

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

        // GPU overlap-add via fused Metal kernel (no CPU roundtrip)
        let windowSqArray = MLXArray(windowSquared)
        return metalISTFTOverlapAdd(
            windowed: windowed,
            windowSq: windowSqArray,
            numFrames: frames,
            nFFT: nFFT,
            hopLength: hopLength,
            targetLength: length,
            center: center,
            finalShape: finalShapePrefix
        )
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
