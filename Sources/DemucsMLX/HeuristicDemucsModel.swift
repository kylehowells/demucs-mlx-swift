import Foundation

final class HeuristicDemucsModel: StemSeparationModel {
    let descriptor: DemucsModelDescriptor

    init(descriptor: DemucsModelDescriptor) {
        self.descriptor = descriptor
    }

    func predict(
        batchData input: [Float],
        batchSize: Int,
        channels: Int,
        frames: Int,
        monitor: SeparationMonitor? = nil
    ) throws -> [Float] {
        let sourceCount = descriptor.sourceNames.count

        guard channels >= 1 else {
            throw DemucsError.invalidAudioData("Batch must have at least one channel")
        }
        let expectedCount = batchSize * channels * frames
        guard input.count == expectedCount else {
            throw DemucsError.invalidAudioData("batchData count mismatch")
        }

        var output = [Float](repeating: 0, count: batchSize * sourceCount * channels * frames)

        for b in 0..<batchSize {
            var left = [Float](repeating: 0, count: frames)
            var right = [Float](repeating: 0, count: frames)

            let base = b * channels * frames
            for t in 0..<frames {
                left[t] = input[base + t]
                right[t] = channels > 1 ? input[base + frames + t] : input[base + t]
            }

            var mid = [Float](repeating: 0, count: frames)
            var side = [Float](repeating: 0, count: frames)
            for t in 0..<frames {
                mid[t] = 0.5 * (left[t] + right[t])
                side[t] = 0.5 * (left[t] - right[t])
            }

            let bass = AudioDSP.onePoleLowpass(mid, cutoffHz: 220.0, sampleRate: descriptor.sampleRate)
            let bodyLow = AudioDSP.onePoleLowpass(mid, cutoffHz: 3_000.0, sampleRate: descriptor.sampleRate)
            let air = AudioDSP.highpass(mid, cutoffHz: 5_000.0, sampleRate: descriptor.sampleRate)

            var body = [Float](repeating: 0, count: frames)
            for t in 0..<frames {
                body[t] = bodyLow[t] - bass[t]
            }

            let drumTransient = AudioDSP.highpass(body, cutoffHz: 900.0, sampleRate: descriptor.sampleRate)

            var vocalsL = [Float](repeating: 0, count: frames)
            var vocalsR = [Float](repeating: 0, count: frames)
            var bassL = [Float](repeating: 0, count: frames)
            var bassR = [Float](repeating: 0, count: frames)
            var drumsL = [Float](repeating: 0, count: frames)
            var drumsR = [Float](repeating: 0, count: frames)
            var otherL = [Float](repeating: 0, count: frames)
            var otherR = [Float](repeating: 0, count: frames)

            for t in 0..<frames {
                let vocalsMid = 0.70 * body[t] + 0.15 * air[t] + 0.15 * bass[t]
                let drumsMid = 0.75 * air[t] + 0.25 * drumTransient[t]

                vocalsL[t] = AudioDSP.clampAudio(vocalsMid + 0.12 * side[t])
                vocalsR[t] = AudioDSP.clampAudio(vocalsMid - 0.12 * side[t])

                bassL[t] = AudioDSP.clampAudio(bass[t])
                bassR[t] = AudioDSP.clampAudio(bass[t])

                drumsL[t] = AudioDSP.clampAudio(drumsMid + 0.55 * side[t])
                drumsR[t] = AudioDSP.clampAudio(drumsMid - 0.55 * side[t])

                otherL[t] = AudioDSP.clampAudio(left[t] - vocalsL[t] - bassL[t] - drumsL[t])
                otherR[t] = AudioDSP.clampAudio(right[t] - vocalsR[t] - bassR[t] - drumsR[t])
            }

            for c in 0..<channels {
                let vocals = c == 0 ? vocalsL : vocalsR
                let bassStem = c == 0 ? bassL : bassR
                let drums = c == 0 ? drumsL : drumsR
                let other = c == 0 ? otherL : otherR

                for t in 0..<frames {
                    output[index(batch: b, source: 0, channel: c, time: t, sources: sourceCount, channels: channels, frames: frames)] = drums[t]
                    output[index(batch: b, source: 1, channel: c, time: t, sources: sourceCount, channels: channels, frames: frames)] = bassStem[t]
                    output[index(batch: b, source: 2, channel: c, time: t, sources: sourceCount, channels: channels, frames: frames)] = other[t]
                    output[index(batch: b, source: 3, channel: c, time: t, sources: sourceCount, channels: channels, frames: frames)] = vocals[t]
                }
            }
        }

        return output
    }

    @inline(__always)
    private func index(
        batch: Int,
        source: Int,
        channel: Int,
        time: Int,
        sources: Int,
        channels: Int,
        frames: Int
    ) -> Int {
        (((batch * sources + source) * channels + channel) * frames + time)
    }
}
