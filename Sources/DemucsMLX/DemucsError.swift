import Foundation

public enum DemucsError: Error, LocalizedError {
    case unknownModel(String)
    case invalidAudioShape([Int])
    case invalidAudioData(String)
    case invalidParameter(String)
    case unsupportedModelBackend(String)
    case audioIO(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let name):
            return "Unknown model '\(name)'."
        case .invalidAudioShape(let shape):
            return "Expected audio tensor shape [channels, time], got \(shape)."
        case .invalidAudioData(let reason):
            return "Invalid audio data: \(reason)"
        case .invalidParameter(let reason):
            return "Invalid separation parameter: \(reason)"
        case .unsupportedModelBackend(let reason):
            return "Unsupported model backend: \(reason)"
        case .audioIO(let reason):
            return "Audio I/O error: \(reason)"
        case .cancelled:
            return "Separation was cancelled."
        }
    }
}
