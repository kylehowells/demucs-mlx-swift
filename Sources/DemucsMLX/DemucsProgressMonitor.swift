import Foundation

/// Token used to cancel an in-progress separation.
/// Create one and pass it to the async separation API, then call `cancel()` to request cancellation.
public final class DemucsCancelToken: @unchecked Sendable {
	private var _isCancelled: Bool = false
	private let lock = NSLock()

	public init() {}

	/// Whether cancellation has been requested.
	public var isCancelled: Bool {
		self.lock.lock()
		defer { self.lock.unlock() }
		return self._isCancelled
	}

	/// Request cancellation of the separation.
	public func cancel() {
		self.lock.lock()
		self._isCancelled = true
		self.lock.unlock()
	}
}

/// Progress information passed to the progress callback.
public struct DemucsSeparationProgress: Sendable {
	/// Overall progress from 0.0 to 1.0.
	public let fraction: Float

	/// A human-readable description of the current stage.
	public let stage: String

	/// Time elapsed since separation started.
	public let elapsedTime: TimeInterval

	/// Estimated time remaining until completion, or nil if not yet estimable.
	public let estimatedTimeRemaining: TimeInterval?
}

/// Internal monitor that the SeparationEngine uses to report progress and check cancellation.
struct SeparationMonitor: @unchecked Sendable {
	let cancelToken: DemucsCancelToken?
	let progressHandler: (@Sendable (_ progress: Float, _ stage: String) -> Void)?

	var isCancelled: Bool {
		return self.cancelToken?.isCancelled ?? false
	}

	func checkCancellation() throws {
		if self.isCancelled {
			throw DemucsError.cancelled
		}
	}

	func reportProgress(_ progress: Float, stage: String) {
		self.progressHandler?(progress, stage)
	}

	/// Create a sub-monitor that maps local progress [0, 1] into [start, end] of the parent range.
	/// Cancellation is shared. Use this to give a model forward pass its own progress slice.
	func scoped(start: Float, end: Float) -> SeparationMonitor {
		SeparationMonitor(
			cancelToken: self.cancelToken,
			progressHandler: { localFraction, stage in
				let mapped = start + localFraction * (end - start)
				self.progressHandler?(mapped, stage)
			}
		)
	}
}

// MARK: - Progress Interpolator

/// Tracks batch timing, estimates ETA, and emits interpolated progress during GPU execution gaps.
///
/// MLX uses lazy evaluation: model sub-steps (encoder, transformer, decoder) fire instantly during
/// graph construction, then the GPU executes the entire batch at once during `MLX.eval()`. This
/// creates ~2-3s gaps between batches with no progress updates.
///
/// The interpolator detects these gaps and emits smooth intermediate progress updates based on
/// observed batch timing, so clients see a continuously-moving progress bar.
///
/// **Threading:** All mutable state is accessed exclusively on the main queue — no locks needed.
/// `onProgress()` captures the arrival timestamp then dispatches to main queue async, so the
/// separation thread is never blocked by progress reporting.
final class ProgressInterpolator: @unchecked Sendable {
	private let userCallback: @Sendable (DemucsSeparationProgress) -> Void
	private let startTime: CFAbsoluteTime // immutable after init

	// All mutable state below is accessed only from the main queue.

	// Real progress state
	private var lastRealFraction: Float = 0
	private var lastRealTime: CFAbsoluteTime
	private var lastEmittedFraction: Float = -1
	private var lastStage: String = ""

	// Batch timing estimates
	// Each completed batch = a burst of instant sub-step updates followed by a GPU execution gap.
	// We track the fraction at the end of each burst and the subsequent gap duration.
	private var completedBatchEndFractions: [Float] = []
	private var completedGapDurations: [TimeInterval] = []
	private var avgBatchRange: Float = 0
	private var avgGapDuration: TimeInterval = 0
	private var hasEstimates: Bool = false

	// Timer for interpolation
	private var timer: DispatchSourceTimer?
	private var stopped = false

	init(callback: @escaping @Sendable (DemucsSeparationProgress) -> Void) {
		let now = CFAbsoluteTimeGetCurrent()
		self.startTime = now
		self.lastRealTime = now
		self.userCallback = callback
	}

	/// Called from the separation (background) queue. Non-blocking — captures the timestamp
	/// and dispatches state updates to the main queue.
	func onProgress(_ fraction: Float, stage: String) {
		let arrivalTime = CFAbsoluteTimeGetCurrent()
		DispatchQueue.main.async { [self] in
			self.handleProgress(fraction, stage: stage, arrivalTime: arrivalTime)
		}
	}

	/// Stop the interpolation timer. Non-blocking.
	func stop() {
		DispatchQueue.main.async { [self] in
			self.stopped = true
			self.timer?.cancel()
			self.timer = nil
		}
	}

	// MARK: - Main-Queue State Management

	private func handleProgress(_ fraction: Float, stage: String, arrivalTime: CFAbsoluteTime) {
		guard !stopped else { return }

		let elapsed = arrivalTime - startTime
		let timeSinceLast = arrivalTime - lastRealTime
		let prevFraction = lastRealFraction

		// Detect batch gap ending: >100ms since last update means GPU just finished a batch.
		// The previous burst ended at prevFraction; the gap was timeSinceLast.
		if timeSinceLast > 0.1 && prevFraction > 0 {
			completedBatchEndFractions.append(prevFraction)
			completedGapDurations.append(timeSinceLast)
			recalculateEstimates()
		}

		lastRealFraction = fraction
		lastRealTime = arrivalTime
		lastStage = stage

		// Only emit if progressing forward (interpolation may have gone ahead)
		if fraction >= lastEmittedFraction {
			lastEmittedFraction = fraction
			let eta = estimateTimeRemaining(fraction: fraction, elapsed: elapsed)
			emit(fraction: fraction, stage: stage, elapsed: elapsed, eta: eta)
		}

		ensureTimerRunning()
	}

	// MARK: - Estimation

	private func recalculateEstimates() {
		var ranges: [Float] = []
		for i in 0..<completedBatchEndFractions.count {
			let prevEnd: Float = i > 0 ? completedBatchEndFractions[i - 1] : 0
			ranges.append(completedBatchEndFractions[i] - prevEnd)
		}
		if !ranges.isEmpty {
			avgBatchRange = ranges.reduce(0, +) / Float(ranges.count)
			avgGapDuration = completedGapDurations.reduce(0, +) / Double(completedGapDurations.count)
			hasEstimates = true
		}
	}

	private func estimateTimeRemaining(fraction: Float, elapsed: TimeInterval) -> TimeInterval? {
		guard fraction > 0.02 else { return nil }
		return elapsed / Double(fraction) * Double(1 - fraction)
	}

	// MARK: - Timer (main queue)

	private func ensureTimerRunning() {
		guard timer == nil, !stopped else { return }
		let t = DispatchSource.makeTimerSource(queue: .main)
		t.schedule(deadline: .now() + 0.05, repeating: 0.05, leeway: .milliseconds(10))
		t.setEventHandler { [weak self] in
			self?.tick()
		}
		t.resume()
		timer = t
	}

	private func tick() {
		guard !stopped, hasEstimates else { return }

		let now = CFAbsoluteTimeGetCurrent()
		let elapsed = now - startTime
		let gapElapsed = now - lastRealTime

		// Only interpolate if we're in a gap (>100ms since last real update)
		guard gapElapsed > 0.1 else { return }

		// Estimate how far through the current batch gap we are.
		// Cap at 95% to avoid overshooting — the real update will provide the true value.
		let gapProgress = min(Float(gapElapsed / avgGapDuration), 0.95)
		let interpolated = lastRealFraction + avgBatchRange * gapProgress
		let clamped = min(interpolated, 0.99)

		if clamped > lastEmittedFraction + 0.001 {
			lastEmittedFraction = clamped
			let eta = estimateTimeRemaining(fraction: clamped, elapsed: elapsed)
			emit(fraction: clamped, stage: lastStage, elapsed: elapsed, eta: eta)
		}
	}

	private func emit(fraction: Float, stage: String, elapsed: TimeInterval, eta: TimeInterval?) {
		userCallback(DemucsSeparationProgress(
			fraction: fraction,
			stage: stage,
			elapsedTime: elapsed,
			estimatedTimeRemaining: eta
		))
	}
}
