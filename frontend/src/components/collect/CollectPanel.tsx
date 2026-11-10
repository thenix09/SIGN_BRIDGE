import React, { useState, useEffect, useCallback, useRef } from 'react';
import { DetectedHandData } from '../../types/landmarks';
import { CollectStatus, SampleCounts, SAMPLES_TARGET, ALL_LETTERS } from '../../types/collect';
import { collectService } from '../../services/collect.service';

interface CollectPanelProps {
  /** Current detected hands from the MediaPipe pipeline. */
  detectedHands: DetectedHandData[];
  /** Whether the camera feed is currently active. */
  cameraActive: boolean;
}

/**
 * Dataset Collection Panel.
 *
 * Lets the user select a letter (A–Z), see how many samples they have
 * collected for it, and capture a new sample from the current hand pose.
 *
 * Data is submitted to POST /api/samples via collectService. The service
 * falls back to local counts if the backend is not reachable.
 */
export const CollectPanel: React.FC<CollectPanelProps> = ({ detectedHands, cameraActive }) => {
  const [selectedLetter, setSelectedLetter] = useState<string>('A');
  const [counts, setCounts] = useState<SampleCounts>({});
  const [captureStatus, setCaptureStatus] = useState<CollectStatus>('idle');
  const [isCapturing, setIsCapturing] = useState(false);
  const [lastSavedToBackend, setLastSavedToBackend] = useState<boolean | null>(null);

  // Timer ref to clear feedback flash automatically
  const feedbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load existing sample counts from the backend on mount
  useEffect(() => {
    collectService.loadStats().then((loadedCounts) => {
      setCounts(loadedCounts);
    });
  }, []);

  // Clear the feedback flash after a short delay
  const scheduleFeedbackClear = useCallback(() => {
    if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current);
    feedbackTimerRef.current = setTimeout(() => {
      setCaptureStatus('idle');
      setLastSavedToBackend(null);
    }, 2000);
  }, []);

  useEffect(() => {
    return () => {
      if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current);
    };
  }, []);

  const handleCapture = useCallback(async () => {
    if (isCapturing) return;

    // Refuse if camera is not running
    if (!cameraActive) {
      setCaptureStatus('no_hand');
      scheduleFeedbackClear();
      return;
    }

    // Refuse if no hands are detected
    if (detectedHands.length === 0) {
      setCaptureStatus('no_hand');
      scheduleFeedbackClear();
      return;
    }

    setIsCapturing(true);

    // Use the first detected hand
    const hand = detectedHands[0];

    try {
      const result = await collectService.captureSample(
        selectedLetter,
        hand.handedness,
        hand.landmarks
      );

      setCounts((prev) => ({ ...prev, [selectedLetter]: result.count }));
      setCaptureStatus('captured');
      setLastSavedToBackend(result.savedToBackend);
    } catch {
      setCaptureStatus('error');
      setLastSavedToBackend(false);
    } finally {
      setIsCapturing(false);
      scheduleFeedbackClear();
    }
  }, [isCapturing, cameraActive, detectedHands, selectedLetter, scheduleFeedbackClear]);

  const selectedCount = counts[selectedLetter] ?? 0;
  const progressPercent = Math.min((selectedCount / SAMPLES_TARGET) * 100, 100);
  const hasHand = detectedHands.length > 0;
  const captureDisabled = isCapturing || !cameraActive || !hasHand;

  return (
    <div className="collect-panel">
      {/* Panel heading */}
      <div className="card-title">
        <h3>Collect Samples</h3>
        <span className="brand-badge">Dataset Collection</span>
      </div>

      {/* Selected letter display + counter */}
      <div className="collect-letter-display">
        <span className="collect-big-letter">{selectedLetter}</span>
        <div className="collect-counter">
          <span className="collect-count-num">{selectedCount}</span>
          <span className="collect-count-sep"> / </span>
          <span className="collect-count-target">{SAMPLES_TARGET}</span>
          <span className="collect-count-label"> samples</span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="collect-progress-track" aria-label={`${selectedCount} of ${SAMPLES_TARGET} samples collected`}>
        <div
          className="collect-progress-bar"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Capture button + status feedback */}
      <div className="collect-capture-row">
        <button
          className="button collect-capture-btn"
          onClick={handleCapture}
          disabled={captureDisabled}
          aria-label={`Capture sample for letter ${selectedLetter}`}
        >
          {isCapturing ? '⏳ Capturing…' : '📸 Capture Sample'}
        </button>

        {/* Inline feedback */}
        {captureStatus === 'captured' && (
          <div className="collect-feedback collect-feedback-ok">
            ✓ Captured!
            {lastSavedToBackend === false && (
              <span className="collect-feedback-note"> Count updated for this session. Sample was not uploaded.</span>
            )}
          </div>
        )}
        {captureStatus === 'no_hand' && (
          <div className="collect-feedback collect-feedback-warn">
            ✋ Hold a hand in view first
          </div>
        )}
        {captureStatus === 'error' && (
          <div className="collect-feedback collect-feedback-error">
            ⚠ Capture failed
          </div>
        )}
      </div>

      {/* Camera / hand presence hint */}
      {!cameraActive && (
        <p className="collect-hint">Start the camera to capture samples.</p>
      )}
      {cameraActive && !hasHand && captureStatus === 'idle' && (
        <p className="collect-hint collect-hint-warn">No hand detected — show your hand to the camera.</p>
      )}

      {/* Letter selector grid A–Z */}
      <div className="collect-letter-grid" aria-label="Letter selector">
        {ALL_LETTERS.map((letter) => {
          const count = counts[letter] ?? 0;
          const done = count >= SAMPLES_TARGET;
          return (
            <button
              key={letter}
              className={`collect-letter-btn${selectedLetter === letter ? ' selected' : ''}${done ? ' done' : ''}`}
              onClick={() => setSelectedLetter(letter)}
              aria-label={`Select letter ${letter}, ${count} samples`}
              aria-pressed={selectedLetter === letter}
            >
              <span className="collect-letter-btn-char">{letter}</span>
              {count > 0 && (
                <span className="collect-letter-btn-badge">{count}</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};
