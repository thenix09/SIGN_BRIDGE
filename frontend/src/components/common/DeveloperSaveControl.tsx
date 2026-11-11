import React, { useState, useCallback, useRef, useEffect } from 'react';
import { DetectedHandData, DetectedPoseData } from '../../types/landmarks';
import { apiClient } from '../../services/api.client';

interface DeveloperSaveControlProps {
  detectedHands: DetectedHandData[];
  detectedPose: DetectedPoseData | null;
  cameraActive: boolean;
}

export const DeveloperSaveControl: React.FC<DeveloperSaveControlProps> = ({
  detectedHands,
  detectedPose,
  cameraActive
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [selectedLabel, setSelectedLabel] = useState<string>('A');
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleSave = useCallback(async () => {
    if (isSaving || !cameraActive || detectedHands.length === 0) return;

    setIsSaving(true);
    setSaveMessage(null);

    const hand = detectedHands[0];
    const landmarkArray = hand.landmarks.map((lm) => [lm.x, lm.y, lm.z]);

    try {
      const response = await apiClient.saveSample({
        label: selectedLabel.toUpperCase(),
        handedness: hand.handedness,
        landmarks: landmarkArray,
        group: 'developer'
      });

      setSaveMessage({
        type: 'success',
        text: `Sample #${response.id} recorded for label "${selectedLabel}" (total: ${response.count_for_label})`
      });
    } catch (err: unknown) {
      const errMsg = (err as Error).message || 'Failed to save sample';
      setSaveMessage({
        type: 'error',
        text: `Error saving sample: ${errMsg}`
      });
    } finally {
      setIsSaving(false);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        setSaveMessage(null);
      }, 3500);
    }
  }, [isSaving, cameraActive, detectedHands, selectedLabel]);

  const hasHand = detectedHands.length > 0;
  const canSave = cameraActive && hasHand && !isSaving;

  return (
    <div className="dev-save-container">
      <button
        type="button"
        className="dev-toggle-btn"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
      >
        <span>Developer Landmark Tool</span>
        <span className="dev-toggle-arrow">{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <div className="dev-panel-content">
          <p className="dev-help-text">
            Developer utility to record actual MediaPipe landmarks directly into <code>/api/samples</code> with <code>group: "developer"</code>.
          </p>

          <div className="dev-telemetry-row">
            <span className="dev-badge">
              Hands: {hasHand ? `${detectedHands.length} (${detectedHands[0].handedness})` : 'None'}
            </span>
            <span className="dev-badge">
              Pose: {detectedPose ? `${detectedPose.landmarks.length} pts` : 'None'}
            </span>
          </div>

          <div className="dev-action-row">
            <label className="dev-label">
              Label:
              <select
                value={selectedLabel}
                onChange={(e) => setSelectedLabel(e.target.value)}
                className="dev-select"
                disabled={isSaving}
              >
                {Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)).map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              className="button dev-save-btn"
              onClick={handleSave}
              disabled={!canSave}
            >
              {isSaving ? 'Saving…' : 'Save Landmarks'}
            </button>
          </div>

          {!cameraActive && (
            <p className="dev-warning">Camera must be active to capture data.</p>
          )}
          {cameraActive && !hasHand && (
            <p className="dev-warning">Hold a hand in frame to enable capture.</p>
          )}

          {saveMessage && (
            <div className={`dev-message dev-message-${saveMessage.type}`}>
              {saveMessage.type === 'success' ? '✓ ' : '⚠ '}
              {saveMessage.text}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
