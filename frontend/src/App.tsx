import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { CameraFeed } from './components/camera/CameraFeed';
import { CameraControls } from './components/camera/CameraControls';
import { cameraService } from './services/camera.service';
import { apiClient } from './services/api.client';
import { CameraStatus, CameraErrorInfo } from './types/camera';
import { BadgeStatus } from './components/common/StatusBadge';
import { DetectedHandData } from './types/landmarks';
import { RecognizeResponse } from './types/api';

const SESSION_ID = 'user-webcam-session';

export const App: React.FC = () => {
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<CameraErrorInfo | null>(null);

  // Real-time recognition state from backend
  const [recognizeResult, setRecognizeResult] = useState<RecognizeResponse>({
    label: '',
    confidence: 0,
    state: 'IDLE',
    confirmed_letter: null,
    current_word: '',
    sentence: '',
    suggestions: [],
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastSendTimeRef = useRef<number>(0);
  const isRecognizingRef = useRef<boolean>(false);

  // Cleanly stop tracks on component unmount
  useEffect(() => {
    return () => {
      cameraService.stopCamera();
    };
  }, []);

  const handleStartCamera = useCallback(async () => {
    setCameraStatus('requesting');
    setCameraError(null);

    try {
      const stream = await cameraService.startCamera();
      setCameraStream(stream);
      setCameraStatus('active');
    } catch (err: unknown) {
      const errorInfo = err as CameraErrorInfo;
      setCameraError(errorInfo);
      if (errorInfo.type === 'NotAllowedError') {
        setCameraStatus('denied');
      } else {
        setCameraStatus('error');
      }
    }
  }, []);

  const handleStopCamera = useCallback(() => {
    cameraService.stopCamera();
    setCameraStream(null);
    setCameraStatus('idle');
    setCameraError(null);
  }, []);

  // Frame handler called from MediaPipe loop in LandmarkCanvas
  const handleHandsDetected = useCallback(async (hands: DetectedHandData[]) => {
    if (!hands || hands.length === 0) return;

    const now = performance.now();
    // Throttle recognition requests to ~15 fps (every 66ms) to prevent network queuing
    if (now - lastSendTimeRef.current < 66 || isRecognizingRef.current) {
      return;
    }

    const primaryHand = hands[0];
    const rawLandmarks = primaryHand.landmarks.map((lm) => [lm.x, lm.y, lm.z]);

    lastSendTimeRef.current = now;
    isRecognizingRef.current = true;

    try {
      const res = await apiClient.recognizeLetter({
        session_id: SESSION_ID,
        language: 'ASL',
        handedness: primaryHand.handedness,
        landmarks: rawLandmarks,
      });
      setRecognizeResult(res);
    } catch {
      // Ignore transient frame request errors
    } finally {
      isRecognizingRef.current = false;
    }
  }, []);

  const handleBackspace = async () => {
    try {
      const res = await apiClient.backspaceSession(SESSION_ID);
      setRecognizeResult(res);
    } catch (err) {
      console.error('Backspace failed', err);
    }
  };

  const handleCommit = async (wordToCommit?: string) => {
    try {
      const res = await apiClient.commitSession(SESSION_ID, wordToCommit);
      setRecognizeResult(res);
    } catch (err) {
      console.error('Commit failed', err);
    }
  };

  const handleClear = async () => {
    try {
      await apiClient.resetSession(SESSION_ID);
      setRecognizeResult({
        label: '',
        confidence: 0,
        state: 'IDLE',
        confirmed_letter: null,
        current_word: '',
        sentence: '',
        suggestions: [],
      });
    } catch (err) {
      console.error('Reset failed', err);
    }
  };

  // Compute status for the main app header
  const getHeaderStatus = (): { status: BadgeStatus; label: string } => {
    switch (cameraStatus) {
      case 'active':
        return { status: 'online', label: 'Camera Live' };
      case 'requesting':
        return { status: 'requesting', label: 'Requesting...' };
      case 'denied':
        return { status: 'warning', label: 'Camera Denied' };
      case 'error':
        return { status: 'error', label: 'Camera Error' };
      case 'idle':
      default:
        return { status: 'idle', label: 'Camera Off' };
    }
  };

  const headerMeta = getHeaderStatus();

  return (
    <MainLayout headerStatus={headerMeta.status} headerLabel={headerMeta.label}>
      {(activeTab) => (
        <>
          {/* Primary Viewport Column */}
          <section className="card camera-card" aria-label="Camera and Sign Recognition Viewport">
            <div className="card-title">
              <h2>Camera Feed</h2>
              <span className="brand-badge">Mode: {activeTab.toUpperCase()}</span>
            </div>

            <CameraFeed
              status={cameraStatus}
              stream={cameraStream}
              error={cameraError}
              videoRef={videoRef}
              onRetry={handleStartCamera}
              onHandsDetected={handleHandsDetected}
            />

            {/* Live Recognition Banner below camera */}
            <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Status:</span>
                <span className={`detect-pill detect-pill-${recognizeResult.state.toLowerCase()}`} style={{ fontWeight: 600 }}>
                  {recognizeResult.state}
                </span>
              </div>

              {recognizeResult.label && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Detected:</span>
                  <span style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38bdf8' }}>
                    {recognizeResult.label}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                    ({Math.round(recognizeResult.confidence * 100)}%)
                  </span>
                </div>
              )}
            </div>
          </section>

          {/* Controls & Live Output Sidebar Column */}
          <aside className="card" aria-label="Controls and System Status">
            <div className="card-title">
              <h3>Camera Controls</h3>
            </div>

            <CameraControls
              status={cameraStatus}
              onStart={handleStartCamera}
              onStop={handleStopCamera}
            />

            {/* Live Recognition Output Panel */}
            <div style={{ marginTop: '20px', borderTop: '1px solid var(--color-border)', paddingTop: '16px' }}>
              <h4 style={{ fontSize: '0.95rem', marginBottom: '8px', color: '#38bdf8' }}>Live Recognition</h4>

              <div style={{ marginBottom: '12px' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', display: 'block' }}>Current Word:</label>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, minHeight: '2rem', letterSpacing: '2px', color: 'var(--color-text)' }}>
                  {recognizeResult.current_word || <span style={{ color: 'var(--color-border)' }}>_</span>}
                </div>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', display: 'block' }}>Sentence:</label>
                <div style={{ fontSize: '1.05rem', minHeight: '1.8rem', color: 'var(--color-text)' }}>
                  {recognizeResult.sentence || <span style={{ color: 'var(--color-text-muted)', fontStyle: 'italic' }}>No words committed yet</span>}
                </div>
              </div>

              {/* Suggestions */}
              {recognizeResult.suggestions.length > 0 && (
                <div style={{ marginBottom: '14px' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', display: 'block', marginBottom: '4px' }}>Suggestions:</label>
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                    {recognizeResult.suggestions.map((sug) => (
                      <button
                        key={sug}
                        onClick={() => handleCommit(sug)}
                        style={{
                          background: 'rgba(56, 189, 248, 0.15)',
                          border: '1px solid rgba(56, 189, 248, 0.4)',
                          borderRadius: '12px',
                          padding: '4px 10px',
                          color: '#38bdf8',
                          fontSize: '0.8rem',
                          cursor: 'pointer',
                        }}
                      >
                        {sug}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                <button
                  onClick={handleBackspace}
                  className="btn btn-secondary"
                  style={{ flex: 1, padding: '8px', fontSize: '0.85rem' }}
                >
                  Backspace
                </button>
                <button
                  onClick={() => handleCommit()}
                  className="btn btn-primary"
                  style={{ flex: 1, padding: '8px', fontSize: '0.85rem' }}
                >
                  Space / Commit
                </button>
                <button
                  onClick={handleClear}
                  className="btn btn-secondary"
                  style={{ padding: '8px 12px', fontSize: '0.85rem' }}
                >
                  Clear
                </button>
              </div>
            </div>
          </aside>
        </>
      )}
    </MainLayout>
  );
};

export default App;
