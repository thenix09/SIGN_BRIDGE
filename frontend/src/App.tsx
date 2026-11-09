import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { CameraFeed } from './components/camera/CameraFeed';
import { CameraControls } from './components/camera/CameraControls';
import { cameraService } from './services/camera.service';
import { CameraStatus, CameraErrorInfo } from './types/camera';
import { BadgeStatus } from './components/common/StatusBadge';

export const App: React.FC = () => {
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<CameraErrorInfo | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);

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
            />
          </section>

          {/* Controls & Sidebar Column */}
          <aside className="card" aria-label="Controls and System Status">
            <div className="card-title">
              <h3>Camera Controls</h3>
            </div>

            <CameraControls
              status={cameraStatus}
              onStart={handleStartCamera}
              onStop={handleStopCamera}
            />

            <div style={{ marginTop: '16px', borderTop: '1px solid var(--color-border)', paddingTop: '16px' }}>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '8px' }}>Milestone 3: Hand Tracking</h4>
              <p style={{ fontSize: '0.85rem', marginBottom: '8px' }}>
                Real-time dual-hand tracking using Google MediaPipe HandLandmarker with 21-landmark skeleton overlay.
              </p>
              <ul style={{ paddingLeft: '18px', fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                <li>Dual-hand detection (Left & Right)</li>
                <li>21 3D coordinates per hand</li>
                <li>Mirrored canvas alignment</li>
                <li>Local model asset & WASM binaries</li>
              </ul>
            </div>
          </aside>
        </>
      )}
    </MainLayout>
  );
};

export default App;
