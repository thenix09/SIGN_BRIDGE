import React, { useEffect, useState, useCallback } from 'react';
import { CameraErrorInfo, CameraStatus } from '../../types/camera';
import { LandmarkCanvas } from './LandmarkCanvas';
import { DetectedHandData, DetectedPoseData, DetectionStatus } from '../../types/landmarks';

interface CameraFeedProps {
  status: CameraStatus;
  stream: MediaStream | null;
  error: CameraErrorInfo | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onRetry?: () => void;
  onHandsDetected?: (hands: DetectedHandData[]) => void;
  onPoseDetected?: (pose: DetectedPoseData | null) => void;
}

export const CameraFeed: React.FC<CameraFeedProps> = ({
  status,
  stream,
  error,
  videoRef,
  onRetry,
  onHandsDetected,
  onPoseDetected
}) => {
  const [resolution, setResolution] = useState<{ width: number; height: number } | null>(null);
  const [detectionStatus, setDetectionStatus] = useState<DetectionStatus>('idle');
  const [detectedHandCount, setDetectedHandCount] = useState<number>(0);

  // Bind media stream to HTMLVideoElement when stream becomes active
  useEffect(() => {
    const videoElement = videoRef.current;
    if (videoElement) {
      if (stream) {
        videoElement.srcObject = stream;
        videoElement.play().catch(() => {
          // Autoplay policy fallback: muted video will play
        });
      } else {
        videoElement.srcObject = null;
        setResolution(null);
        setDetectionStatus('idle');
        setDetectedHandCount(0);
      }
    }

    return () => {
      if (videoElement) {
        videoElement.srcObject = null;
      }
    };
  }, [stream, videoRef]);

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setResolution({
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight
      });
    }
  };

  const handleDetectionStatusChange = useCallback((newStatus: DetectionStatus, count: number) => {
    setDetectionStatus(newStatus);
    setDetectedHandCount(count);
  }, []);

  // Format detection pill status label and style
  const getDetectionLabel = () => {
    switch (detectionStatus) {
      case 'initializing':
        return { label: 'Loading MediaPipe...', className: 'detect-pill-init' };
      case 'detecting':
        return {
          label: `${detectedHandCount} ${detectedHandCount === 1 ? 'Hand' : 'Hands'} Detected`,
          className: 'detect-pill-active'
        };
      case 'no_hands':
        return { label: 'No hands detected', className: 'detect-pill-idle' };
      case 'error':
        return { label: 'Tracking Error', className: 'detect-pill-error' };
      case 'idle':
      default:
        return null;
    }
  };

  const detectionBadge = status === 'active' ? getDetectionLabel() : null;

  return (
    <div className="camera-viewport-wrapper">
      {/* Video Element (mirrored for natural interaction) */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        onLoadedMetadata={handleLoadedMetadata}
        className={`camera-video ${status === 'active' ? 'visible' : 'hidden'}`}
      />

      {/* MediaPipe Skeleton Canvas Overlay (mirrored math matches video) */}
      <LandmarkCanvas
        videoRef={videoRef}
        cameraActive={status === 'active'}
        onDetectionStatusChange={handleDetectionStatusChange}
        onHandsDetected={onHandsDetected}
        onPoseDetected={onPoseDetected}
      />

      {/* Viewfinder Corner Accents */}
      {status === 'active' && (
        <>
          <div className="viewfinder-corner viewfinder-top-left" aria-hidden="true" />
          <div className="viewfinder-corner viewfinder-top-right" aria-hidden="true" />
          <div className="viewfinder-corner viewfinder-bottom-left" aria-hidden="true" />
          <div className="viewfinder-corner viewfinder-bottom-right" aria-hidden="true" />
        </>
      )}

      {/* Live Badge overlay (top-left) */}
      {status === 'active' && (
        <div className="camera-live-badge">
          <span className="live-dot" />
          <span>LIVE</span>
          {resolution && (
            <span className="resolution-tag">
              {resolution.width}×{resolution.height}
            </span>
          )}
        </div>
      )}

      {/* Detection Status Pill (top-right) */}
      {detectionBadge && (
        <div className={`camera-detection-badge ${detectionBadge.className}`}>
          <span className="detection-dot" />
          <span>{detectionBadge.label}</span>
        </div>
      )}

      {/* State overlays */}
      {status === 'requesting' && (
        <div className="camera-state-overlay">
          <div className="spinner" />
          <h3>Requesting Camera Access</h3>
          <p>Please click "Allow" on your browser permission prompt.</p>
        </div>
      )}

      {status === 'idle' && (
        <div className="camera-state-overlay">
          <div className="camera-state-icon">📷</div>
          <h3>Camera is Off</h3>
          <p>Click "Start Camera" in the control panel to begin hand tracking.</p>
        </div>
      )}

      {status === 'denied' && (
        <div className="camera-state-overlay camera-state-error">
          <div className="camera-state-icon">🚫</div>
          <h3>Camera Permission Denied</h3>
          <p className="error-message-text">{error?.message}</p>
          <p className="action-guide-text">{error?.actionableGuide}</p>
          {onRetry && (
            <button className="button" onClick={onRetry} style={{ marginTop: '12px' }}>
              Retry Permission
            </button>
          )}
        </div>
      )}

      {status === 'error' && (
        <div className="camera-state-overlay camera-state-error">
          <div className="camera-state-icon">⚠️</div>
          <h3>Camera Error</h3>
          <p className="error-message-text">{error?.message || 'Failed to initialize camera.'}</p>
          <p className="action-guide-text">{error?.actionableGuide}</p>
          {onRetry && (
            <button className="button" onClick={onRetry} style={{ marginTop: '12px' }}>
              Try Again
            </button>
          )}
        </div>
      )}
    </div>
  );
};
