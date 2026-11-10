import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { ActiveTab } from './components/layout/Navigation';
import { CameraFeed } from './components/camera/CameraFeed';
import { CameraControls } from './components/camera/CameraControls';
import { CollectPanel } from './components/collect/CollectPanel';
import { SpellPanel } from './components/spell/SpellPanel';
import { WordsPanel } from './components/words/WordsPanel';
import { TextToSignPlayer } from './components/text_to_sign/TextToSignPlayer';
import { HistoryDrawer } from './components/history/HistoryDrawer';
import { DeveloperSaveControl } from './components/common/DeveloperSaveControl';
import { cameraService } from './services/camera.service';
import { CameraStatus, CameraErrorInfo } from './types/camera';
import { DetectedHandData, DetectedPoseData } from './types/landmarks';
import { BadgeStatus } from './components/common/StatusBadge';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('spell');
  const [isHistoryOpen, setIsHistoryOpen] = useState<boolean>(false);
  const [ttsInitialText, setTtsInitialText] = useState<string>('HELLO');

  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<CameraErrorInfo | null>(null);

  /** Latest hand landmark data surfaced from the MediaPipe pipeline. */
  const [lastDetectedHands, setLastDetectedHands] = useState<DetectedHandData[]>([]);

  /** Latest upper-body pose landmark data surfaced from MediaPipe. */
  const [lastDetectedPose, setLastDetectedPose] = useState<DetectedPoseData | null>(null);

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
    setLastDetectedHands([]);
    setLastDetectedPose(null);
  }, []);

  /**
   * Receives landmark updates from LandmarkCanvas via CameraFeed.
   * Keeps detected hands in App state so Collect, Spell, and Words panels
   * consume the same data without redundant MediaPipe loops.
   */
  const handleHandsDetected = useCallback((hands: DetectedHandData[]) => {
    setLastDetectedHands(hands);
  }, []);

  /**
   * Receives upper-body pose updates from LandmarkCanvas via CameraFeed.
   */
  const handlePoseDetected = useCallback((pose: DetectedPoseData | null) => {
    setLastDetectedPose(pose);
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

  const handleSelectHistoryPhrase = (phrase: string, direction: string) => {
    if (direction === 'text_to_sign') {
      setTtsInitialText(phrase);
      setActiveTab('text_to_sign');
    } else {
      setActiveTab('spell');
    }
  };

  const headerMeta = getHeaderStatus();

  return (
    <>
      <MainLayout
        headerStatus={headerMeta.status}
        headerLabel={headerMeta.label}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onOpenHistory={() => setIsHistoryOpen(true)}
      >
        {(tab) => {
          if (tab === 'text_to_sign') {
            return <TextToSignPlayer initialText={ttsInitialText} />;
          }

          return (
            <>
              {/* Primary Viewport Column */}
              <section className="card camera-card" aria-label="Camera and Sign Recognition Viewport">
                <div className="card-title">
                  <h2>Camera Feed</h2>
                  <span className="brand-badge">Mode: {tab.toUpperCase()}</span>
                </div>

                <CameraFeed
                  status={cameraStatus}
                  stream={cameraStream}
                  error={cameraError}
                  videoRef={videoRef}
                  onRetry={handleStartCamera}
                  onHandsDetected={handleHandsDetected}
                  onPoseDetected={handlePoseDetected}
                />
              </section>

              {/* Controls & Sidebar Column */}
              <aside className="card sidebar-card" aria-label="Controls and Mode Panel">
                {/* Camera controls — always visible */}
                <div className="card-title">
                  <h3>Camera Controls</h3>
                </div>

                <CameraControls
                  status={cameraStatus}
                  onStart={handleStartCamera}
                  onStop={handleStopCamera}
                />

                {/* Developer-only landmark save tool */}
                <DeveloperSaveControl
                  detectedHands={lastDetectedHands}
                  detectedPose={lastDetectedPose}
                  cameraActive={cameraStatus === 'active'}
                />

                {/* Divider */}
                <div className="sidebar-divider" />

                {/* Tab-specific panel */}
                {tab === 'collect' && (
                  <CollectPanel
                    detectedHands={lastDetectedHands}
                    cameraActive={cameraStatus === 'active'}
                  />
                )}

                {tab === 'spell' && (
                  <SpellPanel
                    detectedHands={lastDetectedHands}
                    cameraActive={cameraStatus === 'active'}
                  />
                )}

                {tab === 'words' && (
                  <WordsPanel
                    detectedHands={lastDetectedHands}
                    detectedPose={lastDetectedPose}
                    cameraActive={cameraStatus === 'active'}
                  />
                )}
              </aside>
            </>
          );
        }}
      </MainLayout>

      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectPhrase={handleSelectHistoryPhrase}
      />
    </>
  );
};

export default App;
