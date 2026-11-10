import React, { useEffect, useRef, useCallback } from 'react';
import { mediaPipeService } from '../../services/mediapipe.service';
import {
  DetectedHandData,
  DetectedPoseData,
  DetectionStatus,
  HAND_CONNECTIONS,
  Handedness,
  UPPER_BODY_POSE_CONNECTIONS,
  UPPER_BODY_POSE_INDICES
} from '../../types/landmarks';

interface LandmarkCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraActive: boolean;
  onDetectionStatusChange?: (status: DetectionStatus, handCount: number) => void;
  onHandsDetected?: (hands: DetectedHandData[]) => void;
  onPoseDetected?: (pose: DetectedPoseData | null) => void;
}

// Visual themes for left and right hands to ensure high contrast and clear distinction
const HAND_THEMES = {
  Right: {
    lineColor: '#38bdf8', // Sky blue
    jointFill: '#0284c7',
    jointBorder: '#ffffff',
    tipFill: '#f0f9ff',
    badgeBg: 'rgba(2, 132, 199, 0.85)',
    badgeText: '#ffffff'
  },
  Left: {
    lineColor: '#34d399', // Emerald
    jointFill: '#059669',
    jointBorder: '#ffffff',
    tipFill: '#ecfdf5',
    badgeBg: 'rgba(5, 150, 105, 0.85)',
    badgeText: '#ffffff'
  }
};

// Theme for upper-body pose skeleton
const POSE_THEME = {
  lineColor: 'rgba(192, 132, 252, 0.75)', // Light purple/violet
  jointFill: '#9333ea',                  // Vibrant purple
  jointBorder: '#ffffff',
  anchorFill: '#38bdf8'                  // Cyan for face anchors
};

export const LandmarkCanvas: React.FC<LandmarkCanvasProps> = ({
  videoRef,
  cameraActive,
  onDetectionStatusChange,
  onHandsDetected,
  onPoseDetected
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const statusRef = useRef<DetectionStatus>('idle');
  const handCountRef = useRef<number>(0);

  const updateStatus = useCallback(
    (newStatus: DetectionStatus, newCount: number) => {
      if (statusRef.current !== newStatus || handCountRef.current !== newCount) {
        statusRef.current = newStatus;
        handCountRef.current = newCount;
        onDetectionStatusChange?.(newStatus, newCount);
      }
    },
    [onDetectionStatusChange]
  );

  useEffect(() => {
    let isCancelled = false;

    if (!cameraActive) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      updateStatus('idle', 0);
      onHandsDetected?.([]);
      onPoseDetected?.(null);
      return;
    }

    const startDetectionLoop = async () => {
      updateStatus('initializing', 0);

      try {
        await mediaPipeService.init();
      } catch (err) {
        console.error('Failed to initialize MediaPipe models:', err);
        updateStatus('error', 0);
        return;
      }

      if (isCancelled) return;

      const processFrame = () => {
        if (isCancelled) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (!video || !canvas || video.paused || video.ended) {
          animationFrameRef.current = requestAnimationFrame(processFrame);
          return;
        }

        // Synchronize internal canvas buffer with video dimensions
        if (
          video.videoWidth > 0 &&
          (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)
        ) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
          animationFrameRef.current = requestAnimationFrame(processFrame);
          return;
        }

        // Only run detection when a new video frame has arrived
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;

          const timestamp = performance.now();
          const combined = mediaPipeService.detectForVideo(video, timestamp);

          // Clear previous skeleton overlays
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Helper: Transform landmark coordinates to match mirrored video display
          // Video uses scaleX(-1), so screen X = (1 - landmark.x) * canvasWidth
          const getScreenPoint = (lm: { x: number; y: number }) => ({
            x: (1 - lm.x) * canvas.width,
            y: lm.y * canvas.height
          });

          // ------------------------------------------------------------------
          // 1. Draw Pose Tracking Skeleton (Upper body)
          // ------------------------------------------------------------------
          const rawPose = combined?.pose?.landmarks?.[0];
          if (rawPose && rawPose.length >= 17) {
            // Draw upper body connections
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = POSE_THEME.lineColor;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            UPPER_BODY_POSE_CONNECTIONS.forEach(({ start, end }) => {
              if (start < rawPose.length && end < rawPose.length) {
                const p1 = getScreenPoint(rawPose[start]);
                const p2 = getScreenPoint(rawPose[end]);
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
              }
            });

            // Draw upper body joints (9 key landmarks)
            UPPER_BODY_POSE_INDICES.forEach((idx) => {
              if (idx < rawPose.length) {
                const pt = getScreenPoint(rawPose[idx]);
                const isAnchor = idx === 0 || idx === 9 || idx === 10;
                const radius = isAnchor ? 3.5 : 4.5;

                ctx.beginPath();
                ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
                ctx.fillStyle = POSE_THEME.jointBorder;
                ctx.fill();

                ctx.beginPath();
                ctx.arc(pt.x, pt.y, radius - 1.5, 0, 2 * Math.PI);
                ctx.fillStyle = isAnchor ? POSE_THEME.anchorFill : POSE_THEME.jointFill;
                ctx.fill();
              }
            });

            onPoseDetected?.({
              landmarks: rawPose.map((lm) => ({
                x: lm.x,
                y: lm.y,
                z: lm.z,
                visibility: lm.visibility
              }))
            });
          } else {
            onPoseDetected?.(null);
          }

          // ------------------------------------------------------------------
          // 2. Draw Hand Tracking Skeleton & Handedness Badges
          // ------------------------------------------------------------------
          const handResults = combined?.hands;
          if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
            const detectedHands: DetectedHandData[] = [];

            handResults.landmarks.forEach((landmarks, index) => {
              const handednessCategory = handResults.handedness?.[index]?.[0];
              const rawHandedness = (handednessCategory?.categoryName || 'Right') as Handedness;
              const score = handednessCategory?.score ?? 0.9;

              detectedHands.push({
                landmarks,
                handedness: rawHandedness,
                score
              });

              const theme = HAND_THEMES[rawHandedness] || HAND_THEMES.Right;

              // 1. Draw Skeleton Connections
              ctx.lineWidth = 3;
              ctx.strokeStyle = theme.lineColor;
              ctx.lineCap = 'round';
              ctx.lineJoin = 'round';

              HAND_CONNECTIONS.forEach(({ start, end }) => {
                const p1 = getScreenPoint(landmarks[start]);
                const p2 = getScreenPoint(landmarks[end]);
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
              });

              // 2. Draw 21 Landmark Joints
              const fingertipIndices = [4, 8, 12, 16, 20];
              landmarks.forEach((lm, pointIdx) => {
                const pt = getScreenPoint(lm);
                const isFingertip = fingertipIndices.includes(pointIdx);
                const radius = isFingertip ? 5.5 : 4;

                ctx.beginPath();
                ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
                ctx.fillStyle = isFingertip ? theme.tipFill : theme.jointBorder;
                ctx.fill();

                ctx.beginPath();
                ctx.arc(pt.x, pt.y, radius - 1.8, 0, 2 * Math.PI);
                ctx.fillStyle = theme.jointFill;
                ctx.fill();
              });

              // 3. Draw Handedness Label Pill near wrist (landmark 0)
              const wristPt = getScreenPoint(landmarks[0]);
              const labelText = `${rawHandedness} (${Math.round(score * 100)}%)`;

              ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
              const textMetrics = ctx.measureText(labelText);
              const paddingX = 8;
              const pillWidth = textMetrics.width + paddingX * 2;
              const pillHeight = 20;
              const pillX = Math.max(10, Math.min(canvas.width - pillWidth - 10, wristPt.x - pillWidth / 2));
              const pillY = Math.max(24, Math.min(canvas.height - 10, wristPt.y + 24));

              ctx.fillStyle = theme.badgeBg;
              ctx.beginPath();
              ctx.roundRect(pillX, pillY - pillHeight + 4, pillWidth, pillHeight, 6);
              ctx.fill();

              ctx.fillStyle = theme.badgeText;
              ctx.textBaseline = 'middle';
              ctx.fillText(labelText, pillX + paddingX, pillY - pillHeight / 2 + 4);
            });

            updateStatus('detecting', handResults.landmarks.length);
            onHandsDetected?.(detectedHands);
          } else {
            updateStatus('no_hands', 0);
            onHandsDetected?.([]);
          }
        }

        animationFrameRef.current = requestAnimationFrame(processFrame);
      };

      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    startDetectionLoop();

    return () => {
      isCancelled = true;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [cameraActive, videoRef, updateStatus, onHandsDetected, onPoseDetected]);

  // Clean up MediaPipe resources on component unmount
  useEffect(() => {
    return () => {
      mediaPipeService.close();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`landmark-canvas ${cameraActive ? 'active' : 'inactive'}`}
      aria-hidden="true"
    />
  );
};
