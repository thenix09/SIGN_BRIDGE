import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  HandLandmarkerResult,
  PoseLandmarkerResult
} from '@mediapipe/tasks-vision';

export interface MediaPipeCombinedResult {
  hands: HandLandmarkerResult | null;
  pose: PoseLandmarkerResult | null;
}

/**
 * Service encapsulating MediaPipe HandLandmarker and PoseLandmarker
 * initialization, execution, and cleanup.
 * Reuses a single camera frame timestamp and WebGL context.
 */
class MediaPipeService {
  private handLandmarker: HandLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;
  private isInitializing: boolean = false;
  private isDetecting: boolean = false;
  private initPromise: Promise<void> | null = null;

  /**
   * Initialize HandLandmarker and PoseLandmarker. Reuses existing instances if ready.
   */
  async init(): Promise<void> {
    if (this.handLandmarker && this.poseLandmarker) {
      return;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    this.isInitializing = true;

    this.initPromise = (async () => {
      try {
        // Resolve WASM binaries (try local /wasm first, fallback to CDN)
        let visionWasm;
        try {
          visionWasm = await FilesetResolver.forVisionTasks('/wasm');
        } catch {
          visionWasm = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm'
          );
        }

        // 1. Initialize HandLandmarker
        if (!this.handLandmarker) {
          try {
            this.handLandmarker = await HandLandmarker.createFromOptions(visionWasm, {
              baseOptions: {
                modelAssetPath: '/models/hand_landmarker.task',
                delegate: 'GPU'
              },
              runningMode: 'VIDEO',
              numHands: 2,
              minHandDetectionConfidence: 0.5,
              minHandPresenceConfidence: 0.5,
              minTrackingConfidence: 0.5
            });
          } catch (gpuError) {
            console.warn('GPU delegate failed for HandLandmarker, falling back to CPU:', gpuError);
            this.handLandmarker = await HandLandmarker.createFromOptions(visionWasm, {
              baseOptions: {
                modelAssetPath: '/models/hand_landmarker.task',
                delegate: 'CPU'
              },
              runningMode: 'VIDEO',
              numHands: 2,
              minHandDetectionConfidence: 0.5,
              minHandPresenceConfidence: 0.5,
              minTrackingConfidence: 0.5
            });
          }
        }

        // 2. Initialize PoseLandmarker (Upper body tracking for Words mode)
        if (!this.poseLandmarker) {
          try {
            this.poseLandmarker = await PoseLandmarker.createFromOptions(visionWasm, {
              baseOptions: {
                modelAssetPath: '/models/pose_landmarker.task',
                delegate: 'GPU'
              },
              runningMode: 'VIDEO',
              numPoses: 1,
              minPoseDetectionConfidence: 0.5,
              minPosePresenceConfidence: 0.5,
              minTrackingConfidence: 0.5
            });
          } catch (poseGpuErr) {
            console.warn('GPU delegate failed for PoseLandmarker, falling back to CPU:', poseGpuErr);
            try {
              this.poseLandmarker = await PoseLandmarker.createFromOptions(visionWasm, {
                baseOptions: {
                  modelAssetPath: '/models/pose_landmarker.task',
                  delegate: 'CPU'
                },
                runningMode: 'VIDEO',
                numPoses: 1,
                minPoseDetectionConfidence: 0.5,
                minPosePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
              });
            } catch (poseErr) {
              console.warn('PoseLandmarker initialization optional fallback failed:', poseErr);
            }
          }
        }
      } finally {
        this.isInitializing = false;
        this.initPromise = null;
      }
    })();

    return this.initPromise;
  }

  /**
   * Synchronously execute hand and pose detection on a single video frame.
   * Prevents overlapping detection calls to avoid queue buildup and latency spikes.
   */
  detectForVideo(video: HTMLVideoElement, timestamp: number): MediaPipeCombinedResult | null {
    if (!this.handLandmarker || this.isDetecting) {
      return null;
    }

    // Ensure video has received playable frame data
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    try {
      this.isDetecting = true;
      const hands = this.handLandmarker.detectForVideo(video, timestamp);
      let pose: PoseLandmarkerResult | null = null;

      if (this.poseLandmarker) {
        try {
          pose = this.poseLandmarker.detectForVideo(video, timestamp);
        } catch (poseErr) {
          console.warn('Pose detection frame error:', poseErr);
        }
      }

      return { hands, pose };
    } catch (err) {
      console.error('Error during MediaPipe detection:', err);
      return null;
    } finally {
      this.isDetecting = false;
    }
  }

  /**
   * Release WebGL and WebAssembly resources allocated by HandLandmarker and PoseLandmarker.
   */
  close(): void {
    if (this.handLandmarker) {
      try {
        this.handLandmarker.close();
      } catch (err) {
        console.warn('Error closing HandLandmarker instance:', err);
      }
      this.handLandmarker = null;
    }

    if (this.poseLandmarker) {
      try {
        this.poseLandmarker.close();
      } catch (err) {
        console.warn('Error closing PoseLandmarker instance:', err);
      }
      this.poseLandmarker = null;
    }

    this.isDetecting = false;
    this.isInitializing = false;
    this.initPromise = null;
  }

  /**
   * Check whether HandLandmarker is initialized and ready for inference.
   */
  isReady(): boolean {
    return Boolean(this.handLandmarker);
  }

  /**
   * Check whether PoseLandmarker is ready.
   */
  isPoseReady(): boolean {
    return Boolean(this.poseLandmarker);
  }

  /**
   * Check whether initialization is in progress.
   */
  isLoading(): boolean {
    return this.isInitializing;
  }
}

export const mediaPipeService = new MediaPipeService();
