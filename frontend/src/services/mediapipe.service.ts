import {
  HandLandmarker,
  FilesetResolver,
  HandLandmarkerResult
} from '@mediapipe/tasks-vision';

/**
 * Service encapsulating MediaPipe HandLandmarker initialization, execution, and cleanup.
 * Uses local model assets and WASM binaries with fallback capability.
 */
class MediaPipeService {
  private handLandmarker: HandLandmarker | null = null;
  private isInitializing: boolean = false;
  private isDetecting: boolean = false;
  private initPromise: Promise<HandLandmarker> | null = null;

  /**
   * Initialize HandLandmarker instance. Reuses existing instance if already ready.
   */
  async init(): Promise<HandLandmarker> {
    if (this.handLandmarker) {
      return this.handLandmarker;
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

        // Initialize HandLandmarker with GPU delegate first, fallback to CPU
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
          console.warn('GPU delegate failed for MediaPipe HandLandmarker, falling back to CPU:', gpuError);
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

        return this.handLandmarker;
      } finally {
        this.isInitializing = false;
        this.initPromise = null;
      }
    })();

    return this.initPromise;
  }

  /**
   * Synchronously or asynchronously execute hand detection on a single video frame.
   * Prevents overlapping detection calls to avoid queue buildup and latency spikes.
   */
  detectForVideo(video: HTMLVideoElement, timestamp: number): HandLandmarkerResult | null {
    if (!this.handLandmarker || this.isDetecting) {
      return null;
    }

    // Ensure video has received playable frame data
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    try {
      this.isDetecting = true;
      const result = this.handLandmarker.detectForVideo(video, timestamp);
      return result;
    } catch (err) {
      console.error('Error during MediaPipe hand detection:', err);
      return null;
    } finally {
      this.isDetecting = false;
    }
  }

  /**
   * Release WebGL and WebAssembly resources allocated by HandLandmarker.
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
   * Check whether initialization is in progress.
   */
  isLoading(): boolean {
    return this.isInitializing;
  }
}

export const mediaPipeService = new MediaPipeService();
