import { CameraErrorInfo, CameraErrorType } from '../types/camera';

/**
 * CameraService manages camera stream lifecycle and error normalization.
 * Follows strict modular separation from presentation components.
 */
class CameraService {
  private stream: MediaStream | null = null;

  /**
   * Check whether MediaDevices API is supported in the current environment.
   */
  isSupported(): boolean {
    return Boolean(
      typeof navigator !== 'undefined' &&
        navigator.mediaDevices &&
        typeof navigator.mediaDevices.getUserMedia === 'function'
    );
  }

  /**
   * Request camera stream with default 640x480 constraints.
   * Avoids re-initialization if a live stream is already active.
   */
  async startCamera(constraints?: MediaStreamConstraints): Promise<MediaStream> {
    // Avoid unnecessary reinitialization if current stream is healthy and active
    if (this.isStreaming() && this.stream) {
      return this.stream;
    }

    // Stop any stale tracks before requesting a new stream
    this.stopCamera();

    if (!this.isSupported()) {
      const error = new Error('MediaDevices API is not supported in this browser or environment.');
      error.name = 'UnsupportedError';
      throw this.normalizeError(error);
    }

    const defaultConstraints: MediaStreamConstraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        constraints || defaultConstraints
      );
      this.stream = stream;
      return stream;
    } catch (err: unknown) {
      this.stopCamera();
      throw this.normalizeError(err);
    }
  }

  /**
   * Cleanly stop all active video tracks and release media devices.
   */
  stopCamera(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore individual track termination errors
        }
      });
      this.stream = null;
    }
  }

  /**
   * Get active MediaStream reference, if any.
   */
  getStream(): MediaStream | null {
    return this.stream;
  }

  /**
   * Check whether the camera is actively capturing frames.
   */
  isStreaming(): boolean {
    if (!this.stream || !this.stream.active) {
      return false;
    }
    const videoTracks = this.stream.getVideoTracks();
    return videoTracks.length > 0 && videoTracks.some((t) => t.readyState === 'live');
  }

  /**
   * Normalize browser DOMExceptions into structured, actionable error objects.
   */
  private normalizeError(err: unknown): CameraErrorInfo {
    let type: CameraErrorType = 'UnknownError';
    let message = 'An unexpected camera error occurred.';
    let actionableGuide = 'Please refresh the page and try again.';

    if (err instanceof DOMException || (err instanceof Error && err.name)) {
      switch (err.name) {
        case 'NotAllowedError':
        case 'PermissionDeniedError':
          type = 'NotAllowedError';
          message = 'Camera permission was denied or dismissed.';
          actionableGuide =
            'Click the camera/lock icon in your browser address bar to allow camera access, then retry.';
          break;

        case 'NotFoundError':
        case 'DevicesNotFoundError':
          type = 'NotFoundError';
          message = 'No video capture device (webcam) was detected on this system.';
          actionableGuide = 'Ensure your webcam is securely connected, turned on, and recognized by your OS.';
          break;

        case 'NotReadableError':
        case 'TrackStartError':
          type = 'NotReadableError';
          message = 'The webcam is currently in use by another application or blocked at the OS level.';
          actionableGuide =
            'Close other applications that might be using the camera (e.g. Zoom, Teams, FaceTime) and try again.';
          break;

        case 'UnsupportedError':
          type = 'UnsupportedError';
          message = 'MediaDevices API is unavailable in this environment.';
          actionableGuide =
            'Make sure you are accessing the app over HTTPS or http://localhost in a modern browser.';
          break;

        default:
          type = 'UnknownError';
          message = err.message || message;
          break;
      }
    } else if (typeof err === 'string') {
      message = err;
    }

    return { type, message, actionableGuide };
  }
}

export const cameraService = new CameraService();
