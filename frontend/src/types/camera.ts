/**
 * Camera status and configuration types for SignBridge.
 */
export type CameraStatus =
  | 'idle'
  | 'requesting'
  | 'active'
  | 'denied'
  | 'error';

export type CameraErrorType =
  | 'NotAllowedError'
  | 'NotFoundError'
  | 'NotReadableError'
  | 'UnsupportedError'
  | 'UnknownError';

export interface CameraErrorInfo {
  type: CameraErrorType;
  message: string;
  actionableGuide: string;
}

export interface CameraDimensions {
  width: number;
  height: number;
}
