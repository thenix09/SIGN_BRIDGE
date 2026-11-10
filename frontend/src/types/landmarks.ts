/**
 * Hand landmark and detection types for SignBridge MediaPipe pipeline.
 */

export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export type Handedness = 'Left' | 'Right';

export interface DetectedHandData {
  landmarks: HandLandmark[];
  handedness: Handedness;
  score: number;
}

export type DetectionStatus =
  | 'idle'
  | 'initializing'
  | 'detecting'
  | 'no_hands'
  | 'error';

export interface HandConnection {
  start: number;
  end: number;
}

export const HAND_CONNECTIONS: HandConnection[] = [
  // Thumb
  { start: 0, end: 1 },
  { start: 1, end: 2 },
  { start: 2, end: 3 },
  { start: 3, end: 4 },
  // Index
  { start: 0, end: 5 },
  { start: 5, end: 6 },
  { start: 6, end: 7 },
  { start: 7, end: 8 },
  // Middle
  { start: 5, end: 9 },
  { start: 9, end: 10 },
  { start: 10, end: 11 },
  { start: 11, end: 12 },
  // Ring
  { start: 9, end: 13 },
  { start: 13, end: 14 },
  { start: 14, end: 15 },
  { start: 15, end: 16 },
  // Pinky
  { start: 13, end: 17 },
  { start: 0, end: 17 },
  { start: 17, end: 18 },
  { start: 18, end: 19 },
  { start: 19, end: 20 }
];
