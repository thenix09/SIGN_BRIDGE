/**
 * Hand and Pose landmark types for SignBridge MediaPipe pipeline.
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

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface DetectedPoseData {
  landmarks: PoseLandmark[];
}

export type DetectionStatus =
  | 'idle'
  | 'initializing'
  | 'detecting'
  | 'no_hands'
  | 'error';

export interface LandmarkConnection {
  start: number;
  end: number;
}

/** 21-joint connections for Hand Landmarker */
export const HAND_CONNECTIONS: LandmarkConnection[] = [
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

/**
 * 9 Upper-body pose landmark indices required by Build Spec v2 and Model Card:
 * 0: nose, 9: mouth left, 10: mouth right, 11: left shoulder, 12: right shoulder,
 * 13: left elbow, 14: right elbow, 15: left wrist, 16: right wrist.
 */
export const UPPER_BODY_POSE_INDICES = [0, 9, 10, 11, 12, 13, 14, 15, 16] as const;

/** Upper body connections for drawing the pose skeleton */
export const UPPER_BODY_POSE_CONNECTIONS: LandmarkConnection[] = [
  // Face anchors
  { start: 9, end: 10 }, // mouth line
  { start: 0, end: 9 },  // nose to left mouth
  { start: 0, end: 10 }, // nose to right mouth
  // Torso / Shoulders
  { start: 11, end: 12 }, // shoulder to shoulder
  // Left arm
  { start: 11, end: 13 }, // shoulder to elbow
  { start: 13, end: 15 }, // elbow to wrist
  // Right arm
  { start: 12, end: 14 }, // shoulder to elbow
  { start: 14, end: 16 }, // elbow to wrist
  // Upper torso links (if hips 23, 24 available in 33-point pose)
  { start: 11, end: 23 },
  { start: 12, end: 24 },
  { start: 23, end: 24 }
];
