/**
 * SignBridge API contract types (owned by Person A, consumed as stub by Person C).
 */

export interface HealthResponse {
  status: 'ok' | 'degraded';
  version: string;
  timestamp: string;
}

export interface VocabularyResponse {
  supported_letters: string[];
  supported_words: string[];
  languages: string[];
}

export interface GesturePrediction {
  label: string;
  confidence: number;
  handedness?: 'Left' | 'Right';
  timestamp: number;
}
