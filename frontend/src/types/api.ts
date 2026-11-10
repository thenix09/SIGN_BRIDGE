/**
 * SignBridge API contract types (matching backend Section 12 Frozen Contract).
 */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  word_model_loaded: boolean;
  version: string;
}

export interface VocabularyWord {
  label: string;
  category: string;
}

export interface VocabularyResponse {
  size: number;
  status: 'available' | 'withheld' | 'experimental' | 'unavailable';
  words: VocabularyWord[];
}

export interface RecognizeRequest {
  session_id: string;
  language?: string;
  handedness?: 'Left' | 'Right';
  landmarks: number[][];
}

export interface RecognizeResponse {
  label: string;
  confidence: number;
  state: string; // 'IDLE' | 'DETECTING' | 'STABILISING' | 'CONFIRMED'
  confirmed_letter: string | null;
  current_word: string;
  sentence: string;
  suggestions: string[];
}

export interface TextToSignRequest {
  language?: string;
  text: string;
}

export interface SignFrameResponse {
  label: string;
  kind: 'word' | 'letter' | 'pause';
  pose: number[][];
  left_hand: number[][] | null;
  right_hand: number[][] | null;
  duration_ms: number;
}

export interface SignSegment {
  token: string;
  kind: 'word' | 'spelled';
  start_index: number;
  end_index: number;
}

export interface TextToSignResponse {
  frames: SignFrameResponse[];
  segments: SignSegment[];
  fallback_tokens: string[];
}
