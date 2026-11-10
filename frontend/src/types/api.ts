/**
 * SignBridge API contract types (owned by Person A, consumed by Person C).
 */

export interface HealthResponse {
  status: 'ok' | 'degraded' | string;
  model_loaded?: boolean;
  word_model_loaded?: boolean;
  version: string;
  timestamp?: string;
}

export interface VocabularyWord {
  label: string;
  category: string;
}

export interface VocabularyResponse {
  size?: number;
  status?: 'available' | 'withheld' | 'experimental' | 'unavailable';
  words?: VocabularyWord[];
  supported_letters?: string[];
  supported_words?: string[];
  languages?: string[];
}

export interface GesturePrediction {
  label: string;
  confidence: number;
  handedness?: 'Left' | 'Right';
  timestamp: number;
}

/* ============================================================
   Backend Frozen Schemas (POST /api/recognize, etc.)
   ============================================================ */

export interface RecognizeRequest {
  session_id: string;
  language?: string;
  handedness?: 'Left' | 'Right';
  landmarks: number[][]; // exactly 21 landmarks [x, y, z]
}

export interface RecognizeResponse {
  label: string;
  confidence: number;
  state: 'IDLE' | 'DETECTING' | 'STABILISING' | 'CONFIRMED' | string;
  confirmed_letter: string | null;
  current_word: string;
  sentence: string;
  suggestions: string[];
}

export interface SessionActionRequest {
  session_id: string;
  word?: string;
}

export interface SessionResetRequest {
  session_id: string;
}

export interface OkResponse {
  ok: boolean;
}

/* ============================================================
   Words Mode Schemas (GET /api/vocabulary, POST /api/recognize-words)
   ============================================================ */

export interface BackendVocabularyResponse {
  size: number;
  status: 'available' | 'experimental' | 'withheld' | 'unavailable';
  words: VocabularyWord[];
}

export interface WordFrameInput {
  pose?: number[][] | null; // [x, y] or [x, y, z]
  hands?: number[][][];     // up to 2 hands, 21 landmarks each
}

export interface RecognizeWordsRequest {
  session_id: string;
  language?: string;
  frame?: WordFrameInput | null;
  window?: WordFrameInput[] | null;
}

export interface WordCandidate {
  label: string;
  confidence: number;
}

export interface RecognizeWordsResponse {
  state: string; // 'WAITING' | 'SIGNING' | 'COMPLETE'
  top_k: WordCandidate[];
  accepted_word: string | null;
  sentence: string;
  message?: string | null;
}

/* ============================================================
   Sample Collection Schemas (POST /api/samples)
   ============================================================ */

export interface CreateSampleRequest {
  label: string;
  handedness: 'Left' | 'Right';
  landmarks: number[][]; // 21 landmarks [x, y, z]
  group?: string;
}

export interface CreateSampleResponse {
  id: number;
  count_for_label: number;
}

export interface SampleStatsResponse {
  counts: Record<string, number>;
  total: number;
}

/* ============================================================
   Text to Sign Schemas (POST /api/text-to-sign)
   ============================================================ */

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

/* ============================================================
   History Schemas (GET /api/history, DELETE /api/history)
   ============================================================ */

export interface HistoryItemResponse {
  id: number | null;
  direction: 'text_to_sign' | 'sign_to_text' | string;
  language: string;
  content: string;
  created_at: string;
}

export interface HistoryListResponse {
  items: HistoryItemResponse[];
}
