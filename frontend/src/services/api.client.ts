import {
  HealthResponse,
  VocabularyResponse,
  GesturePrediction,
  BackendVocabularyResponse,
  RecognizeWordsRequest,
  RecognizeWordsResponse,
  RecognizeRequest,
  RecognizeResponse,
  CreateSampleRequest,
  CreateSampleResponse,
  TextToSignRequest,
  TextToSignResponse,
  HistoryListResponse,
  OkResponse,
} from '../types/api';

declare global {
  interface ImportMeta {
    readonly env: Record<string, string | undefined>;
  }
}

/**
 * SignBridge Unified API Client.
 * Interfaces with backend frozen REST endpoints:
 * - GET  /api/health
 * - GET  /api/vocabulary
 * - POST /api/recognize
 * - POST /api/session/reset
 * - POST /api/session/backspace
 * - POST /api/session/commit
 * - POST /api/recognize-words
 * - POST /api/samples
 * - GET  /api/samples/stats
 * - POST /api/text-to-sign
 * - GET  /api/history
 * - DELETE /api/history
 */
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl =
      baseUrl ||
      (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_URL) ||
      '/api';
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Health check endpoint (/api/health).
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch {
      return {
        status: 'degraded',
        version: '1.0.0-offline',
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Baseline letter vocabulary stub for autocomplete/hints.
   */
  async getVocabulary(): Promise<VocabularyResponse> {
    return {
      supported_letters: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'O', 'V', 'W', 'Y'],
      supported_words: ['hello', 'thank you', 'please', 'yes', 'no', 'help', 'goodbye'],
      languages: ['ASL'],
    };
  }

  /**
   * Fetch closed word vocabulary from backend (GET /api/vocabulary).
   * Falls back to local 30-word specification when backend is offline.
   */
  async getWordVocabulary(language: string = 'ASL'): Promise<BackendVocabularyResponse> {
    try {
      const res = await fetch(`${this.baseUrl}/vocabulary?language=${encodeURIComponent(language)}`);
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Fallback below
    }

    // Specification 30-word closed vocabulary fallback
    return {
      size: 30,
      status: 'available',
      words: [
        { label: 'hello', category: 'greetings' },
        { label: 'thankyou', category: 'polite' },
        { label: 'please', category: 'polite' },
        { label: 'yes', category: 'responses' },
        { label: 'no', category: 'responses' },
        { label: 'help', category: 'emergency' },
        { label: 'eat', category: 'actions' },
        { label: 'drink', category: 'actions' },
        { label: 'water', category: 'objects' },
        { label: 'more', category: 'descriptors' },
        { label: 'stop', category: 'actions' },
        { label: 'again', category: 'descriptors' },
        { label: 'finish', category: 'actions' },
        { label: 'good', category: 'descriptors' },
        { label: 'bad', category: 'descriptors' },
        { label: 'happy', category: 'feelings' },
        { label: 'sad', category: 'feelings' },
        { label: 'like', category: 'feelings' },
        { label: 'want', category: 'feelings' },
        { label: 'need', category: 'feelings' },
        { label: 'who', category: 'questions' },
        { label: 'what', category: 'questions' },
        { label: 'where', category: 'questions' },
        { label: 'when', category: 'questions' },
        { label: 'why', category: 'questions' },
        { label: 'book', category: 'objects' },
        { label: 'table', category: 'objects' },
        { label: 'chair', category: 'objects' },
        { label: 'shoe', category: 'objects' },
        { label: 'pen', category: 'objects' },
      ],
    };
  }

  /**
   * Word sequence recognition (POST /api/recognize-words).
   */
  async recognizeWord(req: RecognizeWordsRequest): Promise<RecognizeWordsResponse> {
    const res = await fetch(`${this.baseUrl}/recognize-words`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      const detail = errorData.detail || errorData.error;
      throw new Error(detail || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Real letter recognition call (POST /api/recognize).
   */
  async recognizeLetter(req: RecognizeRequest): Promise<RecognizeResponse> {
    const res = await fetch(`${this.baseUrl}/recognize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Reset session state (/api/session/reset).
   */
  async resetSession(sessionId: string): Promise<OkResponse> {
    const res = await fetch(`${this.baseUrl}/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`Session reset failed: ${res.statusText}`);
    return res.json();
  }

  /**
   * Backspace session character (/api/session/backspace).
   */
  async backspaceSession(sessionId: string): Promise<RecognizeResponse> {
    const res = await fetch(`${this.baseUrl}/session/backspace`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`Session backspace failed: ${res.statusText}`);
    return res.json();
  }

  /**
   * Commit session word (/api/session/commit).
   */
  async commitSession(sessionId: string, word?: string): Promise<RecognizeResponse> {
    const res = await fetch(`${this.baseUrl}/session/commit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, word }),
    });
    if (!res.ok) throw new Error(`Session commit failed: ${res.statusText}`);
    return res.json();
  }

  /**
   * Developer / Dataset sample creation (POST /api/samples).
   */
  async saveSample(req: CreateSampleRequest): Promise<CreateSampleResponse> {
    const res = await fetch(`${this.baseUrl}/samples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        label: req.label.trim().toUpperCase(),
        handedness: req.handedness,
        landmarks: req.landmarks,
        group: req.group || 'developer',
      }),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Text-to-Sign synthesis (POST /api/text-to-sign).
   */
  async textToSign(req: TextToSignRequest): Promise<TextToSignResponse> {
    const res = await fetch(`${this.baseUrl}/text-to-sign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: req.language || 'ASL',
        text: req.text,
      }),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Translation history list (GET /api/history).
   */
  async getHistory(limit: number = 50): Promise<HistoryListResponse> {
    const res = await fetch(`${this.baseUrl}/history?limit=${limit}`);
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `HTTP ${res.status}`);
    }
    return await res.json();
  }

  /**
   * Purge translation history (DELETE /api/history).
   */
  async clearHistory(): Promise<OkResponse> {
    const res = await fetch(`${this.baseUrl}/history`, {
      method: 'DELETE',
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Mock prediction for testing fallback.
   */
  async mockPredict(): Promise<GesturePrediction> {
    return {
      label: 'A',
      confidence: 0.95,
      handedness: 'Right',
      timestamp: Date.now(),
    };
  }
}

export const apiClient = new ApiClient();
export default apiClient;
