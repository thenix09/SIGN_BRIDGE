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
  OkResponse
} from '../types/api';

declare global {
  interface ImportMeta {
    readonly env: Record<string, string | undefined>;
  }
}

/**
 * SignBridge API Client for Day 3.
 * Interfaces with backend frozen REST endpoints:
 * - GET  /api/health
 * - GET  /api/vocabulary
 * - POST /api/recognize
 * - POST /api/recognize-words
 * - POST /api/samples
 * - GET  /api/samples/stats
 */
class ApiClient {
  private baseUrl: string;

  constructor(
    baseUrl: string = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api')
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
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
        timestamp: new Date().toISOString()
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
      languages: ['ASL']
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
      // Backend offline
    }

    // Default static closed-vocabulary fallback matching backend/data/vocabulary.json
    return {
      size: 30,
      status: 'withheld',
      words: [
        { label: 'hello', category: 'greetings' },
        { label: 'bye', category: 'greetings' },
        { label: 'thankyou', category: 'greetings' },
        { label: 'please', category: 'greetings' },
        { label: 'who', category: 'questions' },
        { label: 'where', category: 'questions' },
        { label: 'why', category: 'questions' },
        { label: 'if', category: 'questions' },
        { label: 'any', category: 'questions' },
        { label: 'mom', category: 'family' },
        { label: 'dad', category: 'family' },
        { label: 'brother', category: 'family' },
        { label: 'grandma', category: 'family' },
        { label: 'apple', category: 'food and drink' },
        { label: 'milk', category: 'food and drink' },
        { label: 'water', category: 'food and drink' },
        { label: 'pizza', category: 'food and drink' },
        { label: 'jump', category: 'actions' },
        { label: 'read', category: 'actions' },
        { label: 'sleep', category: 'actions' },
        { label: 'dance', category: 'actions' },
        { label: 'drink', category: 'actions' },
        { label: 'happy', category: 'feelings' },
        { label: 'sad', category: 'feelings' },
        { label: 'sick', category: 'feelings' },
        { label: 'book', category: 'objects' },
        { label: 'chair', category: 'objects' },
        { label: 'car', category: 'objects' },
        { label: 'shoe', category: 'objects' },
        { label: 'pen', category: 'objects' }
      ]
    };
  }

  /**
   * Word sequence recognition (POST /api/recognize-words).
   * Exact backend route: words.py -> @router.post("/recognize-words")
   */
  async recognizeWord(req: RecognizeWordsRequest): Promise<RecognizeWordsResponse> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/recognize-words`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      });
    } catch {
      throw new Error('Backend server is offline or unreachable on port 8000');
    }

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      const detail = errorData.detail || errorData.error;
      if (res.status === 404) {
        throw new Error('Endpoint /api/recognize-words returned 404 Not Found');
      }
      throw new Error(detail || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Real letter recognition call (POST /api/recognize).
   */
  async recognizeLetter(req: RecognizeRequest): Promise<RecognizeResponse> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      });
    } catch {
      throw new Error('Backend server is offline or unreachable on port 8000');
    }

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `HTTP ${res.status}`);
    }

    return await res.json();
  }

  /**
   * Developer / Dataset sample creation (POST /api/samples).
   */
  async saveSample(req: CreateSampleRequest): Promise<CreateSampleResponse> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/samples`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label: req.label.trim().toUpperCase(),
          handedness: req.handedness,
          landmarks: req.landmarks,
          group: req.group || 'developer'
        })
      });
    } catch {
      throw new Error('Backend server is offline or unreachable on port 8000');
    }

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
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/text-to-sign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language: req.language || 'ASL',
          text: req.text
        })
      });
    } catch {
      throw new Error('Backend server is offline or unreachable on port 8000');
    }

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
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/history?limit=${limit}`);
    } catch {
      throw new Error('Backend server is offline or unreachable on port 8000');
    }

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
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/history`, {
        method: 'DELETE'
      });
    } catch {
      throw new Error('Backend server is offline or unreachable on port 8000');
    }

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
      timestamp: Date.now()
    };
  }
}

export const apiClient = new ApiClient();
