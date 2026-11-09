import {
  HealthResponse,
  VocabularyResponse,
  RecognizeRequest,
  RecognizeResponse,
  TextToSignRequest,
  TextToSignResponse,
} from '../types/api';

/**
 * SignBridge API Client.
 * Connects React frontend directly to the FastAPI backend endpoints.
 */
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  async checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  async getVocabulary(language: string = 'ASL'): Promise<VocabularyResponse> {
    const res = await fetch(`${this.baseUrl}/vocabulary?language=${encodeURIComponent(language)}`);
    if (!res.ok) throw new Error(`Vocabulary fetch failed: ${res.statusText}`);
    return res.json();
  }

  async recognizeLetter(request: RecognizeRequest): Promise<RecognizeResponse> {
    const res = await fetch(`${this.baseUrl}/recognize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) throw new Error(`Recognition failed: ${res.statusText}`);
    return res.json();
  }

  async resetSession(sessionId: string): Promise<{ ok: boolean }> {
    const res = await fetch(`${this.baseUrl}/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`Session reset failed: ${res.statusText}`);
    return res.json();
  }

  async backspaceSession(sessionId: string): Promise<RecognizeResponse> {
    const res = await fetch(`${this.baseUrl}/session/backspace`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`Session backspace failed: ${res.statusText}`);
    return res.json();
  }

  async commitSession(sessionId: string, word?: string): Promise<RecognizeResponse> {
    const res = await fetch(`${this.baseUrl}/session/commit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, word }),
    });
    if (!res.ok) throw new Error(`Session commit failed: ${res.statusText}`);
    return res.json();
  }

  async textToSign(request: TextToSignRequest): Promise<TextToSignResponse> {
    const res = await fetch(`${this.baseUrl}/text-to-sign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) throw new Error(`Text to sign failed: ${res.statusText}`);
    return res.json();
  }
}

export const apiClient = new ApiClient();
