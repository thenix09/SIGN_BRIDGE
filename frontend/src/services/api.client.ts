import { HealthResponse, VocabularyResponse, GesturePrediction } from '../types/api';

/**
 * SignBridge API Client (Stub Implementation for Day 1-3).
 * Keeps API communication strictly decoupled from UI components.
 * Will be swapped with actual backend endpoints once Person A releases the frozen API on Day 4.
 */
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
  }

  /**
   * Get the current API base URL.
   */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Health check endpoint stub.
   */
  async checkHealth(): Promise<HealthResponse> {
    return {
      status: 'ok',
      version: '0.1.0-stub',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Vocabulary stub (ASL letters and baseline words).
   */
  async getVocabulary(): Promise<VocabularyResponse> {
    return {
      supported_letters: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'O', 'V', 'W', 'Y'],
      supported_words: ['hello', 'thank you', 'please', 'yes', 'no', 'help', 'goodbye'],
      languages: ['ASL']
    };
  }

  /**
   * Mock prediction for testing frontend layout pipeline.
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
