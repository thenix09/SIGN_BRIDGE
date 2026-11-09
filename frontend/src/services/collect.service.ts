import { SampleCounts } from '../types/collect';
import { HandLandmark, Handedness } from '../types/landmarks';

/**
 * Response shape from POST /api/samples (Person A's backend).
 */
interface SampleCreateResponse {
  id: number;
  count_for_label: number;
}

/**
 * Response shape from GET /api/samples/stats (Person A's backend).
 */
interface SampleStatsResponse {
  counts: SampleCounts;
  total: number;
}

/**
 * Manage dataset sample collection.
 *
 * - Maintains in-memory counts as the local source of truth.
 * - Attempts to persist each capture to the backend via POST /api/samples.
 * - Falls back gracefully when the backend is unreachable.
 * - Loads existing counts from GET /api/samples/stats on initialisation.
 */
class CollectService {
  /** In-memory counts — always up to date even if backend is offline. */
  private localCounts: SampleCounts = {};

  /**
   * Load existing sample counts from the backend stats endpoint.
   * Returns the merged local + server counts.
   * Safe to call when the server is not running — returns empty counts.
   */
  async loadStats(): Promise<SampleCounts> {
    try {
      const res = await fetch('/api/samples/stats');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: SampleStatsResponse = await res.json();
      // Merge server counts into local counts (server wins for existing keys)
      this.localCounts = { ...this.localCounts, ...data.counts };
    } catch {
      // Backend not available — start from local (empty) counts.
    }
    return { ...this.localCounts };
  }

  /**
   * Submit a captured sample to the backend.
   *
   * Always increments the local count immediately (optimistic update).
   * Returns the backend-confirmed count for the label, or the local count
   * if the request fails.
   *
   * @param label      Single uppercase letter, e.g. "A"
   * @param handedness "Left" | "Right"
   * @param landmarks  Exactly 21 HandLandmark objects from MediaPipe
   * @returns          { count: number; savedToBackend: boolean }
   */
  async captureSample(
    label: string,
    handedness: Handedness,
    landmarks: HandLandmark[]
  ): Promise<{ count: number; savedToBackend: boolean }> {
    // Optimistic local increment
    this.localCounts[label] = (this.localCounts[label] ?? 0) + 1;
    const localCount = this.localCounts[label];

    // Convert HandLandmark[] to [[x,y,z], ...] as required by the API
    const landmarkArray = landmarks.map((lm) => [lm.x, lm.y, lm.z]);

    try {
      const res = await fetch('/api/samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          handedness,
          landmarks: landmarkArray,
          group: 'collected'
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: SampleCreateResponse = await res.json();

      // Trust the server count
      this.localCounts[label] = data.count_for_label;
      return { count: data.count_for_label, savedToBackend: true };
    } catch {
      // Backend unreachable — keep the optimistic local count
      return { count: localCount, savedToBackend: false };
    }
  }

  /** Return a snapshot of the current in-memory counts. */
  getCounts(): SampleCounts {
    return { ...this.localCounts };
  }

  /** Return the count for a single letter. */
  getCount(letter: string): number {
    return this.localCounts[letter] ?? 0;
  }
}

export const collectService = new CollectService();
