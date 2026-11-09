/**
 * Types for the Dataset Collection panel.
 */

/** Per-letter sample counts. Key = uppercase letter A-Z. */
export interface SampleCounts {
  [letter: string]: number;
}

/** UI feedback state after a capture attempt. */
export type CollectStatus = 'idle' | 'captured' | 'no_hand' | 'error';

/** Target number of samples per letter for the progress bar. */
export const SAMPLES_TARGET = 50;

/** All 26 uppercase ASL letters. */
export const ALL_LETTERS: string[] = Array.from({ length: 26 }, (_, i) =>
  String.fromCharCode(65 + i)
);
