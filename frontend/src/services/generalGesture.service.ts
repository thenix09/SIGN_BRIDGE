import { DetectedHandData, DetectedPoseData, HandLandmark } from '../types/landmarks';

export interface GeneralGestureResult {
  word: string;
  confidence: number;
  description: string;
  source: 'word_sign';
  category: string;
}

interface FrameSnapshot {
  timestamp: number;
  wristX: number;
  wristY: number;
  wristZ: number;
  indexTipY: number;
  isFist: boolean;
  isOpen: boolean;
}

// MediaPipe Landmark Index Constants
const WRIST = 0;
const THUMB_MCP = 2;
const THUMB_TIP = 4;

const INDEX_MCP = 5;
const INDEX_PIP = 6;
const INDEX_TIP = 8;

const MIDDLE_PIP = 10;
const MIDDLE_TIP = 12;

const RING_PIP = 14;
const RING_TIP = 16;

const PINKY_MCP = 17;
const PINKY_PIP = 18;
const PINKY_TIP = 20;

function dist(p1: HandLandmark, p2: HandLandmark): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export class GeneralGestureService {
  private history: FrameSnapshot[] = [];
  private maxHistoryLength = 30; // ~1 second at 30 fps
  private lastConfirmedGesture: string | null = null;
  private lastConfirmedTime = 0;
  private cooldownMs = 1200; // Cooldown before same word can confirm again

  /**
   * Evaluate detected hand(s) and upper-body pose for standard ASL whole-word signs.
   */
  public evaluate(
    hands: DetectedHandData[],
    pose: DetectedPoseData | null
  ): GeneralGestureResult | null {
    if (!hands || hands.length === 0) {
      this.history = [];
      return null;
    }

    const primaryHand = hands[0];
    const lm = primaryHand.landmarks;
    if (!lm || lm.length < 21) return null;

    const wrist = lm[WRIST];
    const now = Date.now();

    // Finger extensions
    const isIndexExtended = dist(lm[INDEX_TIP], wrist) > dist(lm[INDEX_PIP], wrist) * 1.25;
    const isMiddleExtended = dist(lm[MIDDLE_TIP], wrist) > dist(lm[MIDDLE_PIP], wrist) * 1.25;
    const isRingExtended = dist(lm[RING_TIP], wrist) > dist(lm[RING_PIP], wrist) * 1.25;
    const isPinkyExtended = dist(lm[PINKY_TIP], wrist) > dist(lm[PINKY_PIP], wrist) * 1.25;
    const isThumbExtended = dist(lm[THUMB_TIP], lm[PINKY_MCP]) > dist(lm[THUMB_MCP], lm[PINKY_MCP]) * 1.25;

    const openFingersCount =
      (isIndexExtended ? 1 : 0) +
      (isMiddleExtended ? 1 : 0) +
      (isRingExtended ? 1 : 0) +
      (isPinkyExtended ? 1 : 0);

    const isOpenHand = openFingersCount >= 3 && isIndexExtended && isMiddleExtended;
    const isFist = openFingersCount === 0;
    const isWHand = isIndexExtended && isMiddleExtended && isRingExtended && !isPinkyExtended;
    const isILYHand = isThumbExtended && isIndexExtended && !isMiddleExtended && !isRingExtended && isPinkyExtended;

    // Record motion snapshot in rolling history
    this.history.push({
      timestamp: now,
      wristX: wrist.x,
      wristY: wrist.y,
      wristZ: wrist.z || 0,
      indexTipY: lm[INDEX_TIP].y,
      isFist,
      isOpen: isOpenHand
    });

    if (this.history.length > this.maxHistoryLength) {
      this.history.shift();
    }

    // Pose landmarks / reference anchors
    let noseX = 0.50;
    let mouthY = 0.38;
    let chestY = 0.55;

    if (pose && pose.landmarks && pose.landmarks.length > 12) {
      const nose = pose.landmarks[0];
      const leftShoulder = pose.landmarks[11];
      const rightShoulder = pose.landmarks[12];
      if (nose) {
        noseX = nose.x;
        mouthY = nose.y + 0.08;
      }
      if (leftShoulder && rightShoulder) {
        chestY = (leftShoulder.y + rightShoulder.y) / 2 + 0.05;
      }
    }

    // -------------------------------------------------------------
    // 1. I LOVE YOU (ILY Sign) - Static, highly unique
    // -------------------------------------------------------------
    if (isILYHand) {
      const spread = dist(lm[INDEX_TIP], lm[PINKY_TIP]);
      const palmWidth = dist(lm[INDEX_MCP], lm[PINKY_MCP]);
      if (spread > palmWidth * 1.4) {
        return {
          word: 'I LOVE YOU',
          confidence: 0.98,
          description: 'Universal ASL "I Love You" sign (thumb, index, pinky)',
          source: 'word_sign',
          category: 'feelings'
        };
      }
    }

    // -------------------------------------------------------------
    // 2. HELLO (Salute at forehead/temple with open flat hand)
    // -------------------------------------------------------------
    if (isOpenHand) {
      const handNearForehead = lm[INDEX_TIP].y <= mouthY - 0.03 || wrist.y < chestY - 0.15;
      const nearTempleX = Math.abs(wrist.x - noseX) < 0.45;
      const fingersPointingUp = lm[INDEX_TIP].y < wrist.y;

      if (handNearForehead && nearTempleX && fingersPointingUp) {
        return {
          word: 'HELLO',
          confidence: 0.96,
          description: 'Flat open hand salute near forehead/temple',
          source: 'word_sign',
          category: 'greetings'
        };
      }
    }

    // -------------------------------------------------------------
    // 3. BYE (Waving open hand side-to-side)
    // -------------------------------------------------------------
    if (isOpenHand && wrist.y < chestY + 0.15) {
      // Analyze horizontal oscillation across history
      if (this.history.length >= 8) {
        let directionChanges = 0;
        let lastDx = 0;
        let totalTravelX = 0;

        for (let i = 1; i < this.history.length; i++) {
          const dx = this.history[i].wristX - this.history[i - 1].wristX;
          totalTravelX += Math.abs(dx);
          if (Math.abs(dx) > 0.005) {
            if (lastDx !== 0 && ((dx > 0 && lastDx < 0) || (dx < 0 && lastDx > 0))) {
              directionChanges++;
            }
            lastDx = dx;
          }
        }

        // At least 2 direction reversals and noticeable horizontal motion
        if (directionChanges >= 2 && totalTravelX > 0.04) {
          return {
            word: 'BYE',
            confidence: 0.95,
            description: 'Open hand waving side-to-side',
            source: 'word_sign',
            category: 'greetings'
          };
        }
      }
    }

    // -------------------------------------------------------------
    // 4. THANK YOU (Flat hand near chin/mouth or moving outward)
    // -------------------------------------------------------------
    if (isOpenHand) {
      const tipsNearChin = Math.abs(lm[INDEX_TIP].y - mouthY) < 0.12 && Math.abs(lm[INDEX_TIP].x - noseX) < 0.25;
      const handNearMouth = Math.abs(wrist.y - mouthY) < 0.18;

      if (tipsNearChin || (handNearMouth && lm[INDEX_TIP].y < wrist.y)) {
        return {
          word: 'THANK YOU',
          confidence: 0.94,
          description: 'Flat fingertips touching or placed at chin/lips',
          source: 'word_sign',
          category: 'greetings'
        };
      }
    }

    // -------------------------------------------------------------
    // 5. YES (Fist nodding up and down)
    // -------------------------------------------------------------
    if (isFist && wrist.y < chestY + 0.2) {
      if (this.history.length >= 10) {
        let directionChangesY = 0;
        let lastDy = 0;
        let totalTravelY = 0;

        for (let i = 1; i < this.history.length; i++) {
          const dy = this.history[i].wristY - this.history[i - 1].wristY;
          totalTravelY += Math.abs(dy);
          if (Math.abs(dy) > 0.005) {
            if (lastDy !== 0 && ((dy > 0 && lastDy < 0) || (dy < 0 && lastDy > 0))) {
              directionChangesY++;
            }
            lastDy = dy;
          }
        }

        if (directionChangesY >= 2 && totalTravelY > 0.035) {
          return {
            word: 'YES',
            confidence: 0.93,
            description: 'Closed fist nodding up and down',
            source: 'word_sign',
            category: 'greetings'
          };
        }
      }
    }

    // -------------------------------------------------------------
    // 6. NO (Index + Middle fingers closing/snapping to thumb)
    // -------------------------------------------------------------
    if (isIndexExtended && isMiddleExtended && !isRingExtended && !isPinkyExtended) {
      const pinchDist = dist(lm[INDEX_TIP], lm[THUMB_TIP]);
      if (pinchDist < 0.065 && wrist.y < chestY + 0.1) {
        return {
          word: 'NO',
          confidence: 0.92,
          description: 'Index and middle fingers pinched shut against thumb',
          source: 'word_sign',
          category: 'greetings'
        };
      }
    }

    // -------------------------------------------------------------
    // 7. WATER ("W" handshape tapped near chin)
    // -------------------------------------------------------------
    if (isWHand) {
      const nearChin = Math.abs(lm[INDEX_TIP].y - mouthY) < 0.15 && Math.abs(lm[INDEX_TIP].x - noseX) < 0.25;
      if (nearChin) {
        return {
          word: 'WATER',
          confidence: 0.94,
          description: '"W" three fingers placed near chin',
          source: 'word_sign',
          category: 'food and drink'
        };
      }
    }

    // -------------------------------------------------------------
    // 8. PLEASE (Flat open hand rubbing chest in circle or on chest)
    // -------------------------------------------------------------
    if (isOpenHand) {
      const onChest = Math.abs(wrist.y - chestY) < 0.14 && Math.abs(wrist.x - noseX) < 0.25;
      if (onChest) {
        return {
          word: 'PLEASE',
          confidence: 0.91,
          description: 'Flat open palm placed on chest',
          source: 'word_sign',
          category: 'greetings'
        };
      }
    }

    // -------------------------------------------------------------
    // 9. BOOK (Two flat hands together like an open book)
    // -------------------------------------------------------------
    if (hands.length >= 2) {
      const h2 = hands[1];
      if (h2.landmarks && h2.landmarks.length >= 21) {
        const w2 = h2.landmarks[WRIST];
        const wristDist = Math.hypot(wrist.x - w2.x, wrist.y - w2.y);
        const isOpen2 = dist(h2.landmarks[INDEX_TIP], w2) > dist(h2.landmarks[INDEX_PIP], w2) * 1.25;

        if (isOpenHand && isOpen2 && wristDist < 0.22 && wrist.y < chestY + 0.15) {
          return {
            word: 'BOOK',
            confidence: 0.93,
            description: 'Two open palms touching pinky-to-pinky like an open book',
            source: 'word_sign',
            category: 'objects'
          };
        }
      }
    }

    return null;
  }

  /**
   * Check if a word can be committed, respecting cooldown.
   */
  public canCommit(word: string): boolean {
    const now = Date.now();
    if (this.lastConfirmedGesture === word && now - this.lastConfirmedTime < this.cooldownMs) {
      return false;
    }
    return true;
  }

  public recordCommitted(word: string): void {
    this.lastConfirmedGesture = word;
    this.lastConfirmedTime = Date.now();
  }

  public reset(): void {
    this.history = [];
    this.lastConfirmedGesture = null;
  }
}

export const generalGestureService = new GeneralGestureService();
