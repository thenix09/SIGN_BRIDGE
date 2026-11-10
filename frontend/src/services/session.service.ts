const sessionIds = new Map<string, string>();

/**
 * Return a stable session ID for a feature during this page lifetime.
 */
export function getClientSessionId(feature: string): string {
  const cached = sessionIds.get(feature);
  if (cached) return cached;

  const id =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const sessionId = `${feature}-${id}`;
  sessionIds.set(feature, sessionId);
  return sessionId;
}
