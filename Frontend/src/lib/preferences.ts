/**
 * Operator preferences: local, persisted, and read by the data layer.
 *
 * These are display and pacing choices only. Nothing here can change a model, a
 * threshold, or what the pipeline computes -- the refresh interval decides how
 * often the browser re-reads results that already exist, and the rest is
 * presentation. That boundary is deliberate: the Settings screen is reachable
 * by anyone with the dashboard open.
 *
 * Implemented as a tiny external store rather than React context so that
 * `hooks.ts` can read the current refresh interval from module scope, where
 * React Query evaluates `refetchInterval`, while components still re-render on
 * change through `useSyncExternalStore`.
 */
import { useSyncExternalStore } from "react";

export interface Preferences {
  /** Seconds between background refetches. 0 turns polling off entirely. */
  refreshSeconds: number;
  /** Route to open when the app is launched at "/". */
  startPage: string;
  /** Suppress entrance/transition animation for motion sensitivity or demos. */
  reduceMotion: boolean;
}

export const DEFAULT_PREFERENCES: Preferences = {
  refreshSeconds: 15,
  startPage: "/",
  reduceMotion: false,
};

/** Offered in the Settings screen; the value is the route. */
export const START_PAGES: Array<{ value: string; label: string }> = [
  { value: "/", label: "Executive Overview" },
  { value: "/health", label: "Health Monitoring" },
  { value: "/anomaly", label: "Anomaly Detection" },
  { value: "/rul", label: "RUL Prediction" },
  { value: "/failure-risk", label: "Failure Risk" },
  { value: "/maintenance", label: "Maintenance Supervisor" },
  { value: "/upload", label: "Upload & Run" },
];

/** 0 is a real choice (polling off), so it is offered explicitly. */
export const REFRESH_CHOICES = [0, 5, 15, 30, 60, 300];

const STORAGE_KEY = "aerotwin.preferences";

let current: Preferences = load();
const listeners = new Set<() => void>();

function load(): Preferences {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PREFERENCES;
    const parsed = JSON.parse(raw) as Partial<Preferences>;
    return {
      // Merged onto the defaults so a preferences blob written by an older
      // build never leaves a field undefined.
      refreshSeconds:
        typeof parsed.refreshSeconds === "number" && parsed.refreshSeconds >= 0
          ? parsed.refreshSeconds
          : DEFAULT_PREFERENCES.refreshSeconds,
      startPage:
        typeof parsed.startPage === "string" && parsed.startPage
          ? parsed.startPage
          : DEFAULT_PREFERENCES.startPage,
      reduceMotion: Boolean(parsed.reduceMotion),
    };
  } catch {
    // Private mode, blocked site data, or malformed JSON -- defaults are fine.
    return DEFAULT_PREFERENCES;
  }
}

export function getPreferences(): Preferences {
  return current;
}

/**
 * Poll interval in milliseconds for React Query, or `false` to stop polling.
 *
 * Passed as a function to `refetchInterval` so a change takes effect on the
 * next tick without every query having to be re-created.
 */
export function refetchIntervalMs(): number | false {
  return current.refreshSeconds > 0 ? current.refreshSeconds * 1000 : false;
}

export function setPreferences(patch: Partial<Preferences>) {
  current = { ...current, ...patch };
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
  } catch {
    /* preference only -- the session still honours it in memory */
  }
  listeners.forEach((l) => l());
}

export function resetPreferences() {
  current = DEFAULT_PREFERENCES;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* nothing to clean up */
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function usePreferences(): Preferences {
  return useSyncExternalStore(subscribe, getPreferences, () => DEFAULT_PREFERENCES);
}
