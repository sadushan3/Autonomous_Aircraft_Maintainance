/**
 * The engine under investigation, carried in the URL as `?engine=27`.
 *
 * Each screen used to hold its own engine in local state, so moving from Health
 * Monitoring to RUL to Failure Risk meant re-finding the same unit three times,
 * and the topbar's engine search could only ever drop you on one fixed page.
 * Putting the selection in the query string makes it survive navigation, the
 * back button, a refresh, and being pasted to someone else -- which is what
 * turns six dashboards into one investigation.
 *
 * The URL is the single source of truth rather than a mirrored copy in state:
 * two stores for one value is how they drift apart.
 */
import { useCallback } from "react";
import { useSearchParams } from "react-router";

export const ENGINE_PARAM = "engine";

/** `[engine, setEngine]`, shaped like `useState` so pages can swap it in directly. */
export function useEngineParam(): [string | null, (id: string | number | null) => void] {
  const [params, setParams] = useSearchParams();
  const engine = params.get(ENGINE_PARAM);

  const setEngine = useCallback(
    (id: string | number | null) => {
      const next = new URLSearchParams(params);
      if (id === null || id === "") next.delete(ENGINE_PARAM);
      else next.set(ENGINE_PARAM, String(id));
      // `replace` so stepping through engines does not bury the previous page
      // under a dozen history entries.
      setParams(next, { replace: true });
    },
    [params, setParams],
  );

  return [engine, setEngine];
}

/** Build a link that opens `path` already focused on `engine`. */
export function engineLink(path: string, engine: string | number | null | undefined) {
  return engine === null || engine === undefined || engine === ""
    ? path
    : `${path}?${ENGINE_PARAM}=${encodeURIComponent(String(engine))}`;
}
