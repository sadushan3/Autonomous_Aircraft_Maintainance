import { useEffect, useRef } from "react";
import { Navigate, Outlet, useLocation } from "react-router";
import { MotionConfig } from "motion/react";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { usePreferences } from "../../lib/preferences";

export function Root() {
  const prefs = usePreferences();
  const location = useLocation();

  // The start-page preference applies to a cold open only. Re-applying it on
  // every visit to "/" would make the Overview link in the sidebar unusable for
  // anyone whose start page is something else.
  const landed = useRef(false);
  const shouldRedirect =
    !landed.current && location.pathname === "/" && prefs.startPage !== "/";
  useEffect(() => {
    landed.current = true;
  }, []);

  return (
    // `reducedMotion="user"` still honours the OS setting; "always" is the
    // explicit opt-out for motion sensitivity or a projector-friendly demo.
    <MotionConfig reducedMotion={prefs.reduceMotion ? "always" : "user"}>
      <div className="flex h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
        <Sidebar />
        <div className="flex flex-1 flex-col overflow-hidden">
          <Topbar />
          <main className="flex-1 overflow-auto p-6">
            {shouldRedirect ? <Navigate to={prefs.startPage} replace /> : <Outlet />}
          </main>
        </div>
      </div>
    </MotionConfig>
  );
}
