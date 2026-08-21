// Mirrors the shapes the harness entries publish (src/harness/*-entry.ts), for the
// Playwright specs (compiled separately from the app's tsconfig).
interface Window {
  __harness?: {
    ok: boolean;
    error?: string;
    running?: boolean;
    nonBlank?: boolean;
    luminanceRange?: [number, number];
  };
  __xrHarness?: {
    ok: boolean;
    error?: string;
    bound?: boolean;
    bodies?: { total: number; mocap: number; equalities: number };
    pushed?: number;
    pushedWhileUntracked?: number;
    lifted?: number;
    handLifted?: number;
    dropped?: number;
    grabbedDuringLift?: boolean;
    grabbedAfterRelease?: boolean;
  };
}
