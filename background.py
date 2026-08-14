"""Off-thread execution of one pipeline run, so the interface stays alive.

The pipeline takes minutes. Called inline it blocks Streamlit's script thread
for the whole run, which means the page cannot repaint and the progress bar
cannot move - the script is busy, so nothing it drew can be updated.

So the run moves to a worker thread and the script polls this object.

## The one rule

`RunHandle` is plain Python and the worker thread touches nothing else. It does
not call `st.*`, and it does not write `st.session_state`. A thread spawned
outside Streamlit's machinery has no script context: widget calls from it are
silently discarded, and session-state writes from it are not a supported
operation. Everything the interface needs is read from this object by the
script thread, which does have a context.

That is also why progress is *stored* here rather than *rendered* here - the
existing per-lead counter in `llm_extraction` was already correct; what was
broken was trying to draw it from a worker thread.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Optional

import pandas as pd

from pipeline import run_pipeline
from providers import ProviderConfigError


class RunHandle:
    """Progress and outcome for one background run."""

    def __init__(self, total: int):
        self.total = max(0, int(total))
        self._lock = threading.Lock()
        self._done = 0
        self.thread: Optional[threading.Thread] = None

        # Outcome, written by the worker and read by the script thread once
        # `finished` is true. Exactly one of these is set.
        self.result: Optional[dict[str, Any]] = None
        self.unavailable = False        # provider not configured for this deployment
        self.failed = False             # anything else; detail goes to the log
        self.profile: Any = None

    # -- progress ----------------------------------------------------------- #

    @property
    def done(self) -> int:
        with self._lock:
            return self._done

    def tick(self, done: int, total: int) -> None:
        """Progress callback for the pipeline. Runs on worker threads, so it
        only ever assigns an integer under a lock."""
        with self._lock:
            self._done = int(done)

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total) if self.total else 0.0

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def started(self) -> bool:
        return self.thread is not None

    @property
    def finished(self) -> bool:
        thread = self.thread
        return thread is not None and not thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)


def start_run(
    raw_df: pd.DataFrame,
    *,
    profile: Any = None,
    limit: Optional[int] = None,
    total: Optional[int] = None,
    runner: Optional[Callable[..., dict[str, Any]]] = None,
) -> RunHandle:
    """Begin a run on a daemon thread and return its handle immediately.

    `runner` is injectable so the flow can be exercised without a provider; the
    app always leaves it as the real pipeline.
    """
    execute = runner or run_pipeline
    planned = total if total is not None else (limit or len(raw_df))
    handle = RunHandle(planned)
    handle.profile = profile

    def work() -> None:
        try:
            handle.result = execute(
                raw_df, limit=limit, profile=profile, progress=handle.tick,
            )
        except ProviderConfigError:
            handle.unavailable = True
        except Exception:                    # noqa: BLE001 - the run ends, the app does not
            # Vendor and Python detail belong in the log, never on screen.
            traceback.print_exc()
            handle.failed = True

    # Daemon: a closed browser tab or a restarted server must not leave a run
    # holding the rate limiter open behind it.
    handle.thread = threading.Thread(target=work, name="koya-run", daemon=True)
    handle.thread.start()
    return handle
