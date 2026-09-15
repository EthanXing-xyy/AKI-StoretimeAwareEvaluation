"""Run the frozen continuous-monitoring pipeline under one lab-availability definition.

From the repository root:
    python Step4_Storetime_Aware_Stress_Test/scripts/run_stress.py monitor --availability charttime --horizons 0 --models masked_cnn
    python Step4_Storetime_Aware_Stress_Test/scripts/run_stress.py downstream --availability storetime

`monitor` calls the Step 2 run_monitoring.py entry point (one checkpoint per process, as
in phase two) with the st_patch availability slicing. `downstream` calls the unchanged CM merge,
upstream-analysis and figure/summary scripts with outputs redirected.
Any availability other than the charttime null control refuses to start unless
charttime/summary/null_control_check.json reports all five horizon-0 architectures passed.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_patch as st  # noqa: E402

NULL_CONTROL = st.STRESS_ROOT / "charttime" / "summary" / "null_control_check.json"


def require_null_control() -> None:
    if not NULL_CONTROL.exists():
        raise SystemExit(f"null control not run: {NULL_CONTROL} missing")
    record = json.loads(NULL_CONTROL.read_text(encoding="utf-8"))
    required = {f"h0_{m}" for m in st.cm.MODELS}
    passed = {row["run_tag"] for row in record["checkpoints"] if row["passed"]}
    if not record.get("all_passed") or not required <= passed:
        raise SystemExit(f"null control has not passed for all five horizon-0 architectures: missing {sorted(required - passed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["monitor", "downstream"])
    parser.add_argument("--availability", choices=st.AVAILABILITY, required=True)
    parser.add_argument("--horizons", nargs=1, type=int, choices=st.cm.HORIZONS)
    parser.add_argument("--models", nargs=1, choices=st.cm.MODELS)
    args = parser.parse_args()
    if args.availability != "charttime":
        require_null_control()
    st.activate(args.availability)

    if args.command == "monitor":
        if not (args.horizons and args.models):
            parser.error("monitor runs exactly one checkpoint: give --horizons H --models M")
        run_tag = f"h{args.horizons[0]}_{args.models[0]}"
        sys.argv = ["run_monitoring.py", "--stage", "full", "--horizons", str(args.horizons[0]), "--models", args.models[0]]
        try:
            st.run_monitoring.main()
        except st.cm.MonitoringAbort as error:
            st.LOGGER.error("ABORTED: %s", error)
            raise SystemExit(2)
        st.write_run_records(run_tag)
        return

    import make_figures_and_summary
    import merge_monitoring_runs
    import run_upstream_analyses

    for module in (merge_monitoring_runs, run_upstream_analyses, make_figures_and_summary):
        if module.cm is not st.cm:
            raise SystemExit(f"{module.__name__} imported a different cm_adapter module")
        module.main()


if __name__ == "__main__":
    main()
