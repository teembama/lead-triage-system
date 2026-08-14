"""Lead triage console - the operator-facing interface.

Four sections, in the order the work happens:

    01 // SOURCE          upload and validate a lead file
    02 // TARGET PROFILE  the industry, budget and optional location to assess against
    03 // QUEUE           run summary and the ranked leads
    04 // LEAD DETAIL     the full assessment for the selected lead

The run-summary tiles live at the head of section 03 rather than in a section of
their own: they describe the queue they sit above, and adding a fifth section
for them would break the four-section structure.

This file owns flow and markup only. Cleaning, assessment, scoring and routing
happen in pipeline.py and the modules beneath it; nothing about a lead's score
or route is decided here.
"""

from __future__ import annotations

import streamlit as st

import background
import ui
from data_cleaning import CANONICAL_COLUMNS, read_leads_csv
from pipeline import export_csv
from target_profile import ANY_INDUSTRY, TargetProfile, TargetProfileError

st.set_page_config(page_title="Koya · Lead Triage", page_icon="◧", layout="wide")
st.markdown(ui.stylesheet(), unsafe_allow_html=True)

# Reports which queue row was clicked. Headless - it draws nothing and owns no
# markup, so the queue table stays exactly the table it was.
queue_row_clicks = st.components.v2.component("koya_queue_rows", js=ui.QUEUE_CLICK_JS)

state = st.session_state
state.setdefault("results", None)
state.setdefault("selected", None)
state.setdefault("profile", None)
state.setdefault("run", None)          # a RunHandle while a run is in flight

results = state["results"]

# A run is "active" from the moment it starts until the script harvests it. The
# whole interface below reads this one flag: it disables the start control,
# reveals Cancel, and locks the results section.
state.setdefault("run_error", None)

run = state["run"]
processing = run is not None and not run.finished


@st.fragment(run_every="0.5s")
def _render_run_state() -> None:
    """Poll the background run and draw its progress.

    A fragment because it must repaint on a timer without re-running the whole
    script: the results section below is expensive to rebuild and would flicker
    twice a second. When the run ends this promotes the outcome into session
    state and reruns the *app*, which is what unlocks the rest of the page.

    Everything here executes on the script thread, so it may call `st.*` freely.
    The worker only ever wrote integers into the handle.
    """
    handle = state.get("run")
    if handle is None:
        return

    if not handle.finished:
        st.markdown(ui.run_banner(handle.done, handle.total), unsafe_allow_html=True)
        st.progress(
            handle.fraction,
            text=f"Processing leads… {handle.done} / {handle.total}",
        )
        return

    # Finished, cleanly or otherwise. Harvest and hand the page back.
    state["run"] = None
    state["run_error"] = None
    if handle.result is not None:
        state["results"] = handle.result
        state["profile"] = handle.profile
        state["selected"] = None
    elif handle.unavailable:
        state["run_error"] = "unavailable"
    elif handle.failed:
        state["run_error"] = "failed"
    st.rerun()

# ---- masthead & hero ------------------------------------------------------ #

if results:
    status = (
        f'<span>LEADS <b>{results["summary"]["total_processed"]}</b></span>'
        f'<span>READY TO CONTACT <b>{results["summary"]["CONTACT_NOW"]}</b></span>'
        f'<span>NEEDS REVIEW <b>{results["summary"]["REVIEW"]}</b></span>'
    )
else:
    status = "<span>NO FILE LOADED</span>"

st.markdown(ui.masthead(status), unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)


# ---- 01 // SOURCE --------------------------------------------------------- #

st.markdown(ui.section("01", "Source"), unsafe_allow_html=True)

uploaded = st.file_uploader("Lead file (CSV)", type=["csv"], label_visibility="collapsed")

raw_df, problem = None, None
if uploaded is not None:
    try:
        raw_df = read_leads_csv(uploaded)
    except ValueError as exc:
        problem = str(exc)
    except Exception:
        problem = (
            "That file could not be read as a CSV. Export it again as comma-separated "
            "values and upload it once more."
        )

SUGGESTED_INDUSTRIES = [
    ANY_INDUSTRY,
    "Marketing & Advertising", "SaaS & Software", "Healthcare", "Financial Services",
    "E-commerce & Retail", "Professional Services", "Education", "Real Estate",
    "Manufacturing", "Logistics & Supply Chain", "Hospitality",
]

INDUSTRY_KEY = "target_industries"
_PREVIOUS_INDUSTRIES = "_target_industries_previous"

# No industry restriction until the operator names one.
DEFAULT_INDUSTRIES = [ANY_INDUSTRY]

# Seeded to match the widget's own default, because the tracker below is only
# written by the on-change callback. Left unset, the very first change would be
# compared against an empty selection - and adding an industry alongside the
# default `Any` would look like `Any` was the new arrival and drop the industry
# instead.
state.setdefault(_PREVIOUS_INDUSTRIES, list(DEFAULT_INDUSTRIES))


def enforce_any_exclusivity() -> None:
    """Keep `Any` and named industries mutually exclusive, in both directions.

    Which one to drop depends on which the operator just added, so the previous
    selection is kept alongside: adding `Any` clears the named industries,
    adding a named industry clears `Any`. Neither one resets the rest of the
    selection, and the state ["Any", "Technology"] never survives a rerun.
    """
    picked = list(state.get(INDUSTRY_KEY) or [])
    previous = list(state.get(_PREVIOUS_INDUSTRIES) or [])

    if ANY_INDUSTRY in picked and len(picked) > 1:
        picked = (
            [name for name in picked if name != ANY_INDUSTRY]
            if ANY_INDUSTRY in previous else [ANY_INDUSTRY]
        )
        state[INDUSTRY_KEY] = picked

    state[_PREVIOUS_INDUSTRIES] = picked

if uploaded is None:
    st.markdown(
        '<div class="k-state empty"><h4>NO FILE LOADED</h4>'
        "<p>Upload a lead export to begin. The file is read, cleaned and assessed in place "
        "&mdash; nothing is written back to it.</p>"
        f'<div class="k-spec"><div><span>Expected columns</span><b>{len(CANONICAL_COLUMNS)}</b></div>'
        "<div><span>Format</span><b>CSV, UTF-8</b></div>"
        "<div><span>Row limit</span><b>none</b></div>"
        "<div><span>Your file</span><b>never modified</b></div></div></div>",
        unsafe_allow_html=True,
    )
elif problem:
    st.markdown(
        f'<div class="k-err"><span class="k">File cannot be read</span>'
        f"<p>{ui._e(problem)}</p>"
        f'<p>Expected columns: {ui._e(", ".join(CANONICAL_COLUMNS))}.</p></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="k-ok">'
        f'<span><span class="k">File</span><br><span class="v">{ui._e(uploaded.name)}</span></span>'
        f'<span><span class="k">Rows detected</span><br><span class="v">{len(raw_df)}</span></span>'
        f'<span><span class="k">Columns</span><br><span class="v">{len(raw_df.columns)}</span></span>'
        '<span><span class="k">Status</span><br><span class="v">Ready to process</span></span></div>',
        unsafe_allow_html=True,
    )

    # ---- 02 // TARGET PROFILE --------------------------------------------- #

    st.markdown(ui.section("02", "Target profile"), unsafe_allow_html=True)

    col_industry, col_low, col_high, col_place = st.columns([2, 1, 1, 1.4])
    with col_industry:
        st.markdown('<span class="k-lbl">Target industries</span>', unsafe_allow_html=True)
        # Selections render as removable chips, and a name that is not on the
        # list can be typed in and added - so a target can be adjusted one
        # industry at a time instead of being rebuilt from scratch.
        industries = st.multiselect(
            "Target industries", SUGGESTED_INDUSTRIES,
            default=DEFAULT_INDUSTRIES,
            key=INDUSTRY_KEY, on_change=enforce_any_exclusivity,
            accept_new_options=True,
            placeholder="Add an industry",
            label_visibility="collapsed",
        )
    with col_low:
        st.markdown('<span class="k-lbl">Budget from</span>', unsafe_allow_html=True)
        budget_min = st.number_input("Budget from", min_value=0, value=5_000, step=500,
                                     label_visibility="collapsed")
    with col_high:
        st.markdown('<span class="k-lbl">Budget to</span>', unsafe_allow_html=True)
        budget_max = st.number_input("Budget to", min_value=0, value=50_000, step=500,
                                     label_visibility="collapsed")
    with col_place:
        st.markdown('<span class="k-lbl">Location &middot; optional</span>',
                    unsafe_allow_html=True)
        location = st.text_input("Location", value="", placeholder="Any",
                                 label_visibility="collapsed")

    profile, profile_problem = None, None
    try:
        profile = TargetProfile.create(industries, budget_min, budget_max, location)
    except TargetProfileError as exc:
        profile_problem = str(exc)

    if profile_problem:
        st.markdown(
            '<div class="k-err"><span class="k">Target profile incomplete</span>'
            f"<p>{ui._e(profile_problem)}</p></div>",
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        partial = st.checkbox("Process a sample only", value=False,
                              disabled=processing)
    with col_b:
        sample_size = (
            st.number_input(
                "Leads to process", min_value=1, max_value=max(1, len(raw_df)),
                value=min(20, len(raw_df)), label_visibility="collapsed",
                disabled=processing,
            )
            if partial else None
        )

    start = st.button("Process leads", type="primary",
                      disabled=profile is None or processing, width="content")

    if start:
        # Returns immediately. The pipeline runs on its own thread so this
        # script can finish, repaint, and keep Cancel operable.
        planned = min(sample_size, len(raw_df)) if partial else len(raw_df)
        state["run"] = background.start_run(
            raw_df, profile=profile,
            limit=sample_size if partial else None, total=planned,
        )
        st.rerun()

    if state["run"] is not None:
        _render_run_state()

    if state["run_error"] == "unavailable":
        st.markdown(
            '<div class="k-err"><span class="k">Assessment unavailable</span>'
            "<p>Lead assessment is not available for this deployment, so leads cannot be "
            "scored right now. Your file was read and validated successfully.</p></div>",
            unsafe_allow_html=True,
        )
    elif state["run_error"] == "failed":
        # The exception text stays in the server log. Function names and
        # signatures are no more appropriate on screen than a 429 body.
        st.markdown(
            '<div class="k-err"><span class="k">Processing stopped</span>'
            "<p>Something went wrong while assessing these leads, so the run "
            "was stopped. Nothing was changed in your file.</p>"
            "<p>Try again, or process a smaller sample to narrow it down.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

if not results:
    # Nothing assessed yet. A first run has no previous queue to dim, so the
    # progress panel above stands alone rather than over an invented one.
    st.stop()

# Everything below is the results section, and all of it locks together while a
# run is in flight. One wrapper rather than a `disabled=` on each control: the
# stylesheet takes the whole subtree out of hit-testing, which covers the custom
# row-click component and the markup tables too - neither of which has a
# `disabled` parameter to set.
if processing:
    st.markdown(ui.results_lock_css(), unsafe_allow_html=True)

with st.container(key=ui.RESULTS_CONTAINER_KEY):

    # ---- 03 // QUEUE ------------------------------------------------------ #

    st.markdown(ui.section("03", "Queue"), unsafe_allow_html=True)
    st.markdown(ui.profile_summary(state.get("profile")), unsafe_allow_html=True)
    st.markdown(ui.metric_tiles(results["summary"]), unsafe_allow_html=True)

    footnotes = []
    if results["excluded"]:
        footnotes.append(f'{results["excluded"]} row(s) set aside as incomplete or not a lead')
    if results["failed"]:
        footnotes.append(f'{results["failed"]} lead(s) could not be assessed and need a look')
    if footnotes:
        st.markdown(
            f'<p class="k-meta" style="margin-top:12px">{ui._e(" · ".join(footnotes))}</p>',
            unsafe_allow_html=True,
        )

    rows = results["rows"]
    route_choices = ["All routes"] + [
        r for r in ("CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW")
        if results["summary"].get(r)
    ]

    col_filter, col_open = st.columns([1, 2])
    with col_filter:
        route_filter = st.selectbox("Route", route_choices, label_visibility="collapsed",
                                    disabled=processing)

    visible = rows if route_filter == "All routes" else [
        r for r in rows if r["recommendation"] == route_filter
    ]

    labels = {f'{r["company"] or r["lead_id"]} · {r["lead_id"]}': r["lead_id"] for r in visible}
    with col_open:
        picked = st.selectbox("Open lead", ["Open a lead…"] + list(labels),
                              label_visibility="collapsed", disabled=processing)
    # Only act on a *change* here. The rows below are selectable too, and a
    # selectbox that re-asserted its value on every rerun would drag the selection
    # back off whichever row was just clicked.
    if picked in labels and picked != state.get("picked_label"):
        state["selected"] = labels[picked]
    state["picked_label"] = picked

    # Mounted before the table so a click is applied to the same run that draws it:
    # the clicked row is highlighted and its detail opens without a second pass.
    clicked = queue_row_clicks(key="queue_rows", on_lead_change=lambda: None).lead
    if clicked and not processing:
        state["selected"] = clicked

    st.markdown(ui.queue_table(visible, state["selected"]), unsafe_allow_html=True)

    st.download_button(
        "Download results (CSV)", data=export_csv(rows),
        file_name="lead_triage_results.csv", mime="text/csv",
        disabled=processing,
    )

    # ---- 04 // LEAD DETAIL ------------------------------------------------ #

    selected = next((r for r in rows if str(r["lead_id"]) == str(state["selected"])), None)

    # One flex row: heading hard left, download hard right, both on the same line.
    # Columns put each side in its own block and left the button sitting off the
    # heading's line; a single horizontal container is one row by construction. The
    # rule and the space above it belong to the container, so the line still runs
    # the full width of the section rather than stopping at the heading.
    with st.container(key="k-detail-head", horizontal=True,
                      horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown(ui.section("04", "Lead detail"), unsafe_allow_html=True, width="content")
        if selected is not None:
            st.download_button(
                "Download detail",
                data=ui.lead_report(selected, state.get("profile")),
                file_name=ui.report_filename(selected),
                mime="text/plain",
                width="content",
                disabled=processing,
            )

    if selected is None:
        st.markdown(
            '<div class="k-state empty"><h4>NO LEAD SELECTED</h4>'
            "<p>Open a lead from the queue above to see its full assessment.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ui.lead_detail(selected), unsafe_allow_html=True)
