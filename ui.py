"""Presentation layer for the lead triage console.

Holds the approved visual system (palette, type scale, geometry) and the HTML
renderers for each section. Kept separate from app.py so the flow logic reads
as flow logic rather than as markup.

Two rules govern the copy in here:

* Nothing user-facing names the implementation. No language, framework, model
  or provider appears in the interface - the reader is evaluating leads, not
  the stack that scored them.
* Internal vocabulary is translated at this boundary. The pipeline's "CRM
  field" is "the original lead data" on screen.
"""

from __future__ import annotations

import base64
import html
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

FONT_DIR = Path(__file__).parent / "assets" / "fonts"

# Route -> (marker modifier, display label). The marker shape carries the
# meaning; colour only reinforces it, so the set survives greyscale.
ROUTE_META = {
    "CONTACT_NOW": ("contact", "Contact now"),
    "NURTURE": ("nurture", "Nurture"),
    "DISQUALIFY": ("disq", "Disqualify"),
    "REVIEW": ("review", "Review"),
}

FACTOR_LABELS = {
    "industry_fit": "Industry fit",
    "company_size_signal": "Company size",
    "seniority": "Seniority",
    "budget_capacity": "Budget",
    "source": "Source",
    "urgency": "Urgency",
    "pain_severity": "Pain severity",
    "purchasing_readiness": "Purchasing readiness",
    "timeline": "Timeline",
    "buying_stage": "Buying stage",
}

FIT_FACTORS = ("industry_fit", "company_size_signal", "seniority", "budget_capacity", "source")
INTENT_FACTORS = ("urgency", "pain_severity", "purchasing_readiness", "timeline", "buying_stage")

# The stated-purpose tokens, translated at this boundary. The token itself is
# internal vocabulary and never reaches the screen or the downloaded report -
# what a reader gets is the sentence, followed by the lead's own words.
NON_BUYER_PHRASES = {
    "competitive_research": "they were comparing our pricing to benchmark their own offering",
    "job_seeking": "they were looking for work rather than for a service",
    "academic_research": "they were researching for academic work",
    "media_enquiry": "they were making a press enquiry",
    "recruiting": "they were hiring, rather than buying",
    "vendor_pitch": "they were selling their own product or service to us",
    "investor_intro": "they were making an introduction rather than buying",
    "free_resource_only": "they wanted a free resource and nothing further",
    "spam": "the enquiry is not a genuine one",
}


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _font_face_css() -> str:
    """Inline the vendored typefaces as data URIs.

    Embedded rather than linked so the interface renders identically wherever
    it is hosted - a blocked or slow font CDN would silently swap the display
    face for a system fallback and quietly undo the design.
    """
    faces = [
        ("Geist Pixel", "GeistPixel-Grid.woff2", "400"),
        ("Geist", "Geist-Variable.woff2", "100 900"),
        ("Geist Mono", "GeistMono-Variable.woff2", "100 900"),
    ]
    out = []
    for family, filename, weight in faces:
        path = FONT_DIR / filename
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        out.append(
            f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
            f"font-display:swap;src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #

def stylesheet() -> str:
    return f"""<style>
{_font_face_css()}

/* Dark palette. Only these token values changed in the light -> dark
   conversion; every component below reads through them, so nothing else moved.
   --surface-2, --rule-strong and --accent-soft are not new hues: they are the
   raised-surface, stronger-border and selected-row tints the existing
   components already consumed, restated for a dark ground. */
:root{{
  --canvas:#0C0C0B; --surface:#131312; --surface-2:#1A1A18;
  --ink:#F5F5F3; --ink-2:#A8A29E; --ink-3:#6B6660;
  --rule:#2A2A27; --rule-strong:#3D3C38;
  --accent:#FF5A16; --accent-soft:#2A150C;
  --mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"Geist",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --pixel:"Geist Pixel","Geist Mono",monospace;
}}

/* Strip the host chrome so the console owns the full frame. */
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
.stApp {{ background:var(--canvas) !important; }}
.block-container {{ padding:0 2rem 5rem !important; max-width:1280px !important; }}

/* The host ships its own palette and picks light or dark from the viewer's OS.
   The theme is pinned in config.toml, and these declarations make the console
   independent of it regardless: every element below states its own colour and
   family rather than inheriting one this file does not control. Inheriting was
   the bug - in dark mode the host's near-white text landed on our light ground
   and vanished. */
html, body, .stApp, .stMarkdown, [data-testid="stMarkdownContainer"] {{
  font-family:var(--sans) !important; color:var(--ink) !important;
}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp span,
.stApp div, .stApp td, .stApp th, .stApp b, .stApp i, .stApp label {{ color:inherit; }}

/* ---- masthead ---- */
.k-head{{display:flex;align-items:center;justify-content:space-between;gap:16px;
        padding:18px 0;border-bottom:1px solid var(--rule-strong);flex-wrap:wrap}}
.k-brand{{display:flex;align-items:baseline;gap:12px}}
.k-brand b{{font-family:var(--mono);font-size:13px;font-weight:600;letter-spacing:.14em;
           text-transform:uppercase;color:var(--ink)}}
.k-brand span{{font-family:var(--mono);font-size:11px;letter-spacing:.12em;
              text-transform:uppercase;color:var(--ink-3)}}
.k-tag{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
       color:var(--accent);border:1px solid var(--accent);padding:2px 7px}}
.k-headmeta{{display:flex;gap:20px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;
            letter-spacing:.04em;color:var(--ink-3)}}
.k-headmeta b{{font-weight:500;color:var(--ink-2)}}

/* ---- hero ---- */
.k-hero{{padding:56px 0 44px}}
.k-hero h1{{font-family:var(--pixel);font-weight:400;font-size:clamp(36px,7vw,78px);
           line-height:.94;letter-spacing:.01em;margin:0 0 20px;text-wrap:balance;
           color:var(--ink)}}
.k-hero p{{max-width:62ch;color:var(--ink-2) !important;font-size:15px;margin:0;
          line-height:1.6;font-family:var(--sans)}}
.k-eyebrow{{display:flex;align-items:center;gap:12px;margin-bottom:20px}}
.k-eyebrow .d{{flex:1;height:1px;background:var(--rule);max-width:120px}}

/* ---- sections ---- */
.k-sec{{display:flex;align-items:baseline;gap:12px;margin:0 0 20px;padding-top:40px;
       border-top:1px solid var(--rule);flex-wrap:wrap}}
.k-sec .n{{font-family:var(--pixel);font-size:20px;color:var(--accent);line-height:1}}
.k-sec .t{{font-family:var(--mono);font-size:13px;font-weight:600;letter-spacing:.14em;
          text-transform:uppercase;color:var(--ink)}}
/* 04's header shares its row with the download, so the rule and the space above
   it are drawn by the container and suppressed on the heading itself -
   otherwise the line would stop where the heading stops, and the heading would
   carry padding the button does not, dropping the two off each other's line.
   Zero margin below: the row sits directly on top of the detail card. */
.st-key-k-detail-head{{border-top:1px solid var(--rule);padding-top:40px;margin-bottom:0}}
.st-key-k-detail-head .k-sec{{border-top:0;padding-top:0;margin:0;flex-wrap:nowrap}}
.st-key-k-detail-head [data-testid="stElementContainer"]{{margin:0}}
.k-lbl{{font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.1em;
       text-transform:uppercase;color:var(--ink-2)}}
.k-meta{{font-family:var(--mono);font-size:11px;letter-spacing:.04em;color:var(--ink-3)}}

/* ---- metric tiles ---- */
/* auto-fit rather than a fixed 5 columns: when the viewport forces a wrap, the
   empty track collapses instead of leaving an orphaned cell where the rule
   colour shows through as a stray grey block. */
.k-tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
         border:1px solid var(--rule-strong);background:var(--rule);gap:1px}}
.k-tile{{background:var(--surface);padding:22px 16px;display:flex;flex-direction:column;gap:8px}}
.k-tile.total{{background:var(--surface-2)}}
.k-tile .k{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
           color:var(--ink-2);display:flex;align-items:center;gap:7px}}
.k-tile .v{{font-family:var(--mono);font-size:34px;font-weight:500;line-height:1;
           font-variant-numeric:tabular-nums;color:var(--ink)}}

/* ---- route markers ---- */
.k-mk{{width:9px;height:9px;flex:0 0 9px;display:inline-block;border:1.5px solid var(--ink)}}
.k-mk.contact{{background:var(--accent);border-color:var(--accent)}}
.k-mk.nurture{{background:linear-gradient(90deg,var(--ink) 0 50%,transparent 50%);border-color:var(--ink)}}
.k-mk.disq{{background:transparent;border-color:var(--ink-3)}}
.k-mk.review{{background:transparent;border-style:dashed}}
.k-route{{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11px;
         font-weight:500;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}}
.k-route.contact{{color:var(--accent)}}
.k-route.nurture{{color:var(--ink)}}
.k-route.disq{{color:var(--ink-3)}}
.k-route.review{{color:var(--ink);text-decoration:underline;text-decoration-style:dashed;
                text-underline-offset:4px;text-decoration-thickness:1px}}

/* ---- queue ---- */
.k-tablewrap{{overflow-x:auto;border:1px solid var(--rule-strong);background:var(--surface)}}
.k-table{{border-collapse:collapse;width:100%;min-width:820px}}
.k-table thead th{{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.1em;
                  text-transform:uppercase;color:var(--ink-2);text-align:left;padding:12px 16px;
                  border-bottom:1.5px solid var(--ink);white-space:nowrap}}
.k-table td{{padding:11px 16px;border-bottom:1px solid var(--rule);font-family:var(--mono);
            font-size:13px;font-variant-numeric:tabular-nums;vertical-align:middle;
            color:var(--ink)}}
.k-table tbody tr:last-child td{{border-bottom:0}}
.k-table tbody tr{{transition:background .12s ease}}
.k-table tbody tr[data-lead]{{cursor:pointer}}
.k-table tbody tr[data-lead]:focus-visible{{outline:2px solid var(--accent);outline-offset:-2px}}
.k-table tbody tr:hover{{background:var(--surface-2)}}
.k-table tbody tr.sel td{{background:var(--accent-soft)}}
.k-table tbody tr.sel td:first-child{{box-shadow:inset 2px 0 0 var(--accent)}}
.k-table .num{{text-align:right}}
.k-table .rk{{color:var(--ink-3);width:1%}}
.k-table .co{{font-family:var(--sans);font-size:13px}}
.k-table .co b{{font-weight:600}}
.k-table .co i{{font-style:normal;color:var(--ink-3);display:block;font-size:11px;
               font-family:var(--mono)}}
.k-dash{{color:var(--ink-3)}}

/* ---- detail ---- */
.k-panel{{border:1px solid var(--rule-strong);background:var(--surface)}}
.k-phead{{padding:24px;border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;
         gap:24px;flex-wrap:wrap;align-items:flex-start}}
/* padding:0 is load-bearing: headings inside a markdown block are given 12px
   top and 16px bottom padding by default, which opened 28px of dead space
   between the lead's name and everything around it. */
.k-phead h3{{margin:0 0 8px;padding:0;font-family:var(--sans);font-size:22px;font-weight:600;
            letter-spacing:-.01em;line-height:1.2;color:var(--ink)}}
.k-id{{font-family:var(--mono);font-size:11px;letter-spacing:.08em;color:var(--ink-3);
      text-transform:uppercase;margin-bottom:4px}}
.k-scores{{display:flex;gap:32px;align-items:flex-start}}
.k-sc{{display:flex;flex-direction:column;gap:2px;text-align:right}}
.k-sc .k{{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
         color:var(--ink-2)}}
.k-sc .v{{font-family:var(--mono);font-size:30px;font-weight:500;line-height:1;
         font-variant-numeric:tabular-nums;color:var(--ink)}}
.k-sc .d{{font-family:var(--mono);font-size:11px;color:var(--ink-3)}}
.k-sc.hero .v{{color:var(--accent);font-size:40px}}
.k-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
         border-bottom:1px solid var(--rule);background:var(--rule)}}
.k-strip div{{background:var(--surface);padding:12px 16px}}
.k-strip .k{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;
            color:var(--ink-3)}}
.k-strip .v{{font-family:var(--mono);font-size:12px;margin-top:3px;word-break:break-word;
            color:var(--ink)}}
.k-profile{{border:1px solid var(--rule-strong);border-bottom:1px solid var(--rule-strong);
          margin-bottom:8px}}
.k-block{{padding:24px;border-bottom:1px solid var(--rule)}}
.k-block:last-child{{border-bottom:0}}
.k-block > .k-lbl{{display:block;margin-bottom:12px}}
.k-expl p{{font-family:var(--sans);font-size:14px;line-height:1.65;margin:0 0 12px;
          max-width:74ch;color:var(--ink) !important}}
.k-expl p:last-child{{margin-bottom:0}}
.k-expl p.verdict{{font-family:var(--mono);font-size:13px;letter-spacing:.02em;font-weight:500}}
.k-notes{{font-family:var(--sans);font-size:13px;line-height:1.7;color:var(--ink-2) !important;margin:0;
         padding:16px;background:var(--surface-2);border-left:2px solid var(--rule-strong);
         max-width:78ch;white-space:pre-wrap}}
.k-ft{{width:100%;border-collapse:collapse}}
.k-ft td{{padding:9px 12px;border-bottom:1px solid var(--rule);font-family:var(--mono);
         font-size:12px;color:var(--ink)}}
.k-ft tr:last-child td{{border-bottom:0}}
.k-ft .fn{{color:var(--ink-2);letter-spacing:.06em;text-transform:uppercase;font-size:10px;
          width:1%;white-space:nowrap}}
.k-ft .fl{{width:1%;white-space:nowrap}}
.k-ft .fp{{text-align:right;width:1%;font-variant-numeric:tabular-nums;font-weight:500}}
.k-ft .fe{{color:var(--ink-2);font-family:var(--sans);font-size:12.5px}}
.k-ft .grp td{{background:var(--surface-2);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
              color:var(--ink-2);font-weight:600}}
.k-flag{{border:1px solid var(--rule-strong);border-left:2px solid var(--accent);
        background:var(--surface-2);padding:16px}}
.k-flag .k{{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
           color:var(--accent)}}
.k-flag p{{margin:5px 0 0;font-size:13px;color:var(--ink-2) !important;max-width:70ch;
          font-family:var(--sans)}}
.k-vs{{display:flex;gap:28px;margin-top:12px;font-family:var(--mono);font-size:12px;flex-wrap:wrap}}
.k-vs b{{display:block;font-size:9px;letter-spacing:.12em;text-transform:uppercase;
        color:var(--ink-3);font-weight:500}}

/* ---- states ---- */
.k-state{{border:1px solid var(--rule-strong);background:var(--surface);padding:44px 24px}}
.k-state.empty{{text-align:center}}
.k-state h4{{font-family:var(--pixel);font-size:26px;font-weight:400;margin:0 0 12px;
            color:var(--ink)}}
.k-state p{{margin:0 auto;max-width:52ch;color:var(--ink-2) !important;font-size:13.5px;
           font-family:var(--sans)}}
.k-spec{{display:grid;gap:8px;margin-top:20px;text-align:left;max-width:420px;margin-inline:auto}}
.k-spec div{{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;
            border-bottom:1px solid var(--rule);padding-bottom:5px;color:var(--ink-2)}}
.k-spec b{{color:var(--ink);font-weight:500}}
.k-ok{{border:1px solid var(--rule-strong);border-left:2px solid var(--accent);background:var(--surface);
      padding:16px 18px;display:flex;gap:24px;flex-wrap:wrap;align-items:baseline}}
.k-ok .k{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
         color:var(--ink-3)}}
.k-ok .v{{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--ink) !important}}
.k-err{{border:1px solid var(--rule-strong);border-left:2px solid var(--accent);
       background:var(--surface);padding:18px}}
.k-err .k{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
          color:var(--accent)}}
.k-err p{{margin:6px 0 0;font-size:13.5px;color:var(--ink-2) !important;max-width:70ch;
         font-family:var(--sans)}}

/* ---- host widget restyling ---- */
[data-testid="stFileUploader"] section{{border:1px dashed var(--rule-strong) !important;
  background:var(--surface) !important;border-radius:0 !important;padding:24px !important}}
[data-testid="stFileUploader"] section:hover{{border-color:var(--accent) !important}}
.stButton > button{{font-family:var(--mono) !important;font-size:11px !important;
  letter-spacing:.1em !important;text-transform:uppercase !important;font-weight:500 !important;
  border-radius:0 !important;border:1px solid var(--ink) !important;background:var(--ink) !important;
  color:var(--canvas) !important;padding:11px 22px !important;transition:background .12s ease}}
.stButton > button:hover{{background:var(--accent) !important;border-color:var(--accent) !important;
  color:var(--canvas) !important}}
.stButton > button:focus-visible{{outline:2px solid var(--accent) !important;outline-offset:2px !important}}
.stDownloadButton > button{{font-family:var(--mono) !important;font-size:11px !important;
  letter-spacing:.1em !important;text-transform:uppercase !important;border-radius:0 !important;
  border:1px solid var(--rule-strong) !important;background:transparent !important;
  color:var(--ink) !important}}

/* The host wraps every button label in a markdown container, which the broad
   colour rule above claims with !important - leaving black text on the black
   button. These re-claim the label from the button's own colour. Descendant
   selectors are required: the text sits several wrappers below the <button>. */
.stButton > button, .stButton > button *{{color:var(--canvas) !important}}
/* Dark text on the accent, not white: white on #FF5A16 measures 3.12:1 and
   fails AA, while the canvas colour on the same orange reaches 6.27:1. */
.stButton > button:hover, .stButton > button:hover *{{color:var(--canvas) !important}}
.stDownloadButton > button, .stDownloadButton > button *{{color:var(--ink) !important}}
.stButton > button p, .stDownloadButton > button p{{
  font-family:var(--mono) !important;font-size:11px !important;font-weight:500 !important;
  letter-spacing:.1em !important;text-transform:uppercase !important;margin:0 !important}}
.stSelectbox div[data-baseweb="select"] > div{{border-radius:0 !important;
  border-color:var(--rule-strong) !important;font-family:var(--mono) !important;font-size:13px !important}}
.stRadio label, .stCheckbox label{{font-family:var(--mono) !important;font-size:12px !important}}
[data-testid="stNumberInput"] input{{border-radius:0 !important;font-family:var(--mono) !important}}
.stProgress > div > div > div{{background-color:var(--accent) !important}}
a:focus-visible, button:focus-visible, [tabindex]:focus-visible{{outline:2px solid var(--accent);
  outline-offset:2px}}

@media (max-width:900px){{
  .k-scores{{gap:20px}}
  .block-container{{padding:0 1rem 3rem !important}}
}}
@media (prefers-reduced-motion:reduce){{*{{transition:none !important;animation:none !important}}}}
</style>"""


# --------------------------------------------------------------------------- #
# Fragments
# --------------------------------------------------------------------------- #

def _e(value: Any) -> str:
    """Escape for HTML. Lead data is user-supplied and goes straight into markup."""
    return html.escape("" if value is None else str(value), quote=True)


def profile_summary(profile: Any) -> str:
    """The active target profile, shown as the yardstick a run was measured by."""
    if profile is None:
        return ""
    cells = [
        ("Industries", profile.industry_label()),
        ("Budget", profile.budget_label()),
        ("Location", profile.location or "Any"),
    ]
    body = "".join(
        f'<div><div class="k">{_e(label)}</div><div class="v">{_e(value)}</div></div>'
        for label, value in cells
    )
    return f'<div class="k-strip k-profile">{body}</div>'


def masthead(status: str) -> str:
    return (
        '<div class="k-head"><div class="k-brand">'
        '<b>Koya</b><span>Lead Triage</span><span class="k-tag">Console</span></div>'
        f'<div class="k-headmeta">{status}</div></div>'
    )


def hero() -> str:
    return (
        '<div class="k-hero">'
        '<div class="k-eyebrow"><span class="k-lbl">Inbound qualification</span>'
        '<span class="d"></span></div>'
        "<h1>LEAD TRIAGE</h1>"
        "<p>Upload a lead export and get a ranked, explained queue. Every score traces back "
        "to a phrase in the lead&rsquo;s own words, so you can see why a lead was ranked where "
        "it was &mdash; not just that it was.</p></div>"
    )


def section(number: str, title: str) -> str:
    return f'<div class="k-sec"><span class="n">{number}</span><span class="t">// {title}</span></div>'


def route_pill(route: str) -> str:
    mod, label = ROUTE_META.get(route, ("disq", route.title()))
    return f'<span class="k-route {mod}"><span class="k-mk {mod}"></span>{_e(label)}</span>'


def metric_tiles(summary: dict[str, int]) -> str:
    order = [
        ("Total processed", summary.get("total_processed", 0), "total", None),
        ("Contact now", summary.get("CONTACT_NOW", 0), "", "contact"),
        ("Nurture", summary.get("NURTURE", 0), "", "nurture"),
        ("Disqualify", summary.get("DISQUALIFY", 0), "", "disq"),
        ("Review", summary.get("REVIEW", 0), "", "review"),
    ]
    cells = []
    for label, value, extra, mod in order:
        marker = f'<span class="k-mk {mod}"></span>' if mod else ""
        cells.append(
            f'<div class="k-tile {extra}"><div class="k">{marker}{_e(label)}</div>'
            f'<div class="v">{value}</div></div>'
        )
    return f'<div class="k-tiles">{"".join(cells)}</div>'


# Makes a queue row selectable. Mounted as a headless component beside the
# table: it renders nothing and only reports which row was clicked.
#
# The listener is bound to the document once and delegates, rather than being
# attached to each row. Rows are re-rendered on every rerun, so per-row handlers
# would be lost the moment the queue redrew; delegation survives that. The
# callback into the current script run is kept on `window` and replaced on each
# mount, so a click always reaches the run that is live now.
QUEUE_CLICK_JS = """
export default function (component) {
    const { setTriggerValue } = component;
    window.__koyaSelectLead = (id) => setTriggerValue('lead', id);

    if (window.__koyaQueueBound) return;
    window.__koyaQueueBound = true;

    const rowFor = (event) => {
        const target = event.target;
        return target && target.closest ? target.closest('tr[data-lead]') : null;
    };
    const select = (row) => {
        if (row && window.__koyaSelectLead) {
            window.__koyaSelectLead(row.getAttribute('data-lead'));
        }
    };

    document.addEventListener('click', (event) => select(rowFor(event)));
    document.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const row = rowFor(event);
        if (!row) return;
        event.preventDefault();
        select(row);
    });
}
"""


def queue_table(rows: list[dict[str, Any]], selected_id: Optional[str]) -> str:
    body = []
    for row in rows:
        is_sel = str(row.get("lead_id")) == str(selected_id)
        rank = row.get("rank")
        fit, intent, total = (
            row.get("company_fit_score"), row.get("buying_intent_score"), row.get("total_score"),
        )
        cell = lambda v: (  # noqa: E731
            f'<td class="num k-dash">&mdash;</td>' if v is None else f'<td class="num">{_e(v)}</td>'
        )
        # `data-lead` is what makes the row clickable: QUEUE_CLICK_JS reads it
        # back off the clicked row. The id is escaped like every other cell, so
        # a lead id out of a CSV cannot break out of the attribute.
        body.append(
            f'<tr class="{"sel" if is_sel else ""}" data-lead="{_e(row.get("lead_id"))}"'
            f' tabindex="0" role="button"'
            f' aria-label="Open {_e(row.get("company") or row.get("lead_id"))}"'
            f'{" aria-current=true" if is_sel else ""}>'
            f'<td class="rk">{_e(rank) if rank else "&mdash;"}</td>'
            f'<td class="co"><b>{_e(row.get("company") or "&mdash;")}</b>'
            f'<i>{_e(row.get("lead_id"))}'
            f'{" &middot; " + _e(row.get("name")) if row.get("name") else ""}</i></td>'
            f'<td>{_e(row.get("source") or "&mdash;")}</td>'
            f'<td>{_e(row.get("title") or "&mdash;")}</td>'
            f"{cell(fit)}{cell(intent)}{cell(total)}"
            f"<td>{route_pill(row.get('recommendation', ''))}</td></tr>"
        )
    return (
        '<div class="k-tablewrap"><table class="k-table"><thead><tr>'
        '<th class="rk">#</th><th>Lead</th><th>Source</th><th>Title</th>'
        '<th class="num">Fit</th><th class="num">Intent</th><th class="num">Total</th>'
        "<th>Route</th></tr></thead><tbody>"
        f'{"".join(body)}</tbody></table></div>'
    )


def _factor_rows(breakdown: dict[str, Any], factors, show_points: bool) -> str:
    out = []
    for key in factors:
        entry = breakdown.get(key)
        if not isinstance(entry, dict):
            continue
        evidence = entry.get("evidence") or "not specified"
        quoted = f"&ldquo;{_e(evidence)}&rdquo;" if evidence != "not specified" else "not specified"
        target_note = ""
        if key == "budget_capacity" and entry.get("target_range"):
            target_note = f' <span class="k-meta">&middot; target {_e(entry["target_range"])}</span>'
        origin = ""
        if entry.get("source") == "notes":
            origin = ' <span class="k-meta">&middot; from notes</span>'
        elif entry.get("source") == "structured":
            origin = ' <span class="k-meta">&middot; from lead data</span>'
        points = f'<td class="fp">{_e(entry.get("points"))}</td>' if show_points else ""
        out.append(
            f'<tr><td class="fn">{_e(FACTOR_LABELS.get(key, key))}</td>'
            f'<td class="fl">{_e(entry.get("level"))}</td>{points}'
            f'<td class="fe">{quoted}{origin}{target_note}</td></tr>'
        )
    return "".join(out)


def factor_table(breakdown: dict[str, Any], fit: Any, intent: Any, show_points: bool) -> str:
    span = 4 if show_points else 3
    if show_points:
        head_fit = f"Company fit &mdash; {_e(fit)} / 50"
        head_int = f"Buying intent &mdash; {_e(intent)} / 50"
    else:
        head_fit, head_int = "Company signals", "Intent signals"
    return (
        '<table class="k-ft">'
        f'<tr class="grp"><td colspan="{span}">{head_fit}</td></tr>'
        f"{_factor_rows(breakdown, FIT_FACTORS, show_points)}"
        f'<tr class="grp"><td colspan="{span}">{head_int}</td></tr>'
        f"{_factor_rows(breakdown, INTENT_FACTORS, show_points)}"
        "</table>"
    )


def conflict_flag(breakdown: dict[str, Any]) -> str:
    blocks = []
    for key, label in (("company_size_signal", "Headcount"), ("budget_capacity", "Budget")):
        entry = breakdown.get(key)
        if not (isinstance(entry, dict) and entry.get("conflict")):
            continue
        blocks.append(
            f'<div class="k-flag"><span class="k">{_e(label)} &middot; notes take precedence</span>'
            "<p>The notes state a figure the uploaded lead data contradicts. The value from the "
            "notes was used for scoring; the original value is kept, not overwritten.</p>"
            f'<div class="k-vs"><span><b>From notes</b>{_e(entry.get("notes_value"))}</span>'
            f'<span><b>Original lead data</b>{_e(entry.get("structured_value"))}</span>'
            f'<span><b>Scored</b>{_e(entry.get("level"))} '
            f'&middot; {_e(entry.get("points"))} pts</span></div></div>'
        )
    return "".join(blocks)


def explanation_block(parts: list[str]) -> str:
    """Verdict on its own line, reasoning and supporting detail as separate
    paragraphs - a wall of text is harder to act on than the same words spaced."""
    if not parts:
        return ""
    rendered = [f'<p class="verdict">{_e(parts[0])}</p>']
    rendered += [f"<p>{_e(p)}</p>" for p in parts[1:] if p]
    return f'<div class="k-expl">{"".join(rendered)}</div>'


def exclusion_reason(lead: dict[str, Any]) -> str:
    """The stated-purpose exclusion as a sentence, or "" when there is none.

    Worded for whoever is reading the queue, not for whoever wrote the enum. A
    lead held for review because its signals contradict each other gets the
    softer wording: nothing has been decided about it yet.
    """
    phrase = NON_BUYER_PHRASES.get(lead.get("non_buyer_signal") or "none")
    if not phrase:
        return ""
    if lead.get("scores_suppressed"):
        return (
            f"This lead stated that {phrase}, but the rest of the enquiry reads "
            "like a genuine one. It is held for a decision rather than routed "
            "automatically."
        )
    return f"Disqualified because the lead stated that {phrase}."


def lead_detail(lead: dict[str, Any]) -> str:
    """The full assessment panel for one lead.

    A review lead is rendered through the same structure, minus anything that
    would imply a routing score it deliberately does not have: no score row, no
    points column, no subtotals.
    """
    suppressed = bool(lead.get("scores_suppressed"))
    breakdown = lead.get("breakdown") or {}

    scores_html = ""
    if not suppressed:
        scores_html = (
            '<div class="k-scores">'
            f'<div class="k-sc"><span class="k">Fit</span>'
            f'<span class="v">{_e(lead.get("company_fit_score"))}</span>'
            '<span class="d">/ 50</span></div>'
            f'<div class="k-sc"><span class="k">Intent</span>'
            f'<span class="v">{_e(lead.get("buying_intent_score"))}</span>'
            '<span class="d">/ 50</span></div>'
            f'<div class="k-sc hero"><span class="k">Total</span>'
            f'<span class="v">{_e(lead.get("total_score"))}</span>'
            '<span class="d">/ 100</span></div></div>'
        )

    strip = "".join(
        f'<div><div class="k">{_e(label)}</div><div class="v">{_e(value or "—")}</div></div>'
        for label, value in (
            ("Buyer status", lead.get("is_potential_buyer")),
            ("Buyer type", lead.get("buyer_type")),
            ("Source", lead.get("source")),
            ("Title", lead.get("title")),
            ("Employees on file", lead.get("employees_display")),
            ("Budget on file", lead.get("budget_display")),
        )
    )

    blocks = []

    if suppressed and lead.get("assessment_failed"):
        # A technical failure is not a judgement about the lead, so it does not
        # borrow the model's voice. The provider's own error text never reaches
        # this layer - it stops at the extraction boundary and goes to the logs.
        blocks.append(
            '<div class="k-block"><span class="k-lbl">Why this needs a person</span>'
            '<div class="k-expl"><p>This lead could not be assessed automatically '
            "and has been set aside for manual review.</p>"
            "<p>Its details are shown below exactly as they arrived, so it can be "
            "read and routed by hand.</p></div></div>"
        )
    elif suppressed:
        reason = lead.get("review_reason") or "The notes do not settle whether this is a buyer."
        blocks.append(
            '<div class="k-block"><span class="k-lbl">Why this needs a person</span>'
            f'<div class="k-expl"><p>{_e(reason)}</p>'
            "<p>Nothing in the lead&rsquo;s own words decides between a genuine enquiry and an "
            "approach made for another reason, so it is held for a decision rather than "
            "routed automatically.</p></div></div>"
        )
    elif lead.get("explanation_parts"):
        blocks.append(
            '<div class="k-block"><span class="k-lbl">Assessment</span>'
            f'{explanation_block(lead["explanation_parts"])}</div>'
        )

    exclusion = exclusion_reason(lead)
    evidence = lead.get("disqualification_evidence")
    if exclusion or evidence:
        said = f'<div class="k-expl"><p>{_e(exclusion)}</p></div>' if exclusion else ""
        quoted = (
            '<div class="k-flag"><span class="k">From the lead&rsquo;s own words</span>'
            f'<p>&ldquo;{_e(evidence)}&rdquo;</p></div>'
            if evidence else ""
        )
        blocks.append(
            '<div class="k-block"><span class="k-lbl">Basis for this decision</span>'
            f"{said}{quoted}</div>"
        )

    location = breakdown.get("location")
    if isinstance(location, dict) and location.get("level") in {"match", "mismatch", "unknown"}:
        label = {"match": "In target location", "mismatch": "Outside target location",
                 "unknown": "Location not stated"}[location["level"]]
        stated = location.get("stated_location")
        detail_text = f"{_e(stated)}" if stated else "not stated in the lead's notes"
        blocks.append(
            '<div class="k-block"><span class="k-lbl">Location</span>'
            f'<table class="k-ft"><tr><td class="fn">{_e(label)}</td>'
            f'<td class="fe">{detail_text}'
            f' <span class="k-meta">&middot; target {_e(location.get("target_location"))}</span>'
            "</td></tr></table></div>"
        )

    conflict = conflict_flag(breakdown)
    if conflict and not suppressed:
        blocks.append(f'<div class="k-block"><span class="k-lbl">Data conflict</span>{conflict}</div>')

    # A lead that was never assessed has no signals to show. Rendering the
    # all-unknown fallback under "Signals observed" would imply the model looked
    # and found nothing, which is a different claim from "we never got an answer".
    if not lead.get("assessment_failed"):
        blocks.append(
            '<div class="k-block"><span class="k-lbl">'
            f'{"Signals observed" if suppressed else "Scoring factors"}</span>'
            + factor_table(breakdown, lead.get("company_fit_score"),
                           lead.get("buying_intent_score"), show_points=not suppressed)
            + "</div>"
        )

    if lead.get("notes"):
        blocks.append(
            '<div class="k-block"><span class="k-lbl">Original notes</span>'
            f'<p class="k-notes">{_e(lead["notes"])}</p></div>'
        )

    company = f' &middot; {_e(lead["company"])}' if lead.get("company") else ""
    return (
        '<div class="k-panel"><div class="k-phead"><div>'
        f'<div class="k-id">{_e(lead.get("lead_id"))}{company}</div>'
        f'<h3>{_e(lead.get("name") or lead.get("lead_id"))}</h3>'
        f'{route_pill(lead.get("recommendation", ""))}</div>{scores_html}</div>'
        f'<div class="k-strip">{strip}</div>{"".join(blocks)}</div>'
    )


# --------------------------------------------------------------------------- #
# Single-lead report - the download behind 04 // LEAD DETAIL
# --------------------------------------------------------------------------- #

REPORT_WIDTH = 74
# Wide enough to leave a gap after the longest label there is ("Purchasing
# readiness", at 20) - at 20 the column ran straight into its own value.
_LABEL_WIDTH = 22


def _report_field(label: str, value: Any, indent: int = 2) -> str:
    shown = "" if value is None else str(value).strip()
    if not shown or shown == "unknown":
        shown = "not recorded"
    return f'{" " * indent}{label.ljust(_LABEL_WIDTH)}{shown}'


def _report_paragraph(text: str, indent: int = 2) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=REPORT_WIDTH, initial_indent=pad, subsequent_indent=pad)


def _report_factors(breakdown: dict[str, Any], factors, show_points: bool) -> list[str]:
    out = []
    for key in factors:
        entry = breakdown.get(key)
        if not isinstance(entry, dict):
            continue
        level = entry.get("level") or "unknown"
        head = f"{level} · {entry.get('points')} pts" if show_points else level
        out.append(f'    {FACTOR_LABELS.get(key, key).ljust(_LABEL_WIDTH)}{head}')

        detail = []
        evidence = (entry.get("evidence") or "").strip()
        detail.append(f'"{evidence}"' if evidence and evidence != "not specified"
                      else "nothing stated in the notes")
        if entry.get("source") == "notes":
            detail.append("from the notes")
        elif entry.get("source") == "structured":
            detail.append("from the lead data")
        if key == "budget_capacity" and entry.get("target_range"):
            detail.append(f"target {entry['target_range']}")
        out.append(_report_paragraph(" · ".join(detail), indent=8))
    return out


def lead_report(lead: dict[str, Any], profile: Any = None) -> bytes:
    """One selected lead, as a plain-text report a person can read or forward.

    Built entirely from the record already on screen - nothing is recomputed and
    nothing is fetched. It carries the same facts as 04 // LEAD DETAIL, in the
    same words: a review lead's scores stay withheld here too, and the technical
    reason an assessment failed is no more printable than it is displayable.
    """
    suppressed = bool(lead.get("scores_suppressed"))
    breakdown = lead.get("breakdown") or {}
    route = ROUTE_META.get(lead.get("recommendation", ""), ("", "—"))[1]

    lines = [
        "KOYA · LEAD TRIAGE",
        "Lead assessment",
        "=" * REPORT_WIDTH,
        "",
        "LEAD",
        _report_field("Lead ID", lead.get("lead_id")),
        _report_field("Name", lead.get("name")),
        _report_field("Company", lead.get("company")),
        _report_field("Title", lead.get("title")),
        _report_field("Source", lead.get("source")),
        _report_field("Buyer status", lead.get("is_potential_buyer")),
        _report_field("Buyer type", lead.get("buyer_type")),
        _report_field("Employees on file", lead.get("employees_display")),
        _report_field("Budget on file", lead.get("budget_display")),
        "",
    ]

    if profile is not None:
        lines += [
            "TARGET PROFILE",
            _report_paragraph("The yardstick this lead was assessed against."),
            "",
            _report_field("Industries", profile.industry_label()),
            _report_field("Budget", profile.budget_label()),
            _report_field("Location", profile.location or "Any"),
            "",
        ]

    lines.append("ASSESSMENT")
    lines.append(_report_field("Route", route))
    if suppressed:
        # Ranked like every other row - by route, not by a score it does not
        # have. Only the score stays withheld.
        lines.append(_report_field("Rank", lead.get("rank")))
        lines.append(_report_field("Score", "withheld pending review"))
        lines.append("")
        reason = lead.get("review_reason") or (
            "The notes do not settle whether this is a buyer."
        )
        lines.append(_report_paragraph(reason))
        lines.append("")
        lines.append(_report_paragraph(
            "It is held for a decision rather than routed automatically, and "
            "carries no score for that reason."
        ))
    else:
        lines.append(_report_field("Rank", lead.get("rank")))
        lines.append(_report_field("Company fit", f'{lead.get("company_fit_score")} / 50'))
        lines.append(_report_field("Buying intent", f'{lead.get("buying_intent_score")} / 50'))
        lines.append(_report_field("Total", f'{lead.get("total_score")} / 100'))
        for part in lead.get("explanation_parts") or []:
            lines += ["", _report_paragraph(part)]
    lines.append("")

    exclusion = exclusion_reason(lead)
    if exclusion or lead.get("disqualification_evidence"):
        lines.append("BASIS FOR THIS DECISION")
        if exclusion:
            lines += [_report_paragraph(exclusion), ""]
        if lead.get("disqualification_evidence"):
            lines.append(_report_paragraph(
                f'From the lead\'s own words: "{lead["disqualification_evidence"]}"'
            ))
        lines.append("")

    if not lead.get("assessment_failed"):
        lines.append("SIGNALS OBSERVED" if suppressed else "SCORING FACTORS")
        head_fit = "Company signals" if suppressed else (
            f'Company fit — {lead.get("company_fit_score")} / 50'
        )
        head_intent = "Intent signals" if suppressed else (
            f'Buying intent — {lead.get("buying_intent_score")} / 50'
        )
        lines.append(f"  {head_fit}")
        lines += _report_factors(breakdown, FIT_FACTORS, show_points=not suppressed)
        lines.append(f"  {head_intent}")
        lines += _report_factors(breakdown, INTENT_FACTORS, show_points=not suppressed)
        lines.append("")

    location = breakdown.get("location")
    if isinstance(location, dict) and location.get("level") in {"match", "mismatch", "unknown"}:
        label = {"match": "In target location", "mismatch": "Outside target location",
                 "unknown": "Location not stated"}[location["level"]]
        stated = location.get("stated_location") or "not stated in the lead's notes"
        lines += [
            "LOCATION",
            _report_field(label, f'{stated} · target {location.get("target_location")}'),
            "",
        ]

    conflicts = []
    for key, label in (("company_size_signal", "Headcount"), ("budget_capacity", "Budget")):
        entry = breakdown.get(key)
        if isinstance(entry, dict) and entry.get("conflict"):
            conflicts.append(_report_paragraph(
                f'{label} was scored from the notes ({entry.get("notes_value")}); '
                f'the original lead data says {entry.get("structured_value")}. '
                "The original value is kept, not overwritten."
            ))
    if conflicts and not suppressed:
        lines += ["DATA CONFLICT", *conflicts, ""]

    if lead.get("notes"):
        lines += ["ORIGINAL NOTES", _report_paragraph(str(lead["notes"])), ""]

    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def report_filename(lead: dict[str, Any]) -> str:
    """A filename that identifies the lead without leaking anything else."""
    stem = "".join(
        ch if ch.isalnum() or ch in "-_" else "-" for ch in str(lead.get("lead_id") or "lead")
    )
    return f"lead-detail-{stem or 'lead'}.txt"
