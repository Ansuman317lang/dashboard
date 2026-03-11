import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os
import tempfile
from difflib import get_close_matches

# ──────────────────────────────────────────────
# Intelligent Column Matching & Value Normalization
# ──────────────────────────────────────────────

def _normalize_name(name):
    """Reduce a name to lowercase alphanumeric only for fuzzy comparison."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower().strip())


# Known aliases: maps canonical column name → set of alternative normalized names.
# These catch completely different naming conventions used in production files.
COLUMN_ALIASES = {
    "Remediation_status": {
        "remediationstatus", "remedstatus", "remediationstat", "remstatus",
        "remediation", "statusremediation", "remediationstate",
        "remstats", "remedstat", "statusofrem", "statusrem",
        "remeditiationstatus", "remediationsatus",  # common typos
    },
    "CTI overdue": {
        "ctioverdue", "ctiover", "overduecti", "ctioverduestatus",
        "ctioverdueflag", "overduestatus", "ctidue", "ctiovdue",
        "ctioverdu", "ctioverdeu",  # common typos
    },
    "CTI Overdue": {
        "ctioverdue", "ctiover", "overduecti", "ctioverduestatus",
        "ctioverdueflag", "overduestatus", "ctidue", "ctiovdue",
        "ctioverdu", "ctioverdeu",
    },
    "owner": {
        "owner", "ownername", "assetowner", "ownergroup", "assignedowner",
        "teamowner", "vulnowner", "ownedby", "assignee",
        "ownerteam", "ownerdetail", "responsibleowner",
    },
    "status as on last week date": {
        "statusasonlastweekdate", "lastweekstatus", "statuslastweek",
        "lastweekdate", "weeklystatus", "statuslastweekdate",
        "statlastweek", "prevweekstatus", "previousweekstatus",
        "lastwkstatus", "statuslastwk", "lastwkdate",
    },
    "Status as on last week date": {
        "statusasonlastweekdate", "lastweekstatus", "statuslastweek",
        "lastweekdate", "weeklystatus", "statuslastweekdate",
        "statlastweek", "prevweekstatus", "previousweekstatus",
        "lastwkstatus", "statuslastwk", "lastwkdate",
    },
    "priority": {
        "priority", "prioritylevel", "prio", "vulnpriority",
        "priolevel", "priorityrating", "prlevel", "prty",
        "vulnprio", "riskpriority", "priorityclass",
    },
    "Priority": {
        "priority", "prioritylevel", "prio", "vulnpriority",
        "priolevel", "priorityrating", "prlevel", "prty",
        "vulnprio", "riskpriority", "priorityclass",
    },
    "criticity": {
        "criticity", "criticality", "severity", "criticallevel",
        "severitylevel", "critiity", "criticalitylevel",
        "riskrating", "vulnseverity", "severityrating",
        "criticity1", "critcity",  # common typos
    },
    "Metier": {
        "metier", "metiername", "businessunit", "metiergroup",
        "metierlabel", "bu", "businessline", "dept",
        "department", "metiercode", "mtier", "metir",  # common typos
    },
    "Status as on present date": {
        "statusasonpresentdate", "presentstatus", "currentstatus",
        "statuspresent", "statusasofnow", "statustoday",
        "todaystatus", "currentdatestatus", "presentdatestatus",
        "statuspresentdate", "statusnow", "statcurrent",
    },
}

# Keyword-based matching: if ALL keywords appear in a normalized column name, it's a match.
# This catches arbitrary word ordering and extra words in column names.
COLUMN_KEYWORDS = {
    "Remediation_status":           [["remediat", "status"], ["remediat", "stat"]],
    "CTI overdue":                  [["cti", "overdue"], ["cti", "over", "due"]],
    "CTI Overdue":                  [["cti", "overdue"], ["cti", "over", "due"]],
    "owner":                        [["owner"]],
    "status as on last week date":  [["status", "last", "week"], ["last", "week", "date"]],
    "Status as on last week date":  [["status", "last", "week"], ["last", "week", "date"]],
    "priority":                     [["priority"], ["prio"]],
    "Priority":                     [["priority"], ["prio"]],
    "criticity":                    [["criticit"], ["critical"], ["severity"]],
    "Metier":                       [["metier"]],
    "Status as on present date":    [["status", "present"], ["status", "current"], ["present", "date"]],
}


def _alias_lookup(canonical, actual_normalized, used):
    """Check if any known alias for a canonical column matches an actual column."""
    aliases = COLUMN_ALIASES.get(canonical, set())
    for alias in aliases:
        if alias in actual_normalized and actual_normalized[alias] not in used:
            return actual_normalized[alias]
    return None


def _keyword_lookup(canonical, actual_normalized, used):
    """Check if all required keywords appear in any actual column name."""
    keyword_sets = COLUMN_KEYWORDS.get(canonical, [])
    if not keyword_sets:
        return None
    for actual_norm, actual_orig in actual_normalized.items():
        if actual_orig in used:
            continue
        for kw_set in keyword_sets:
            if all(kw in actual_norm for kw in kw_set):
                return actual_orig
    return None


# ── Content-based fingerprinting ──
# When column names are completely different (e.g. "col_7"), detect columns
# by scanning the actual values they contain.

def _get_unique_lower(series, sample_size=200):
    """Return a set of unique lowered string values from a sample of the series."""
    sample = series.dropna().head(sample_size).astype(str).str.strip().str.lower()
    return set(sample.unique())


# Fingerprint rules: each canonical column has test functions that score how
# likely a given column's data matches it (0.0 = no match, 1.0 = certain).
_REMEDIATION_VALUES = {
    'plan_defined_out_of_eta', 'plandefinedoutofeta', 'plan defined out of eta',
    'under_risk_acceptance_process', 'underriskacceptanceprocess', 'under risk acceptance process',
    'wrong_owner', 'wrongowner', 'wrong owner',
}

_PRIORITY_PATTERNS = [r'pr[0-9]', r'priority', r'act$', r'production.*asset']

_LASTWEEK_PATTERNS = [r'newly\s*added', r'existing', r'removed', r'closed']

_PRESENT_PATTERNS = [r'open', r'closed', r'remediated', r'in\s*progress']


def _score_remediation_status(vals):
    """Score how likely column data is Remediation_status."""
    norm_vals = {re.sub(r'[^a-z0-9]', '', v) for v in vals}
    hits = norm_vals & {re.sub(r'[^a-z0-9]', '', v) for v in _REMEDIATION_VALUES}
    return min(1.0, len(hits) / 2) if hits else 0.0


def _score_cti_overdue(vals):
    """Score how likely column data is CTI overdue (yes/no style)."""
    yes_no = {'yes', 'no', 'y', 'n', 'true', 'false', 'oui', 'non',
              'overdue sla', 'within sla', 'overdue', 'within'}
    overlap = vals & yes_no
    # Must look like binary yes/no and have few unique values
    if len(vals) <= 6 and len(overlap) >= 1:
        return min(1.0, len(overlap) / 2)
    return 0.0


def _score_owner(vals):
    """Score how likely column data is owner (many unique string names)."""
    # Owner columns tend to have many diverse proper-name-like strings
    if len(vals) < 3:
        return 0.0
    # Not a yes/no column, not numeric-heavy
    yes_no = {'yes', 'no', 'y', 'n', 'true', 'false'}
    if vals & yes_no:
        return 0.0
    return 0.3  # low confidence, mostly rely on name matching


def _score_priority(vals):
    """Score how likely column data is priority."""
    score = 0.0
    for v in vals:
        for pat in _PRIORITY_PATTERNS:
            if re.search(pat, v):
                score += 0.3
                break
    return min(1.0, score)


def _score_status_lastweek(vals):
    """Score how likely column data is status as on last week date."""
    score = 0.0
    for v in vals:
        for pat in _LASTWEEK_PATTERNS:
            if re.search(pat, v):
                score += 0.4
                break
    return min(1.0, score)


def _score_criticity(vals):
    """Score how likely column data is criticity/severity."""
    sev_words = {'critical', 'high', 'medium', 'low', 'info', 'informational',
                 'urgent', 'severe', 'moderate', 'minor', 'negligible'}
    overlap = vals & sev_words
    if len(vals) <= 8 and len(overlap) >= 2:
        return min(1.0, len(overlap) / 3)
    return 0.0


def _score_status_present(vals):
    """Score how likely column data is status as on present date."""
    score = 0.0
    for v in vals:
        for pat in _PRESENT_PATTERNS:
            if re.search(pat, v):
                score += 0.3
                break
    return min(1.0, score)


CONTENT_FINGERPRINTS = {
    "Remediation_status": _score_remediation_status,
    "CTI overdue": _score_cti_overdue,
    "CTI Overdue": _score_cti_overdue,
    "owner": _score_owner,
    "priority": _score_priority,
    "Priority": _score_priority,
    "status as on last week date": _score_status_lastweek,
    "Status as on last week date": _score_status_lastweek,
    "criticity": _score_criticity,
    "Status as on present date": _score_status_present,
}


def _content_lookup(canonical, df, used, threshold=0.5):
    """Identify a column by scanning actual data values."""
    scorer = CONTENT_FINGERPRINTS.get(canonical)
    if scorer is None:
        return None
    best_col = None
    best_score = 0.0
    for col in df.columns:
        if col in used:
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            vals = _get_unique_lower(df[col])
            score = scorer(vals)
            if score > best_score:
                best_score = score
                best_col = col
    if best_score >= threshold:
        return best_col
    return None


def match_columns(df_columns, required_columns, df=None):
    """
    Find the best matching actual column in the DataFrame for each required column.
    7-step pipeline: exact → case-insensitive → normalized → alias → keyword
                     → fuzzy name → content fingerprint.
    Returns dict {canonical_name: actual_column_name or None}.
    """
    actual_cols = list(df_columns)
    actual_lower = {c.lower().strip(): c for c in actual_cols}
    actual_normalized = {_normalize_name(c): c for c in actual_cols}

    mapping = {}
    used = set()
    for req in required_columns:
        # 1. Exact match
        if req in actual_cols and req not in used:
            mapping[req] = req
            used.add(req)
            continue
        # 2. Case-insensitive
        req_lower = req.lower().strip()
        if req_lower in actual_lower and actual_lower[req_lower] not in used:
            mapping[req] = actual_lower[req_lower]
            used.add(actual_lower[req_lower])
            continue
        # 3. Normalized (ignore underscores, spaces, hyphens, case)
        req_norm = _normalize_name(req)
        if req_norm in actual_normalized and actual_normalized[req_norm] not in used:
            mapping[req] = actual_normalized[req_norm]
            used.add(actual_normalized[req_norm])
            continue
        # 4. Alias lookup (known production naming variants)
        alias_hit = _alias_lookup(req, actual_normalized, used)
        if alias_hit:
            mapping[req] = alias_hit
            used.add(alias_hit)
            continue
        # 5. Keyword-based matching (checks if key concepts appear in column name)
        kw_hit = _keyword_lookup(req, actual_normalized, used)
        if kw_hit:
            mapping[req] = kw_hit
            used.add(kw_hit)
            continue
        # 6. Fuzzy match (catches typos and abbreviations)
        available = [k for k, v in actual_normalized.items() if v not in used]
        close = get_close_matches(req_norm, available, n=1, cutoff=0.55)
        if close:
            mapping[req] = actual_normalized[close[0]]
            used.add(actual_normalized[close[0]])
            continue
        # 7. Content fingerprint: scan actual data values to identify the column
        if df is not None:
            content_hit = _content_lookup(req, df, used)
            if content_hit:
                mapping[req] = content_hit
                used.add(content_hit)
                continue
        mapping[req] = None
    return mapping


def select_and_rename_columns(df, required_columns):
    """
    Fuzzy-match required columns from df, rename to canonical names, drop extras.
    Uses name matching first, then falls back to content-based detection.
    """
    col_map = match_columns(df.columns, required_columns, df=df)
    rename_dict = {}
    keep = []
    for canonical, actual in col_map.items():
        if actual is not None and actual in df.columns:
            rename_dict[actual] = canonical
            keep.append(actual)
    return df[keep].rename(columns=rename_dict)


def find_sheet(excel_path, target_name):
    """
    Find best matching sheet name from an Excel file.
    Handles casing, spelling differences, and typos.
    """
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    if target_name in sheets:
        return target_name
    for s in sheets:
        if s.lower().strip() == target_name.lower().strip():
            return s
    for s in sheets:
        if _normalize_name(s) == _normalize_name(target_name):
            return s
    norm_sheets = {_normalize_name(s): s for s in sheets}
    close = get_close_matches(_normalize_name(target_name), list(norm_sheets.keys()), n=1, cutoff=0.6)
    if close:
        return norm_sheets[close[0]]
    return target_name


# ── Value normalization helpers ──

_YES_WORDS = {'yes', 'y', 'true', '1', 'oui', 'si', 'yeah', 'yep'}
_NO_WORDS = {'no', 'n', 'false', '0', 'non', 'nah', 'nope'}


def _safe_lower(val):
    if pd.isna(val):
        return None
    return str(val).strip().lower()


def normalize_yes_no(series, yes_val="YES", no_val="NO"):
    """Map any yes/no variant (YES, yes, Yes, Y, true, etc.) to canonical values."""
    def _f(v):
        s = _safe_lower(v)
        if s is None:
            return v
        if s in _YES_WORDS:
            return yes_val
        if s in _NO_WORDS:
            return no_val
        return v
    return series.map(_f)


def normalize_remediation_status(series):
    """Normalize Remediation_status spelling variants."""
    canonical_map = {
        'plandefinedoutofeta': 'plan_defined_out_of_eta',
        'underriskacceptanceprocess': 'under_risk_acceptance_process',
        'wrongowner': 'wrong_owner',
    }
    def _f(v):
        s = _safe_lower(v)
        if s is None:
            return v
        key = re.sub(r'[^a-z0-9]', '', s)
        return canonical_map.get(key, v)
    return series.map(_f)


def normalize_status_week(series, canonical="Newly added"):
    """Normalize 'Newly added'/'Newly Added' regardless of casing/spacing."""
    def _f(v):
        s = _safe_lower(v)
        if s is None:
            return v
        if 'newly' in s and 'add' in s:
            return canonical
        return v
    return series.map(_f)


def normalize_priority_infra(series):
    """Normalize infra priority values to canonical forms."""
    rules = [
        (r'pr3.*other.*production.*act', 'PR3-other production asset ACT'),
        (r'pr4.*all.*other.*assets.*act', 'PR4-all other assets ACT'),
    ]
    def _f(v):
        s = _safe_lower(v)
        if s is None:
            return v
        for pat, canon in rules:
            if re.search(pat, s):
                return canon
        return v
    return series.map(_f)


def normalize_priority_aps(series):
    """Normalize APS priority values to canonical forms."""
    rules = [
        (r'pr3.*other.*production.*act', 'PR3 - Other Production Asset ACT'),
        (r'pr4.*all.*other.*assets.*act', 'PR4 - All Other Assets ACT'),
        (r'pr4.*other.*production.*act', 'PR4 - Other Production Asset ACT'),
    ]
    def _f(v):
        s = _safe_lower(v)
        if s is None:
            return v
        for pat, canon in rules:
            if re.search(pat, s):
                return canon
        return v
    return series.map(_f)


# Required columns per sheet (canonical names as used by the rest of the code)
INFRA_COLUMNS = ["Remediation_status", "CTI overdue", "owner",
                 "status as on last week date", "priority"]
NETWORK_COLUMNS = ["owner", "CTI overdue", "criticity"]
APS_COLUMNS = ["Remediation_status", "Status as on last week date", "Priority",
               "CTI Overdue", "Metier", "Status as on present date"]

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

#*******************************************************
#***************FROM HERE 10th MARCH 2026 ****************


def detect_file_type(uploaded_file):
    """
    Auto-detect whether an uploaded Excel file is 'data' (infra/network) or 'aps'.
    Examines sheet names, column names, and data content.
    Returns: 'data', 'aps', or 'unknown'
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names
        norm_sheets = [_normalize_name(s) for s in sheets]

        # --- Step 1: Check sheet names for strong signals ---
        aps_sheet_signals = {'aps', 'application', 'appsec', 'applicationsecurity'}
        data_sheet_signals = {'infra', 'infrastructure', 'network', 'newtwork', 'nework'}

        has_aps_sheet = any(ns in aps_sheet_signals or
                           any(a in ns for a in ['aps', 'applic'])
                           for ns in norm_sheets)
        has_data_sheet = any(ns in data_sheet_signals or
                            any(d in ns for d in ['infra', 'network'])
                            for ns in norm_sheets)

        if has_aps_sheet and not has_data_sheet:
            return 'aps'
        if has_data_sheet and not has_aps_sheet:
            return 'data'

        # --- Step 2: Check columns in first sheet for fingerprints ---
        df_sample = pd.read_excel(uploaded_file, sheet_name=0, nrows=50)
        col_norms = [_normalize_name(c) for c in df_sample.columns]

        # APS-specific: "Metier" column is unique to APS
        metier_keywords = {'metier', 'metiername', 'businessunit'}
        has_metier = any(cn in metier_keywords or 'metier' in cn for cn in col_norms)

        # Data-specific: "owner" without "Metier" suggests infra/network
        has_owner = any('owner' in cn for cn in col_norms)

        # APS-specific: "Status as on present date" is unique to APS
        has_present_status = any(
            ('present' in cn and 'status' in cn) or
            ('current' in cn and 'status' in cn)
            for cn in col_norms
        )

        # Data-specific: "criticity" / "severity" suggests network sheet
        has_criticity = any(
            cn in {'criticity', 'criticality', 'severity'} or 'criticit' in cn
            for cn in col_norms
        )

        aps_score = 0
        data_score = 0

        if has_metier:
            aps_score += 3
        if has_present_status:
            aps_score += 2
        if has_criticity:
            data_score += 2
        if has_owner and not has_metier:
            data_score += 1

        # Check multiple sheets (data.xlsx typically has infra + network)
        if len(sheets) >= 2:
            data_score += 1

        # --- Step 3: Content-based detection on sample data ---
        for col in df_sample.columns:
            if df_sample[col].dtype == 'object':
                vals = _get_unique_lower(df_sample[col])
                # Metier-like values (business unit names) → APS
                if _score_remediation_status(vals) > 0.5:
                    # Both have remediation — not decisive
                    pass
                if _score_criticity(vals) > 0.5:
                    data_score += 2

        if aps_score > data_score:
            return 'aps'
        if data_score > aps_score:
            return 'data'

        # --- Step 4: Fallback — check all sheets for column patterns ---
        for sheet in sheets:
            try:
                df_s = pd.read_excel(uploaded_file, sheet_name=sheet, nrows=10)
                s_norms = [_normalize_name(c) for c in df_s.columns]
                if any('metier' in cn for cn in s_norms):
                    return 'aps'
                if any('criticit' in cn or cn == 'severity' for cn in s_norms):
                    return 'data'
            except Exception:
                continue

        return 'unknown'
    except Exception:
        return 'unknown'


def save_uploaded_file(uploaded_file, target_name):
    """Save an uploaded file to the uploads directory with the target name."""
    path = os.path.join(UPLOAD_DIR, target_name)
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return path


def get_data_path():
    """Return the path to the data Excel file (uploaded or default)."""
    uploaded = os.path.join(UPLOAD_DIR, "data.xlsx")
    if os.path.exists(uploaded):
        return uploaded
    return "data.xlsx"


def get_aps_path():
    """Return the path to the APS Excel file (uploaded or default)."""
    uploaded = os.path.join(UPLOAD_DIR, "aps.xlsx")
    if os.path.exists(uploaded):
        return uploaded
    return "aps.xlsx"


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vulnerability Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state for selected card & page
# ──────────────────────────────────────────────
if "selected" not in st.session_state:
    st.session_state.selected = None
if "page" not in st.session_state:
    st.session_state.page = "infra"
if "net_selected" not in st.session_state:
    st.session_state.net_selected = None
if "aps_selected" not in st.session_state:
    st.session_state.aps_selected = None
if "upload_msg" not in st.session_state:
    st.session_state.upload_msg = None

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
        max-width: 100%;
    }
    h1 { text-align: center; margin: 0 0 0.3rem 0; font-size: 1.6rem; }

    /* Card button styling */
    div.stButton > button {
        width: 100%;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.5rem 0.3rem;
        background: white;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s, border 0.2s;
        color: black !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 3px 12px rgba(0,0,0,0.12);
        border-color: #667eea;
        color: black !important;
    }
    div.stButton > button:focus {
        color: black !important;
    }
    div.stButton > button p {
        color: black !important;
    }

    /* Reduce streamlit element spacing */
    .element-container { margin-bottom: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        min-width: 180px;
        max-width: 180px;
        background: #1B3A6B;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    [data-testid="stSidebar"] div.stButton > button {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.2);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(255,255,255,0.1);
        border-color: white;
    }
    [data-testid="stSidebar"] div.stButton > button p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:white;text-align:center;margin-bottom:1rem;'>🛡️ Navigation</h2>", unsafe_allow_html=True)
    if st.button("🖥️ Infra", key="nav_infra"):
        st.session_state.page = "infra"
        st.session_state.selected = None
        st.experimental_rerun()
    if st.button("🌐 Network", key="nav_network"):
        st.session_state.page = "network"
        st.session_state.net_selected = None
        st.experimental_rerun()
    if st.button("📱 APS", key="nav_aps"):
        st.session_state.page = "aps"
        st.experimental_rerun()
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2);margin:0.5rem 0;'>", unsafe_allow_html=True)
    if st.button("📤 Upload Data", key="nav_upload"):
        st.session_state.page = "upload"
        st.experimental_rerun()

# ──────────────────────────────────────────────
# Upload Data Page
# ──────────────────────────────────────────────
if st.session_state.page == "upload":
    st.markdown("# 📤 Upload Excel Data")
    st.markdown("---")
    st.markdown("""
    Upload your Excel files here. The system will **automatically detect** whether 
    each file is for **Infra/Network (Data)** or **APS** — no matter what the file is named.
    """)

    # Show current data sources
    st.markdown("### 📁 Current Data Sources")
    col_a, col_b = st.columns(2)
    with col_a:
        data_p = get_data_path()
        if os.path.exists(data_p):
            src = "✅ Uploaded" if "uploads" in data_p else "📄 Default"
            st.success(f"**Data (Infra/Network):** {src}\n\n`{os.path.basename(data_p)}`")
        else:
            st.warning("**Data (Infra/Network):** No file found")
    with col_b:
        aps_p = get_aps_path()
        if os.path.exists(aps_p):
            src = "✅ Uploaded" if "uploads" in aps_p else "📄 Default"
            st.success(f"**APS:** {src}\n\n`{os.path.basename(aps_p)}`")
        else:
            st.warning("**APS:** No file found")

    st.markdown("---")

    # Auto-detect upload
    st.markdown("### 🔍 Smart Upload (Auto-Detect)")
    st.markdown("Upload any Excel file — the system will figure out if it's Data or APS.")
    auto_file = st.file_uploader(
        "Drop your Excel file here",
        type=["xlsx", "xls"],
        key="auto_upload",
    )
    if auto_file is not None:
        with st.spinner("Analyzing file..."):
            detected = detect_file_type(auto_file)

        if detected == "data":
            save_uploaded_file(auto_file, "data.xlsx")
            st.success(f"✅ **Detected as Data (Infra/Network)**  \nFile `{auto_file.name}` saved as data source.")
            st.cache_data.clear()
        elif detected == "aps":
            save_uploaded_file(auto_file, "aps.xlsx")
            st.success(f"✅ **Detected as APS**  \nFile `{auto_file.name}` saved as APS source.")
            st.cache_data.clear()
        else:
            st.warning(
                f"⚠️ Could not auto-detect file type for `{auto_file.name}`.  \n"
                "Please use the manual upload below."
            )

    st.markdown("---")

    # Manual upload as fallback
    st.markdown("### 📎 Manual Upload (choose target)")
    man_col1, man_col2 = st.columns(2)
    with man_col1:
        st.markdown("**Data (Infra/Network)**")
        data_file = st.file_uploader(
            "Upload Data Excel",
            type=["xlsx", "xls"],
            key="manual_data_upload",
        )
        if data_file is not None:
            save_uploaded_file(data_file, "data.xlsx")
            st.success(f"✅ `{data_file.name}` saved as Data source.")
            st.cache_data.clear()

    with man_col2:
        st.markdown("**APS**")
        aps_file = st.file_uploader(
            "Upload APS Excel",
            type=["xlsx", "xls"],
            key="manual_aps_upload",
        )
        if aps_file is not None:
            save_uploaded_file(aps_file, "aps.xlsx")
            st.success(f"✅ `{aps_file.name}` saved as APS source.")
            st.cache_data.clear()

    st.markdown("---")

    # Reset to defaults
    st.markdown("### 🔄 Reset to Default Files")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        if st.button("Reset Data to default", key="reset_data"):
            p = os.path.join(UPLOAD_DIR, "data.xlsx")
            if os.path.exists(p):
                os.remove(p)
            st.cache_data.clear()
            st.success("Data source reset to default `data.xlsx`.")
    with r_col2:
        if st.button("Reset APS to default", key="reset_aps"):
            p = os.path.join(UPLOAD_DIR, "aps.xlsx")
            if os.path.exists(p):
                os.remove(p)
            st.cache_data.clear()
            st.success("APS source reset to default `aps.xlsx`.")

    st.stop()

# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
@st.cache(ttl=60)
def load_data():
    data_path = get_data_path()
    sheet = find_sheet(data_path, "infra")
    df = pd.read_excel(data_path, sheet_name=sheet)
    df = select_and_rename_columns(df, INFRA_COLUMNS)
    if "Remediation_status" in df.columns:
        df["Remediation_status"] = normalize_remediation_status(df["Remediation_status"])
        df["Remediation_status"] = df["Remediation_status"].fillna("(No Status)")
    if "CTI overdue" in df.columns:
        df["CTI overdue"] = normalize_yes_no(df["CTI overdue"])
        df["CTI overdue"] = df["CTI overdue"].replace({"YES": "Overdue SLA", "NO": "Within SLA"})
    if "status as on last week date" in df.columns:
        df["status as on last week date"] = normalize_status_week(
            df["status as on last week date"], "Newly added")
    if "priority" in df.columns:
        df["priority"] = normalize_priority_infra(df["priority"])
    return df

@st.cache(ttl=60)
def load_network_data():
    try:
        data_path = get_data_path()
        sheet = find_sheet(data_path, "newtwork")
        df = pd.read_excel(data_path, sheet_name=sheet)
        df = select_and_rename_columns(df, NETWORK_COLUMNS)
        if "CTI overdue" in df.columns:
            df["CTI overdue"] = normalize_yes_no(df["CTI overdue"])
            df["CTI overdue"] = df["CTI overdue"].replace({"YES": "Overdue SLA", "NO": "Within SLA"})
        return df
    except Exception:
        return pd.DataFrame()

df = load_data()
df_network = load_network_data()

@st.cache(ttl=60)
def load_aps_data():
    try:
        aps_path = get_aps_path()
        sheet = find_sheet(aps_path, "aps")
        df = pd.read_excel(aps_path, sheet_name=sheet)
        df = select_and_rename_columns(df, APS_COLUMNS)
        if "Remediation_status" in df.columns:
            df["Remediation_status"] = normalize_remediation_status(df["Remediation_status"])
        if "Status as on last week date" in df.columns:
            df["Status as on last week date"] = normalize_status_week(
                df["Status as on last week date"], "Newly Added")
        if "Priority" in df.columns:
            df["Priority"] = normalize_priority_aps(df["Priority"])
        if "CTI Overdue" in df.columns:
            df["CTI Overdue"] = normalize_yes_no(df["CTI Overdue"], yes_val="Yes", no_val="No")
        return df
    except Exception:
        return pd.DataFrame()

df_aps = load_aps_data()

# ──────────────────────────────────────────────
# Precompute all datasets
# ──────────────────────────────────────────────
datasets = {
    "overall": {
        "data": df,
        "title": "Overall Vulnerability",
        "x": "owner",
        "color": "#667eea",
    },
    "newrem": {
        "data": df[df["Remediation_status"] == "(No Status)"],
        "title": "New Remediation",
        "x": "owner",
        "color": "#43e97b",
    },
    "newly": {
        "data": df[df["status as on last week date"] == "Newly added"],
        "title": "Newly Added Vulnerability",
        "x": "owner",
        "color": "#fa709a",
    },
    "sct": {
        "data": df[
            (df["Remediation_status"] == "(No Status)")
            & (df["priority"].isin(["PR3-other production asset ACT", "PR4-all other assets ACT"]))
        ],
        "title": "ACT Vulnerability",
        "x": "owner",
        "color": "#a18cd1",
    },
    "pdoeta": {
        "data": df[df["Remediation_status"] == "plan_defined_out_of_eta"],
        "title": "Plan Defined Out of ETA",
        "x": "owner",
        "color": "#f093fb",
    },
    "pnd": {
        "data": df[
            (df["Remediation_status"] == "(No Status)")
            & (df["owner"].notna())
            & (df["owner"].str.strip() != "")
        ],
        "title": "Plan Not Defined",
        "x": "status as on last week date",
        "color": "#ff9a9e",
    },
    "urap": {
        "data": df[df["Remediation_status"] == "under_risk_acceptance_process"],
        "title": "Under Risk Acceptance",
        "x": "owner",
        "color": "#4facfe",
    },
    "wo": {
        "data": df[df["Remediation_status"] == "wrong_owner"],
        "title": "Wrong Owner",
        "x": "owner",
        "color": "#f6d365",
    },
}

keys = list(datasets.keys())

# ──────────────────────────────────────────────
# Build a chart figure
# ──────────────────────────────────────────────
def make_chart(info, height=200, font_size=9, show_legend=False):
    data = info["data"]
    x_col = info["x"]
    if len(data) == 0:
        return None
    # Aggregate and limit to top N categories for large datasets
    N = 20  # Show top 20 categories, group rest as 'Other'
    grp = data.groupby([x_col, "CTI overdue"]).size().reset_index(name="Count")
    # Get top N categories by total count
    top_cats = grp.groupby(x_col)["Count"].sum().nlargest(N).index
    grp[x_col] = grp[x_col].where(grp[x_col].isin(top_cats), other="Other")
    grp = grp.groupby([x_col, "CTI overdue"]).sum().reset_index()
    # Remove text labels if too many bars
    show_text = grp[x_col].nunique() <= 30
    fig = px.bar(
        grp, x=x_col, y="Count", color="CTI overdue",
        barmode="group",
        color_discrete_map={"Overdue SLA": "#87CEEB", "Within SLA": "#1B3A6B"},
        text="Count" if show_text else None,
    )
    y_max = grp["Count"].max()
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=25, b=40),
        showlegend=show_legend,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(size=font_size), gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title="", tickfont=dict(size=font_size),
                   range=[0, y_max * 1.35] if y_max else None,
                   gridcolor="rgba(0,0,0,0.08)"),
        bargap=0.35,
    )
    if show_text:
        fig.update_traces(
            textposition="outside",
            textfont=dict(size=max(font_size, 11), color="black", family="Arial Black"),
            texttemplate="<b>%{text}</b>",
            cliponaxis=False,
        )
    return fig

# ──────────────────────────────────────────────
# Network Page
# ──────────────────────────────────────────────
if st.session_state.page == "network":
    st.markdown("# 🌐 Network Dashboard")
    st.markdown("---")

    if df_network.empty:
        st.warning("No 'Network' sheet found in data.xlsx. Please add a sheet named 'Network'.")
    else:
        # Columns are already renamed to canonical names by select_and_rename_columns
        owner_col = "owner" if "owner" in df_network.columns else None
        cti_col = "CTI overdue" if "CTI overdue" in df_network.columns else None
        criticity_col = "criticity" if "criticity" in df_network.columns else None

        # ── Button row: Show All + cards ──
        btn_cols = st.columns(3)
        with btn_cols[0]:
            if st.button("📊 Show All", key="net_show_all"):
                st.session_state.net_selected = None
        with btn_cols[1]:
            cti_total = len(df_network[df_network[cti_col].notna()]) if cti_col else 0
            if st.button(f"**{cti_total}**\nCTI Overdue", key="card_cti"):
                st.session_state.net_selected = "cti"
        with btn_cols[2]:
            crit_total = len(df_network[df_network[criticity_col].notna()]) if criticity_col else 0
            if st.button(f"**{crit_total}**\nCriticity", key="card_crit"):
                st.session_state.net_selected = "criticity"

        st.markdown("---")

        net_sel = st.session_state.net_selected

        if net_sel == "cti":
            # ── Single view: Owner vs CTI Overdue ──
            st.markdown("### Owner vs CTI Overdue")
            if owner_col and cti_col:
                grp1 = df_network.groupby([owner_col, cti_col]).size().reset_index(name="Count")
                fig1 = px.bar(
                    grp1, x=owner_col, y="Count", color=cti_col,
                    barmode="group",
                    color_discrete_map={"Overdue SLA": "#87CEEB", "Within SLA": "#1B3A6B"},
                    text="Count",
                )
                y_max1 = grp1["Count"].max()
                fig1.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=25, b=40),
                    showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="", tickfont=dict(size=11), gridcolor="rgba(0,0,0,0.05)"),
                    yaxis=dict(title="", tickfont=dict(size=11),
                               range=[0, y_max1 * 1.35] if y_max1 else None,
                               gridcolor="rgba(0,0,0,0.08)"),
                    bargap=0.35,
                )
                fig1.update_traces(
                    textposition="outside",
                    textfont=dict(size=11, color="black", family="Arial Black"),
                    texttemplate="<b>%{text}</b>",
                    cliponaxis=False,
                )
                st.plotly_chart(fig1, use_container_width=True, key="net_cti_single")
            else:
                st.info(f"Missing columns. Need 'owner' and 'CTI overdue'. Found: {list(df_network.columns)}")

        elif net_sel == "criticity":
            # ── Single view: Owner vs Criticity ──
            st.markdown("### Owner vs Criticity")
            if owner_col and criticity_col:
                grp2 = df_network.groupby([owner_col, criticity_col]).size().reset_index(name="Count")
                fig2 = px.bar(
                    grp2, x=owner_col, y="Count", color=criticity_col,
                    barmode="group",
                    text="Count",
                )
                y_max2 = grp2["Count"].max()
                fig2.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=25, b=40),
                    showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="", tickfont=dict(size=11), gridcolor="rgba(0,0,0,0.05)"),
                    yaxis=dict(title="", tickfont=dict(size=11),
                               range=[0, y_max2 * 1.35] if y_max2 else None,
                               gridcolor="rgba(0,0,0,0.08)"),
                    bargap=0.35,
                )
                fig2.update_traces(
                    textposition="outside",
                    textfont=dict(size=11, color="black", family="Arial Black"),
                    texttemplate="<b>%{text}</b>",
                    cliponaxis=False,
                )
                st.plotly_chart(fig2, use_container_width=True, key="net_crit_single")
            else:
                st.info(f"Missing columns. Need 'owner' and 'criticity'. Found: {list(df_network.columns)}")

        else:
            # ── Gallery view: both graphs side by side ──
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:#87CEEB;font-size:1.3rem;'>{cti_total}</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center;font-size:0.8rem;font-weight:600;color:#333;margin-bottom:0.2rem;'>Owner vs CTI Overdue</div>", unsafe_allow_html=True)
                if owner_col and cti_col:
                    grp1 = df_network.groupby([owner_col, cti_col]).size().reset_index(name="Count")
                    fig1 = px.bar(
                        grp1, x=owner_col, y="Count", color=cti_col,
                        barmode="group",
                        color_discrete_map={"Overdue SLA": "#87CEEB", "Within SLA": "#1B3A6B"},
                        text="Count",
                    )
                    y_max1 = grp1["Count"].max()
                    fig1.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=25, b=40),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="", tickfont=dict(size=9), gridcolor="rgba(0,0,0,0.05)"),
                        yaxis=dict(title="", tickfont=dict(size=9),
                                   range=[0, y_max1 * 1.35] if y_max1 else None,
                                   gridcolor="rgba(0,0,0,0.08)"),
                        bargap=0.35,
                    )
                    fig1.update_traces(
                        textposition="outside",
                        textfont=dict(size=11, color="black", family="Arial Black"),
                        texttemplate="<b>%{text}</b>",
                        cliponaxis=False,
                    )
                    st.plotly_chart(fig1, use_container_width=True, key="net_cti_thumb")
                else:
                    st.caption("No data")

            with chart_cols[1]:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:#667eea;font-size:1.3rem;'>{crit_total}</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center;font-size:0.8rem;font-weight:600;color:#333;margin-bottom:0.2rem;'>Owner vs Criticity</div>", unsafe_allow_html=True)
                if owner_col and criticity_col:
                    grp2 = df_network.groupby([owner_col, criticity_col]).size().reset_index(name="Count")
                    fig2 = px.bar(
                        grp2, x=owner_col, y="Count", color=criticity_col,
                        barmode="group",
                        text="Count",
                    )
                    y_max2 = grp2["Count"].max()
                    fig2.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=25, b=40),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="", tickfont=dict(size=9), gridcolor="rgba(0,0,0,0.05)"),
                        yaxis=dict(title="", tickfont=dict(size=9),
                                   range=[0, y_max2 * 1.35] if y_max2 else None,
                                   gridcolor="rgba(0,0,0,0.08)"),
                        bargap=0.35,
                    )
                    fig2.update_traces(
                        textposition="outside",
                        textfont=dict(size=11, color="black", family="Arial Black"),
                        texttemplate="<b>%{text}</b>",
                        cliponaxis=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True, key="net_crit_thumb")
                else:
                    st.caption("No data")

    st.stop()

# ──────────────────────────────────────────────
# APS Page
# ──────────────────────────────────────────────
if st.session_state.page == "aps":
    st.markdown("# 📱 APS Dashboard")
    st.markdown("---")
    if df_aps.empty:
        st.warning("No 'aps' sheet found in aps.xlsx. Please add a sheet named 'aps'.")
    else:
        df_newrem_aps = df_aps[df_aps["Remediation_status"].isna()]
        df_newly_aps = df_aps[df_aps["Status as on last week date"] == "Newly Added"]
        df_act_aps = df_aps[df_aps["Priority"].isin([
            "PR3 - Other Production Asset ACT",
            "PR4 - All Other Assets ACT",
            "PR4 - Other Production Asset ACT",
        ])]
        df_pdoeta_aps = df_aps[df_aps["Remediation_status"] == "plan_defined_out_of_eta"]
        df_pnd_aps = df_aps[df_aps["Remediation_status"].isna()]
        df_urap_aps = df_aps[df_aps["Remediation_status"] == "under_risk_acceptance_process"]
        df_wo_aps = df_aps[df_aps["Remediation_status"] == "wrong_owner"]

        aps_datasets = {
            "aps_overall": {
                "data": df_aps,
                "title": "Overall Vulnerabilities",
                "color": "#667eea",
            },
            "aps_newrem": {
                "data": df_newrem_aps,
                "title": "New Remediation",
                "color": "#43e97b",
            },
            "aps_newly": {
                "data": df_newly_aps,
                "title": "Newly Added Vulnerability",
                "color": "#fa709a",
            },
            "aps_act": {
                "data": df_act_aps,
                "title": "ACT Vulnerability",
                "color": "#a18cd1",
            },
            "aps_pdoeta": {
                "data": df_pdoeta_aps,
                "title": "Plan Defined Out of ETA",
                "color": "#f093fb",
            },
            "aps_pnd": {
                "data": df_pnd_aps,
                "title": "Plan Not Defined",
                "x": "Status as on present date",
                "color": "#ff9a9e",
            },
            "aps_urap": {
                "data": df_urap_aps,
                "title": "Under Risk Acceptance",
                "color": "#4facfe",
            },
            "aps_wo": {
                "data": df_wo_aps,
                "title": "Wrong Owner",
                "color": "#f6d365",
            },
        }
        aps_keys = list(aps_datasets.keys())

        def make_aps_chart(aps_data, height=200, font_size=9, show_legend=False, x_col="Metier"):
            if len(aps_data) == 0:
                return None
            grp = aps_data.groupby([x_col, "CTI Overdue"]).size().reset_index(name="Count")
            grp["CTI Overdue"] = grp["CTI Overdue"].replace({"Yes": "Overdue SLA", "No": "Within SLA"})
            fig = px.bar(
                grp, x=x_col, y="Count", color="CTI Overdue",
                barmode="group",
                color_discrete_map={"Overdue SLA": "#87CEEB", "Within SLA": "#1B3A6B"},
                text="Count",
            )
            y_max = grp["Count"].max()
            fig.update_layout(
                height=height,
                margin=dict(l=10, r=10, t=25, b=40),
                showlegend=show_legend,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="", tickfont=dict(size=font_size), gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(title="", tickfont=dict(size=font_size),
                           range=[0, y_max * 1.35] if y_max else None,
                           gridcolor="rgba(0,0,0,0.08)"),
                bargap=0.35,
            )
            fig.update_traces(
                textposition="outside",
                textfont=dict(size=max(font_size, 11), color="black", family="Arial Black"),
                texttemplate="<b>%{text}</b>",
                cliponaxis=False,
            )
            return fig

        # ── Card buttons row ──
        btn_cols = st.columns([1] + [1] * len(aps_keys))
        with btn_cols[0]:
            if st.button("📊 Show All", key="aps_show_all"):
                st.session_state.aps_selected = None
        for i, k in enumerate(aps_keys):
            info = aps_datasets[k]
            with btn_cols[i + 1]:
                if st.button(f"**{len(info['data'])}**\n{info['title']}", key=f"btn_{k}"):
                    st.session_state.aps_selected = k

        st.markdown("---")

        aps_sel = st.session_state.get("aps_selected", None)

        if aps_sel is not None:
            # ── Single chart view ──
            info = aps_datasets[aps_sel]
            st.markdown(f"### {info['title']}  ({len(info['data'])} items)")
            x_col = info.get("x", "Metier")
            fig = make_aps_chart(info["data"], height=450, font_size=12, show_legend=True, x_col=x_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"aps_single_{aps_sel}")
            else:
                st.info("No data for this category.")
        else:
            # ── Gallery view — 4 × 2 grid ──
            r1 = st.columns(4)
            for i, col in enumerate(r1):
                if i < len(aps_keys):
                    k = aps_keys[i]
                    info = aps_datasets[k]
                    with col:
                        st.markdown(f"<div style='text-align:center;font-weight:700;color:{info['color']};font-size:1.3rem;'>{len(info['data'])}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;font-size:0.8rem;font-weight:600;color:#333;margin-bottom:0.2rem;'>{info['title']}</div>", unsafe_allow_html=True)
                        x_col = info.get("x", "Metier")
                        fig = make_aps_chart(info["data"], x_col=x_col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"aps_thumb_{k}")
                        else:
                            st.caption("No data")

            if len(aps_keys) > 4:
                r2 = st.columns(4)
                for i, col in enumerate(r2):
                    idx = i + 4
                    if idx < len(aps_keys):
                        k = aps_keys[idx]
                        info = aps_datasets[k]
                        with col:
                            st.markdown(f"<div style='text-align:center;font-weight:700;color:{info['color']};font-size:1.3rem;'>{len(info['data'])}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='text-align:center;font-size:0.8rem;font-weight:600;color:#333;margin-bottom:0.2rem;'>{info['title']}</div>", unsafe_allow_html=True)
                            x_col = info.get("x", "Metier")
                            fig = make_aps_chart(info["data"], x_col=x_col)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"aps_thumb_{k}")
                            else:
                                st.caption("No data")

    st.stop()

# ──────────────────────────────────────────────
# Infra Page
# ──────────────────────────────────────────────
st.markdown("# 🛡️ Infra Dashboard")
st.markdown("---")

# ──────────────────────────────────────────────
# Card buttons row — always visible at top
# ──────────────────────────────────────────────
def on_card_click(key):
    if st.session_state.selected == key:
        st.session_state.selected = None  # toggle off
    else:
        st.session_state.selected = key

# Show All button + 8 card buttons
btn_cols = st.columns([1] + [1] * 8)

with btn_cols[0]:
    if st.button("📊 Show All", key="btn_all"):
        st.session_state.selected = None

for i, k in enumerate(keys):
    info = datasets[k]
    count = len(info["data"])
    with btn_cols[i + 1]:
        label = f"**{count}**\n{info['title']}"
        if st.button(label, key=f"btn_{k}"):
            on_card_click(k)

st.markdown("---")

# ──────────────────────────────────────────────
# Display: single chart or gallery
# ──────────────────────────────────────────────
selected = st.session_state.selected

if selected is not None:
    # ── Single chart view ──
    info = datasets[selected]
    st.markdown(f"### {info['title']}  ({len(info['data'])} items)")
    fig = make_chart(info, height=450, font_size=12, show_legend=True)
    if fig:
        st.plotly_chart(fig, width="stretch", key=f"single_{selected}")
    else:
        st.info("No data for this category.")
else:
    # ── Gallery view — 4 × 2 grid ──
    r1 = st.columns(4)
    for i, col in enumerate(r1):
        with col:
            info = datasets[keys[i]]
            st.markdown(f"<div style='text-align:center;font-weight:700;color:{info['color']};font-size:1.3rem;'>{len(info['data'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;font-size:0.8rem;font-weight:600;color:#333;margin-bottom:0.2rem;'>{info['title']}</div>", unsafe_allow_html=True)
            fig = make_chart(info)
            if fig:
                st.plotly_chart(fig, width="stretch", key=f"thumb_{keys[i]}")
            else:
                st.caption("No data")

    r2 = st.columns(4)
    for i, col in enumerate(r2):
        idx = i + 4
        with col:
            info = datasets[keys[idx]]
            st.markdown(f"<div style='text-align:center;font-weight:700;color:{info['color']};font-size:1.3rem;'>{len(info['data'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;font-size:0.8rem;font-weight:600;color:#333;margin-bottom:0.2rem;'>{info['title']}</div>", unsafe_allow_html=True)
            fig = make_chart(info)
            if fig:
                st.plotly_chart(fig, width="stretch", key=f"thumb_{keys[idx]}")
            else:
                st.caption("No data")

# ──────────────────────────────────────────────
# Legend
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:0.5rem;">
    <span style="display:inline-block; width:14px; height:14px; background:#87CEEB; border-radius:3px; vertical-align:middle;"></span>
    <span style="font-size:0.8rem; margin-right:1rem;"> Overdue SLA</span>
    <span style="display:inline-block; width:14px; height:14px; background:#1B3A6B; border-radius:3px; vertical-align:middle;"></span>
    <span style="font-size:0.8rem;"> Within SLA</span>
</div>
""", unsafe_allow_html=True)
