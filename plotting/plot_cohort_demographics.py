r"""Plot cohort membership cross-referenced against the demographics table.

The cohort lists live under ``Z:\cohorts`` as one-column TSV files with a
``subjectkey`` column. The demographics file lives at
``Z:\ds005237\phenotype\demos.tsv`` and is joined on the same subject key.

This script produces separate figures for:

* control vs anhedonic cohort sizes,
* age distributions per cohort,
* sex composition per cohort,
* a cross-reference summary table,
* anhedonia diagnosis and clinical distributions,
* hammer scan coverage across the subject pool.
"""

from __future__ import annotations

from io import StringIO
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.fs.get_script_name import get_name
from plotting import plot_config


COHORTS_DIR = Path(r"Z:\cohorts")
DEMOS_PATH = Path(r"Z:\ds005237\phenotype\demos.tsv")


def cohort_label_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("desc-"):
        stem = stem[len("desc-") :]
    if stem.endswith("_subjects"):
        stem = stem[: -len("_subjects")]
    return stem.replace("_", " ").title()


def load_cohort_memberships(cohorts_dir: Path) -> pd.DataFrame:
    records = []
    for cohort_file in sorted(cohorts_dir.glob("*.tsv")):
        cohort_name = cohort_label_from_path(cohort_file)
        cohort_df = pd.read_csv(cohort_file, sep="\t", dtype=str).fillna("")
        if "subjectkey" not in cohort_df.columns:
            raise ValueError(f"Missing subjectkey column in {cohort_file}")

        subjectkeys = (
            cohort_df["subjectkey"].astype(str).str.strip().str.upper().replace("", pd.NA)
        )
        for subjectkey in subjectkeys.dropna():
            records.append(
                {
                    "cohort": cohort_name,
                    "subjectkey": subjectkey,
                    "cohort_file": cohort_file.name,
                }
            )

    cohort_memberships = pd.DataFrame.from_records(records)
    if cohort_memberships.empty:
        return cohort_memberships

    cohort_memberships = cohort_memberships.drop_duplicates(
        subset=["cohort", "subjectkey"]
    )
    return cohort_memberships


def load_demographics(demos_path: Path) -> pd.DataFrame:
    # Read once from disk (important for slow network mounts), then decode with fallbacks.
    encodings_to_try = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    parse_layouts = (
        {"sep": "\t", "header": 1},
        {"sep": ",", "header": 1},
        {"sep": "\t", "header": 0},
        {"sep": ",", "header": 0},
    )
    try:
        raw_bytes = demos_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Unable to open demographics file at {demos_path}: {exc}") from exc

    demos = None
    decode_errors: list[str] = []
    parse_errors: list[str] = []
    for encoding in encodings_to_try:
        try:
            decoded_text = raw_bytes.decode(encoding)
            for layout in parse_layouts:
                parsed = pd.read_csv(StringIO(decoded_text), dtype=str, **layout)
                parsed.columns = parsed.columns.astype(str).str.strip()
                if "subjectkey" in parsed.columns:
                    demos = parsed
                    break
                parse_errors.append(
                    f"encoding={encoding}, sep={layout['sep']!r}, header={layout['header']}: "
                    f"missing subjectkey"
                )
            if demos is not None:
                break
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")
            continue
        except pd.errors.ParserError as exc:
            parse_errors.append(f"encoding={encoding}: {exc}")
        except Exception as exc:
            parse_errors.append(f"encoding={encoding}: unexpected parse error: {exc}")
        if demos is not None:
            break

    if demos is None:
        decode_summary = "; ".join(decode_errors) if decode_errors else "none"
        parse_summary = "; ".join(parse_errors[-6:]) if parse_errors else "none"
        raise RuntimeError(
            f"Unable to parse demographics file at {demos_path}. "
            f"Tried encodings {encodings_to_try} and layouts {parse_layouts}. "
            f"Decode issues: {decode_summary}. Parse issues: {parse_summary}"
        )

    if "subjectkey" not in demos.columns:
        raise ValueError(f"Missing subjectkey column in {demos_path}")

    demos = demos.copy()
    demos["subjectkey"] = demos["subjectkey"].astype(str).str.strip().str.upper()
    for column in ["src_subject_id", "sex", "Group", "Primary_Dx", "Non-Primary_Dx"]:
        if column in demos.columns:
            demos[column] = demos[column].astype("string").str.strip()
        else:
            demos[column] = pd.Series(pd.NA, index=demos.index, dtype="string")
    demos["Age"] = pd.to_numeric(
        demos.get("Age", pd.Series(pd.NA, index=demos.index, dtype="string")), errors="coerce"
    )
    demos.loc[demos["Age"] >= 200, "Age"] = pd.NA
    return demos


def merge_cohorts_with_demographics(
    cohort_memberships: pd.DataFrame, demos: pd.DataFrame
) -> pd.DataFrame:
    merged = cohort_memberships.merge(
        demos[["subjectkey", "src_subject_id", "Age", "sex", "Group", "Primary_Dx", "Non-Primary_Dx"]],
        on="subjectkey",
        how="left",
        validate="many_to_one",
    )
    sex_clean = merged["sex"].astype("string").str.upper().str.strip()
    sex_clean = sex_clean.replace({"": pd.NA, "NAN": pd.NA})
    merged["sex_clean"] = sex_clean.where(sex_clean.isin(["M", "F"]), other="Other")
    merged.loc[sex_clean.isna(), "sex_clean"] = pd.NA
    return merged


def normalize_category_series(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "NAN": pd.NA, "nan": pd.NA, "None": pd.NA})
    return cleaned.fillna(missing_label)


def prepare_category_counts(series: pd.Series, keep_top: int = 7, always_keep: tuple[str, ...] = ()) -> pd.Series:
    counts = normalize_category_series(series).value_counts()
    if counts.empty:
        return counts

    keep = []
    for label in always_keep:
        if label in counts.index and label not in keep:
            keep.append(label)

    for label in counts.index:
        if label in keep:
            continue
        keep.append(label)
        if len(keep) >= keep_top + len(always_keep):
            break

    plotted = counts.loc[keep].copy()
    other = counts.drop(keep, errors="ignore").sum()
    if other > 0:
        plotted.loc["Other"] = other

    plotted = plotted.sort_values(ascending=True)
    return plotted


def normalize_primary_diagnosis_series(series: pd.Series) -> pd.Series:
    """Normalize synonymous primary diagnosis labels before counting.

    Collapses common bipolar aliases used in the source data so the plot
    shows one category per disorder class.
    """
    cleaned = normalize_category_series(series, missing_label="999")
    mapping = {
        "BP1": "Bipolar 1",
        "BPI": "Bipolar 1",
        "BP2": "Bipolar 2",
        "BPII": "Bipolar 2",
    }
    return cleaned.replace(mapping)


def reorder_primary_counts(counts: pd.Series) -> pd.Series:
    """Reorder primary diagnosis counts so related disorders appear adjacent.

    This groups common depressive labels (e.g., MDD, Depression, Dysthymia)
    and bipolar labels (Bipolar 1/2) together, preserving descending
    order within each group and then appending remaining labels.
    """
    if counts.empty:
        return counts

    # preserve descending ordering within groups
    desc_index = list(counts.sort_values(ascending=False).index)

    groups_keywords = [
        ("depressive", ["depress", "mdd", "dysthym"]),
        ("bipolar", ["bipolar", "bp1", "bp2", "bpi", "bpii"]),
    ]

    ordered: list[str] = []
    used = set()

    for _group, keys in groups_keywords:
        for lbl in desc_index:
            low = str(lbl).lower()
            if lbl in used:
                continue
            for k in keys:
                if k in low:
                    ordered.append(lbl)
                    used.add(lbl)
                    break

    # Append remaining labels in descending frequency order
    for lbl in desc_index:
        if lbl not in used:
            ordered.append(lbl)
            used.add(lbl)

    # Return counts reindexed in the new order
    return counts.loc[ordered]


def format_disorder_label(label: str) -> str:
    # Display the sentinel '999' as a human-readable "No Dx" label.
    if label == "999":
        return "No Dx"
    # Keep canonical diagnosis labels and acronyms exactly as they were normalized.
    if label in {
        "Other",
        "Missing",
        "Panic Disorder",
        "GAD",
        "Past AUD",
        "Past CUD",
        "ADHD",
        "BED",
        "Mild AUD",
        "Past AN",
        "PTSD",
        "Past Social Anxiety",
    }:
        return label
    if label.isupper() or label.isdigit():
        return label
    return label.replace("_", " ").strip().title()


def expand_multi_label_counts(series: pd.Series, keep_top: int = 7, always_keep: tuple[str, ...] = ()) -> pd.Series:
    """Split multi-label cells (comma/semicolon/pipe) and return normalized counts.

    Treats '999' as No comorbidity and normalizes known abbreviations.
    """
    import re

    mapping = {
        "panid disorder": "Panic Disorder",
        "panic disorder": "Panic Disorder",
        "gad": "GAD",
        "past aud": "Past AUD",
        "past cud": "Past CUD",
        "adhd": "ADHD",
        "bed": "BED",
        "mild aud": "Mild AUD",
        "past an": "Past AN",
        "ptsd": "PTSD",
        "past social anxiety": "Past Social Anxiety",
        "past aud": "Past AUD",
        "past cud": "Past CUD",
    }

    tokens: list[str] = []
    no_comorbid_count = 0
    for val in series.dropna().astype(str):
        txt = val.strip()
        if txt == "" or txt == "999":
            no_comorbid_count += 1
            continue
        # split on comma/semicolon/vertical bar
        parts = re.split(r"[,;|]", txt)
        found_any = False
        for p in parts:
            token = p.strip()
            if token == "" or token == "999":
                continue
            key = token.lower()
            key = " ".join(key.split())
            # Only treat a token as the generic 'Other' when it is exactly 'other'.
            # Do not collapse phrases like 'other trauma- and stressor-related' to 'Other'.
            if key == "other":
                tokens.append("Other")
                continue
            if key in mapping:
                tokens.append(mapping[key])
            else:
                # if token looks like an acronym (all letters, len<=4) keep uppercase
                alpha = re.sub(r"[^A-Za-z]", "", key)
                if alpha.isalpha() and len(alpha) <= 4 and alpha.upper() == alpha.lower().upper():
                    tokens.append(alpha.upper())
                else:
                    tokens.append(key.title())
            found_any = True
        if not found_any:
            no_comorbid_count += 1

    counts = pd.Series(tokens).value_counts() if tokens else pd.Series(dtype=int)
    if no_comorbid_count > 0:
        counts["No comorbidity"] = counts.get("No comorbidity", 0) + no_comorbid_count

    if counts.empty:
        return counts

    # select top labels similar to prepare_category_counts
    keep = []
    for label in always_keep:
        if label in counts.index and label not in keep:
            keep.append(label)
    for label in counts.index:
        if label in keep:
            continue
        keep.append(label)
        if len(keep) >= keep_top + len(always_keep):
            break

    plotted = counts.loc[keep].copy()
    other = counts.drop(keep, errors="ignore").sum()
    if other > 0:
        plotted.loc["Other"] = other

    return plotted.sort_values(ascending=True)


def build_cohort_color_map(cohorts: list[str]) -> dict[str, str]:
    """Return a deterministic mapping cohort -> hex color.

    Prioritises mapping 'control(s)' to the first matlab color and
    'anhedonic'/'anhedonia' to the second matlab color when present,
    then assigns remaining colours from the project's palette.
    """
    base = plot_config.matlab_colors.copy()
    mapping: dict[str, str] = {}

    def pop_color(color: str) -> None:
        if color in base:
            base.remove(color)

    # strong preferences
    prefs = [("control", plot_config.matlab_colors[0]), ("anhedonic", plot_config.matlab_colors[1])]
    for pref_key, pref_color in prefs:
        for c in cohorts:
            if c.lower() == pref_key or pref_key in c.lower():
                mapping[c] = pref_color
        pop_color(pref_color)

    # assign remaining cohorts in order
    for c in cohorts:
        if c in mapping:
            continue
        if base:
            mapping[c] = base.pop(0)
        else:
            # fallback to tab10 colours if palette exhausted
            cmap = plt.get_cmap("tab10")
            mapping[c] = mpl.colors.to_hex(cmap(len(mapping) % cmap.N))

    return mapping


def plot_barh_counts(ax, counts: pd.Series, title: str, color: str, highlight: str | None = None) -> None:
    if counts.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    labels = counts.index.tolist()
    values = counts.to_numpy()
    y = np.arange(len(labels))
    colors = [color] * len(labels)
    if highlight and highlight in labels:
        colors[labels.index(highlight)] = plot_config.matlab_colors[1]

    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.invert_yaxis()
    # force integer ticks for subject counts
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    max_value = values.max() if len(values) else 0
    for ypos, value in zip(y, values):
        ax.text(value + max_value * 0.02, ypos, f"{int(value)}", va="center", fontsize=9)


def plot_doughnut_counts(
    ax,
    counts: pd.Series,
    title: str,
    highlight: str | None = None,
    center_text: str | None = None,
    use_legend: bool = False,
    exclude_colors: list[str] | None = None,
) -> None:
    if counts.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    labels = counts.index.tolist()
    values = counts.to_numpy()
    # Normalize exclude colors set
    exclude_set = {c.lower() for c in (exclude_colors or [])}

    # Choose a distinct qualitative palette for the doughnut. Prefer the
    # project's `matlab_colors` for small n, but filter out any excluded
    # colours (for example cohort colours reserved for controls).
    n = len(labels)
    base_palette = [c for c in plot_config.matlab_colors if c.lower() not in exclude_set]
    colors: list[str] = []
    if len(base_palette) >= n:
        colors = [base_palette[i % len(base_palette)] for i in range(n)]
    else:
        # Start with whatever remains from base_palette
        colors = base_palette.copy()
        # Fill remaining from tab20/tab10, excluding unwanted colours
        try:
            cmap = plt.get_cmap("tab20")
            cmap_colors = [mpl.colors.to_hex(cmap(i)) for i in range(cmap.N)]
        except Exception:
            cmap = plt.get_cmap("tab10")
            cmap_colors = [mpl.colors.to_hex(cmap(i)) for i in range(cmap.N)]
        cmap_colors = [c for c in cmap_colors if c.lower() not in exclude_set]
        for c in cmap_colors:
            if len(colors) >= n:
                break
            if c not in colors:
                colors.append(c)
        # If still short, repeat base_palette (last resort)
        while len(colors) < n and base_palette:
            colors.append(base_palette[len(colors) % len(base_palette)])
    if highlight and highlight in labels:
        # ensure the highlighted slice uses a consistent highlight color
        try:
            highlight_color = "#7f7f7f"
        except Exception:
            highlight_color = colors[0]
        colors[labels.index(highlight)] = highlight_color

    total = int(values.sum())
    display_labels = [format_disorder_label(label) for label in labels]

    if use_legend:
        wedges, _ = ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        legend_labels = []
        for display_label, value in zip(display_labels, values):
            pct = (value / total * 100.0) if total else 0.0
            legend_labels.append(f"{display_label} (n={int(value)}, {pct:.1f}%)")
        ax.legend(
            wedges,
            legend_labels,
            title=title,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
        )
    else:
        def autopct(pct: float) -> str:
            value = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n(n={value})" if pct >= 6 else ""

        wedges, texts, autotexts = ax.pie(
            values,
            labels=display_labels,
            colors=colors,
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": "white"},
            autopct=autopct,
            pctdistance=0.78,
            labeldistance=1.08,
            textprops={"fontsize": 9},
        )

    ax.set_aspect("equal")
    ax.set_title(title)
    if center_text is None:
        center_text = f"N={total}"
    ax.text(0, 0, center_text, ha="center", va="center", fontsize=11)


def plot_anhedonia_clinical_profile(merged: pd.DataFrame, out_dir: Path) -> list[Path]:
    anhedonia = merged.loc[merged["cohort"] == "Anhedonic"].copy()
    if anhedonia.empty:
        raise RuntimeError("No anhedonia cohort rows were found after cross-referencing the demographics file.")

    primary_counts = prepare_category_counts(
        normalize_primary_diagnosis_series(anhedonia["Primary_Dx"]),
        keep_top=8,
        always_keep=("999",),
    )
    # Order related diagnosis labels (depressive, bipolar, etc.) adjacent
    primary_counts = reorder_primary_counts(primary_counts.sort_values(ascending=False))
    comorbid_counts = expand_multi_label_counts(anhedonia["Non-Primary_Dx"], keep_top=6)
    group_counts = normalize_category_series(anhedonia["Group"], missing_label="Missing").value_counts()
    group_counts = group_counts.sort_values(ascending=True)
    genpop_count = int((normalize_category_series(anhedonia["Group"], missing_label="Missing") == "GenPop").sum())
    primary_no_dx = int((normalize_primary_diagnosis_series(anhedonia["Primary_Dx"]) == "999").sum())
    comorbid_no_dx = int((normalize_category_series(anhedonia["Non-Primary_Dx"], missing_label="999") == "999").sum())
    total = len(anhedonia)
    genpop_pct = (genpop_count / total * 100.0) if total else np.nan

    saved_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    # Build cohort colour map for the merged dataset and avoid reusing
    # control colours in this anhedonia-only doughnut.
    cohort_list = merged["cohort"].dropna().astype(str).unique().tolist()
    cohort_color_map = build_cohort_color_map(cohort_list)
    control_reserved = [plot_config.matlab_colors[0], plot_config.matlab_colors[1]]

    plot_doughnut_counts(
        ax,
        primary_counts,
        "Anhedonia: primary diagnosis distribution",
        highlight="999",
        center_text=f"No Dx (999)\n{primary_no_dx}",
        use_legend=True,
        exclude_colors=control_reserved,
    )
    primary_path = out_dir / f"{get_name()}_anhedonia_primary_diagnosis_doughnut.pdf"
    fig.savefig(primary_path, format="pdf")
    plt.close(fig)
    saved_paths.append(primary_path)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    comorbid_display = comorbid_counts.sort_values(ascending=False)
    if not comorbid_display.empty:
        labels = [format_disorder_label(label) for label in comorbid_display.index.tolist()]
        values = comorbid_display.to_numpy()
        y = np.arange(len(labels))
        colors = [plot_config.matlab_colors[5] if label != "999" else plot_config.matlab_colors[1] for label in comorbid_display.index.tolist()]
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Subjects")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("Anhedonia: non-primary diagnosis distribution")
        max_value = values.max() if len(values) else 0
        for ypos, value in zip(y, values):
            ax.text(value + max_value * 0.02, ypos, f"n={int(value)}", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
    comorbid_path = out_dir / f"{get_name()}_anhedonia_comorbid_diagnosis_barh.pdf"
    fig.savefig(comorbid_path, format="pdf")
    plt.close(fig)
    saved_paths.append(comorbid_path)

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    labels = group_counts.index.tolist()
    values = group_counts.to_numpy()
    # Use colours that are distinct from the cohort blue/orange mapping,
    # because this figure describes original group membership within the
    # anhedonia cohort rather than a control-vs-anhedonia comparison.
    colors = [plot_config.matlab_colors[4] if label == "GenPop" else plot_config.matlab_colors[6] for label in labels]
    ax.barh(np.arange(len(labels)), values, color=colors)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Anhedonia: original group membership")
    ax.set_xlabel("Subjects")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    max_value = values.max() if len(values) else 0
    for ypos, value in zip(np.arange(len(labels)), values):
        ax.text(value + max_value * 0.02, ypos, f"{int(value)}", va="center", fontsize=9)
    # ax.text(
    #     0.98,
    #     0.06,
    #     f"GenPop count: {genpop_count}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=10,
    #     bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    # )
    group_path = out_dir / f"{get_name()}_anhedonia_group_membership_barh.pdf"
    fig.savefig(group_path, format="pdf")
    plt.close(fig)
    saved_paths.append(group_path)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axis("off")
    summary_text = [
        f"Anhedonia subjects: {total}",
        f"GenPop subjects: {genpop_count} ({genpop_pct:.1f}%)" if total else "GenPop subjects: 0",
        f"No primary diagnosis (999): {primary_no_dx}",
        f"No comorbid diagnosis (999): {comorbid_no_dx}",
        "",
        "Diagnosis values are counted as exact strings from demos.tsv.",
    ]
    ax.text(0.02, 0.98, "\n".join(summary_text), ha="left", va="top", fontsize=12)
    ax.set_title("Anhedonia: clinical summary")
    summary_path = out_dir / f"{get_name()}_anhedonia_clinical_summary_text.pdf"
    fig.savefig(summary_path, format="pdf")
    plt.close(fig)
    saved_paths.append(summary_path)

    return saved_paths


def plot_hammer_scan_coverage(merged: pd.DataFrame, out_dir: Path) -> Path:
    total_subjects = merged["subjectkey"].nunique()
    hammer_subjects = merged.loc[merged["cohort"] == "Hammer", "subjectkey"].nunique()
    no_hammer_subjects = max(total_subjects - hammer_subjects, 0)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    plot_doughnut_counts(
        ax,
        pd.Series({"Hammer scan": hammer_subjects, "No hammer scan": no_hammer_subjects}),
        "Subjects with hammer scans",
        highlight="Hammer scan",
        center_text=f"{hammer_subjects}/{total_subjects}",
        use_legend=True,
    )
    out_path = out_dir / f"{get_name()}_hammer_scan_coverage_doughnut.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def build_summary_table(merged: pd.DataFrame) -> pd.DataFrame:
    summary = (
        merged.groupby("cohort", dropna=False)
        .agg(
            n_subjects=("subjectkey", "nunique"),
            age_n=("Age", lambda s: s.notna().sum()),
            age_mean=("Age", "mean"),
            age_median=("Age", "median"),
            female_count=("sex_clean", lambda s: (s == "F").sum()),
            male_count=("sex_clean", lambda s: (s == "M").sum()),
            other_sex_count=("sex_clean", lambda s: s.isna().sum() + (s == "Other").sum()),
        )
        .reset_index()
    )
    summary["female_pct"] = np.where(
        summary["age_n"] > 0, summary["female_count"] / summary["age_n"] * 100.0, np.nan
    )
    return summary


def plot_cohort_demographics(merged: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    cohort_summary = summary.loc[
        summary["cohort"].astype(str).str.casefold().isin({"controls", "anhedonic"})
    ].copy()
    if cohort_summary.empty:
        cohort_summary = summary.loc[summary["cohort"].str.casefold() != "hammer"].copy()
    cohort_order = cohort_summary.sort_values(["n_subjects", "cohort"], ascending=[False, True])["cohort"].tolist()
    plot_merged = merged.loc[merged["cohort"].isin(cohort_order)].copy()
    plot_merged["cohort"] = pd.Categorical(plot_merged["cohort"], categories=cohort_order, ordered=True)
    summary = cohort_summary.set_index("cohort").loc[cohort_order].reset_index()
    saved_paths: list[Path] = []

    # Membership counts.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    x = np.arange(len(cohort_order))
    counts = summary["n_subjects"].to_numpy()
    cohort_color_map = build_cohort_color_map(cohort_order)
    bar_colors = [cohort_color_map.get(cohort, plot_config.matlab_colors[0]) for cohort in cohort_order]
    ax.bar(x, counts, color=bar_colors)
    for xpos, total in zip(x, counts):
        ax.text(xpos, total + max(counts) * 0.01, f"n={int(total)}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_order, rotation=20, ha="right")
    ax.set_ylabel("Subjects")
    # ensure subject counts use integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Control vs anhedonic cohort sizes")
    membership_path = out_dir / f"{get_name()}_cohort_membership_counts.pdf"
    fig.savefig(membership_path, format="pdf")
    plt.close(fig)
    saved_paths.append(membership_path)

    # Age distributions.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    age_data = [plot_merged.loc[plot_merged["cohort"] == cohort, "Age"].dropna().to_numpy() for cohort in cohort_order]
    max_age = max((float(np.max(values)) for values in age_data if len(values)), default=np.nan)
    box = ax.boxplot(
        age_data,
        tick_labels=cohort_order,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    for patch, cohort in zip(box["boxes"], cohort_order):
        color = cohort_color_map.get(cohort, plot_config.matlab_colors[0])
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in box["medians"]:
        median.set_color("black")
    # Add legend for median (line) and mean (triangle marker)
    handles = [
        Line2D([0], [0], color="black", lw=1.5, label="Median"),
        Line2D([0], [0], marker="^", color="black", linestyle="", label="Mean"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.set_ylabel("Age (months)")
    ax.set_title("Age by cohort")
    if np.isfinite(max_age):
        ax.set_ylim(0, int(np.ceil(max_age / 10.0) * 10))
    ax.tick_params(axis="x", rotation=20)
    age_path = out_dir / f"{get_name()}_cohort_age_boxplot.pdf"
    fig.savefig(age_path, format="pdf")
    plt.close(fig)
    saved_paths.append(age_path)

    # Sex composition.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    sex_counts = (
        plot_merged.groupby(["cohort", "sex_clean"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=cohort_order, fill_value=0)
    )
    # Use baby-blue for males and light-pink for females for clearer gender cues
    male_blue = "#5FC0EC"  # baby blue
    female_pink = "#FF637B"  # light pink
    for label, color in [("F", female_pink), ("M", male_blue), ("Other", "#999999")]:
        values = sex_counts[label] if label in sex_counts.columns else pd.Series(0, index=sex_counts.index)
        if label == "F":
            bottoms = np.zeros(len(sex_counts))
        elif label == "M":
            bottoms = sex_counts[[c for c in ["F"] if c in sex_counts.columns]].sum(axis=1).to_numpy()
        else:
            bottoms = sex_counts[[c for c in ["F", "M"] if c in sex_counts.columns]].sum(axis=1).to_numpy()
        ax.bar(cohort_order, values.to_numpy(), bottom=bottoms, label=label, color=color)
    ax.set_ylabel("Subjects")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Sex composition")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Sex", frameon=False)
    sex_path = out_dir / f"{get_name()}_cohort_sex_composition_stacked_bar.pdf"
    fig.savefig(sex_path, format="pdf")
    plt.close(fig)
    saved_paths.append(sex_path)

    # Summary table.
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    ax.axis("off")
    table_df = summary.copy()
    table_df["age_mean"] = table_df["age_mean"].round(1)
    table_df["age_median"] = table_df["age_median"].round(1)
    table_df["female_pct"] = table_df["female_pct"].round(1)
    display_cols = [
        "cohort",
        "n_subjects",
        "age_mean",
        "age_median",
        "female_pct",
    ]
    table = ax.table(
        cellText=table_df[display_cols].values,
        colLabels=["Cohort", "N", "Age mean", "Age median", "Female %"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.35)
    ax.set_title("Control vs anhedonic cross-reference summary", pad=12)
    table_path = out_dir / f"{get_name()}_cohort_cross_reference_table.pdf"
    fig.savefig(table_path, format="pdf")
    plt.close(fig)
    saved_paths.append(table_path)

    return saved_paths


def main() -> None:
    out_dir = plot_config.get_figs_output_dir()
    cohort_memberships = load_cohort_memberships(COHORTS_DIR)
    if cohort_memberships.empty:
        raise RuntimeError(f"No cohort TSV files found in {COHORTS_DIR}")

    demos = load_demographics(DEMOS_PATH)
    merged = merge_cohorts_with_demographics(cohort_memberships, demos)
    merged = merged.loc[merged["src_subject_id"].notna()].copy()
    if merged.empty:
        raise RuntimeError("No matched subjects remained after filtering missing demographics rows.")
    summary = build_summary_table(merged)

    cohort_paths = plot_cohort_demographics(merged, summary, out_dir)
    clinical_paths = plot_anhedonia_clinical_profile(merged, out_dir)
    hammer_path = plot_hammer_scan_coverage(merged, out_dir)
    for path in cohort_paths + clinical_paths + [hammer_path]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()