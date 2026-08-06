"""
Render the JSON outputs from gap_detector.py and topic_scan.py into a
single readable Markdown gap report.

Takes the most recent gap_report_*.json and topic_scan_*.json from the
results/ folder and produces gap_report.md at the project root.
"""
import os
import json
import glob
from datetime import datetime


RESULTS_DIR = "results"
OUTPUT_PATH = "gap_report.md"


def load_latest(pattern: str) -> dict | None:
    """Load the most recent file matching a glob pattern from results/."""
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not matches:
        return None
    with open(matches[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def render_section_findings(section: dict) -> list[str]:
    """Render one policy section's analysis as Markdown lines."""
    out = []
    num = section["section_number"]
    title = section["section_title"]
    analysis = section["analysis"]

    out.append(f"### Section {num} — {title}\n")

    summary = analysis.get("section_summary", "").strip()
    if summary:
        out.append(f"*{summary}*\n")

    gaps = analysis.get("gaps", [])
    partial = analysis.get("partial", [])
    covered = analysis.get("covered", [])

    if not (gaps or partial or covered):
        out.append("_No findings for this section._\n")
        return out

    if gaps:
        out.append(f"**Gaps ({len(gaps)}):**\n")
        for g in gaps:
            severity = g.get("severity", "?")
            req = g.get("requirement", "").strip()
            why = g.get("why_gap", "").strip()
            quote = g.get("hkma_quote", "").strip()
            sources = g.get("source_chunks", [])
            out.append(f"- **[{severity}] {req}**")
            if why:
                out.append(f"  - _Why this is a gap:_ {why}")
            if quote:
                out.append(f"  - _HKMA:_ \u201c{quote}\u201d")
            if sources:
                out.append(f"  - _Source:_ `{', '.join(sources[:3])}`")
        out.append("")

    if partial:
        out.append(f"**Partial coverage ({len(partial)}):**\n")
        for p in partial:
            req = p.get("requirement", "").strip()
            covered_part = p.get("what_is_covered", "").strip()
            missing_part = p.get("what_is_missing", "").strip()
            sources = p.get("source_chunks", [])
            out.append(f"- **{req}**")
            if covered_part:
                out.append(f"  - _Addressed:_ {covered_part}")
            if missing_part:
                out.append(f"  - _Missing:_ {missing_part}")
            if sources:
                out.append(f"  - _Source:_ `{', '.join(sources[:3])}`")
        out.append("")

    if covered:
        out.append(f"**Covered ({len(covered)}):**\n")
        for c in covered:
            req = c.get("requirement", "").strip()
            evidence = c.get("policy_evidence", "").strip()
            out.append(f"- {req}")
            if evidence:
                out.append(f"  - _Policy:_ {evidence}")
        out.append("")

    return out


def render_topic_scan(scan: dict) -> list[str]:
    """Render the silent-omission scan as Markdown."""
    out = []
    findings = scan.get("findings", [])
    omitted = [f for f in findings if f["status"] == "OMITTED"]

    out.append(f"## Silent-Omission Scan\n")
    out.append(
        f"Scanned {len(findings)} distinct regulatory topics from the HKMA corpus "
        f"and checked whether each is addressed anywhere in the policy. "
        f"Threshold for omission: vector distance > {scan.get('omission_threshold')}.\n"
    )

    if not omitted:
        out.append("_No topics silently omitted._\n")
        return out

    out.append(f"### Topics silently omitted ({len(omitted)}):\n")
    for f in omitted:
        out.append(f"- **{f['topic']}**  (distance {f['closest_policy_distance']:.2f})")
        desc = f.get("description", "").strip()
        if desc:
            out.append(f"  - _Regulatory expectation:_ {desc}")
        sources = f.get("example_sources", [])
        if sources:
            out.append(f"  - _Source circulars:_ `{', '.join(sources[:3])}`")
    out.append("")

    out.append(f"### Topics found in the policy ({len(findings) - len(omitted)}):\n")
    for f in findings:
        if f["status"] == "OMITTED":
            continue
        out.append(f"- {f['topic']}  (distance {f['closest_policy_distance']:.2f})")
    out.append("")
    return out


def main():
    gap = load_latest("gap_report_*.json")
    scan = load_latest("topic_scan_*.json")

    if gap is None and scan is None:
        print("No results found. Run gap_detector.py and topic_scan.py first.")
        return

    lines = []
    lines.append(f"# HKMA AML/CFT Compliance Gap Report\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    if gap:
        lines.append(f"_Policy analysed: `{gap['policy_file']}`_")
        lines.append(f"_Model: `{gap['model']}`_\n")

    if gap:
        # Headline counts
        total_covered = sum(len(s["analysis"].get("covered", [])) for s in gap["sections"])
        total_gaps = sum(len(s["analysis"].get("gaps", [])) for s in gap["sections"])
        total_partial = sum(len(s["analysis"].get("partial", [])) for s in gap["sections"])
        high_gaps = sum(
            1 for s in gap["sections"]
            for g in s["analysis"].get("gaps", [])
            if g.get("severity") == "HIGH"
        )

        lines.append("## Summary\n")
        lines.append(f"- **Policy sections analysed:** {len(gap['sections'])}")
        lines.append(f"- **Requirements covered:** {total_covered}")
        lines.append(f"- **Gaps identified:** {total_gaps}  (of which HIGH severity: {high_gaps})")
        lines.append(f"- **Partial coverage areas:** {total_partial}")
        if scan:
            omitted = [f for f in scan["findings"] if f["status"] == "OMITTED"]
            lines.append(f"- **Silent omissions (topic scan):** {len(omitted)}")
        lines.append("")

        lines.append("## Section-by-Section Findings\n")
        for section in gap["sections"]:
            lines.extend(render_section_findings(section))

    if scan:
        lines.extend(render_topic_scan(scan))

    lines.append("---\n")
    lines.append(
        "_This report was produced by an automated RAG pipeline over HKMA AML/CFT "
        "circulars. It is intended as a first-pass triage tool for compliance analysts, "
        "not a final compliance determination. All findings should be reviewed by a "
        "qualified compliance professional. See README for methodology and known "
        "limitations._"
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUTPUT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()