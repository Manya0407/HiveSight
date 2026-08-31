import argparse
import gzip
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from debug import format_debug_result, format_log_summary
from phase2.debugger import HiveDebugger


TEXT_SUFFIXES = {
    ".log", ".out", ".txt", ".err", ".stderr", ".stdout",
}


@dataclass
class BundleLog:
    path: Path
    role: str
    node: str
    size_bytes: int
    compressed_source: Path | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quasar diagnostic bundle analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  /opt/homebrew/bin/python3.11 diag_bundle.py --bundle "QUASAR_DIAG_LOGS (2).zip"
  /opt/homebrew/bin/python3.11 diag_bundle.py --bundle "QUASAR_DIAG_LOGS (2).zip" --action discover
  /opt/homebrew/bin/python3.11 diag_bundle.py --bundle "QUASAR_DIAG_LOGS (2).zip" --action full --no-prompts
        """,
    )
    parser.add_argument("--bundle", required=True, help="Path to Quasar diag bundle zip")
    parser.add_argument(
        "--extract-dir",
        default="extracted_diag_bundles",
        help="Directory used for recursive extraction",
    )
    parser.add_argument(
        "--reports-dir",
        default="diag_bundle_reports",
        help="Directory where output reports are saved",
    )
    parser.add_argument(
        "--action",
        choices=[
            "interactive",
            "discover",
            "debug-hive",
            "summarize-hive",
            "summarize-rest",
            "full",
        ],
        default="interactive",
        help="Action to run after extraction",
    )
    parser.add_argument(
        "--max-rest",
        type=int,
        default=10,
        help="Maximum non-HS2/HMS logs to summarize in rest/full modes",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract even if the target extraction directory exists",
    )
    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="Do not ask interactive follow-up questions",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Debugger config path",
    )

    args = parser.parse_args()
    bundle = Path(args.bundle).expanduser().resolve()
    if not bundle.exists():
        print(f"Bundle not found: {bundle}")
        sys.exit(1)

    extract_root = prepare_bundle(bundle, Path(args.extract_dir), args.force_extract)
    hs2_hms_logs, other_logs = discover_logs(extract_root)

    action = args.action
    if action == "interactive" and args.no_prompts:
        action = "discover"
    if action == "interactive":
        action = ask_action(hs2_hms_logs, other_logs)

    report_dir = Path(args.reports_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)

    if action == "discover":
        print_inventory(hs2_hms_logs, other_logs)
        save_inventory(report_dir, hs2_hms_logs, other_logs)
        return

    print_inventory(hs2_hms_logs, other_logs)
    debugger = HiveDebugger(config_path=args.config)

    if action in {"debug-hive", "full"}:
        run_debugger_on_hive_logs(debugger, hs2_hms_logs, report_dir)

    if action in {"summarize-hive"}:
        run_summarizer(debugger, hs2_hms_logs, report_dir, label="hive")

    if action in {"summarize-rest", "full"}:
        selected = select_rest_logs(other_logs, args.max_rest)
        run_summarizer(debugger, selected, report_dir, label="other")

    print(f"\nReports saved under: {report_dir}")


def prepare_bundle(bundle: Path, base_extract_dir: Path, force: bool) -> Path:
    """Extract the top-level bundle and any nested zip files."""
    stem = safe_name(bundle.stem)
    extract_root = base_extract_dir / stem

    if force and extract_root.exists():
        shutil.rmtree(extract_root)

    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"Extracting bundle: {bundle}")
        safe_extract_zip(bundle, extract_root)
    else:
        print(f"Using existing extracted bundle: {extract_root}")

    extract_nested_zips(extract_root)
    decompress_gz_logs(extract_root)
    return extract_root


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip while preventing path traversal."""
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if Path(member.filename).is_absolute():
                print(f"  Skipping unsafe absolute zip entry: {member.filename}")
                continue
            member_path = target_dir / member.filename
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(target_dir)):
                print(f"  Skipping unsafe zip entry: {member.filename}")
                continue
            if member.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(resolved, "wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_nested_zips(root: Path) -> None:
    """Recursively extract nested zip files into sibling *_unzipped folders."""
    seen = set()
    while True:
        nested = [
            path for path in root.rglob("*.zip")
            if path.resolve() not in seen
            and not is_inside_extracted_zip(path)
            and should_extract_nested_zip(path)
        ]
        if not nested:
            return

        for zip_path in nested:
            seen.add(zip_path.resolve())
            target = zip_path.with_suffix("")
            target = target.parent / f"{target.name}_unzipped"
            if target.exists():
                continue
            print(f"Extracting nested zip: {zip_path.name}")
            target.mkdir(parents=True, exist_ok=True)
            try:
                safe_extract_zip(zip_path, target)
            except zipfile.BadZipFile:
                print(f"  Skipping invalid zip: {zip_path}")


def is_inside_extracted_zip(path: Path) -> bool:
    return any(part.endswith("_unzipped") for part in path.parts)


def should_extract_nested_zip(path: Path) -> bool:
    """Keep extraction scoped to service/support bundles, not config/profile zips."""
    name = path.name.lower()
    return (
        name.endswith(".logs.zip")
        or name.startswith("diag-bundle-")
        or name.startswith("yarn-bundle-")
    )


def decompress_gz_logs(root: Path) -> None:
    """Create plain-text copies for .gz log files so the debugger can read them."""
    for gz_path in root.rglob("*.gz"):
        if not is_probable_log(gz_path):
            continue
        output_path = gz_path.with_suffix("")
        if output_path.exists():
            continue
        try:
            with gzip.open(gz_path, "rb") as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except OSError:
            print(f"  Skipping invalid gzip: {gz_path}")


def discover_logs(root: Path) -> tuple[list[BundleLog], list[BundleLog]]:
    """Find HS2/HMS logs and other summarizable text logs."""
    hs2_hms = []
    other = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix == ".gz":
            continue
        if "_unzipped" not in path.parts and path.suffix == ".zip":
            continue
        if not is_probable_log(path):
            continue

        role = classify_role(path)
        log = BundleLog(
            path=path,
            role=role,
            node=infer_node(path),
            size_bytes=path.stat().st_size,
            compressed_source=path.with_suffix(path.suffix + ".gz")
            if path.with_suffix(path.suffix + ".gz").exists()
            else None,
        )

        if role in {"HS2", "HMS"}:
            hs2_hms.append(log)
        else:
            other.append(log)

    hs2_hms = dedupe_hive_logs(hs2_hms)
    hs2_hms.sort(key=lambda item: (item.role, item.node, str(item.path)))
    other.sort(key=lambda item: (-item.size_bytes, str(item.path)))
    return hs2_hms, other


def dedupe_hive_logs(logs: list[BundleLog]) -> list[BundleLog]:
    """Collapse duplicate service logs present in both node and support bundles."""
    selected = {}
    for log in logs:
        key = (log.role, log.node, log.path.name)
        current = selected.get(key)
        if current is None:
            selected[key] = log
            continue
        if hive_log_preference(log) > hive_log_preference(current):
            selected[key] = log
    return list(selected.values())


def hive_log_preference(log: BundleLog) -> tuple[int, int]:
    """Prefer node-archive logs, then larger copies when duplicates exist."""
    path_text = str(log.path)
    node_archive_score = 1 if "debugging_files_" in path_text else 0
    return (node_archive_score, log.size_bytes)


def is_probable_log(path: Path) -> bool:
    name = path.name.lower()
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".gz" in suffixes:
        suffixes.remove(".gz")
    if any(suffix in TEXT_SUFFIXES for suffix in suffixes):
        return True
    return any(term in name for term in [".log.", "stdout", "stderr"])


def classify_role(path: Path) -> str:
    if not is_primary_hive_service_log(path):
        return "OTHER"

    text = path.name.lower()
    if any(term in text for term in ["hiveserver2", "hive-server2", "hive_on_tez"]):
        return "HS2"
    if any(term in text for term in ["hivemetastore", "metastore", "hms"]):
        return "HMS"
    return "OTHER"


def is_primary_hive_service_log(path: Path) -> bool:
    """Return True for main CM service logs, not process configs/stdout noise."""
    name = path.name.lower()
    if not name.startswith("hadoop-cmf-"):
        return False
    if ".log.out" not in name:
        return False
    return "hiveserver2" in name or "hivemetastore" in name


def infer_node(path: Path) -> str:
    text = str(path)
    match = re.search(r"(quasar-[A-Za-z0-9-]+\.vpc\.eng\.cloudera\.com)", text)
    if match:
        return match.group(1)
    match = re.search(r"(quasar-[A-Za-z0-9-]+)", text)
    if match:
        return match.group(1)
    return "unknown-node"


def ask_action(hs2_hms_logs: list[BundleLog], other_logs: list[BundleLog]) -> str:
    print_inventory(hs2_hms_logs, other_logs)
    print(
        "\nWhat would you like to do?\n"
        "  1. Discover/list logs only\n"
        "  2. Debug HS2/HMS logs\n"
        "  3. Summarize HS2/HMS logs\n"
        "  4. Summarize other logs\n"
        "  5. Full run: debug HS2/HMS and summarize other logs\n"
    )
    choice = input("Choose 1-5 [1]: ").strip() or "1"
    return {
        "1": "discover",
        "2": "debug-hive",
        "3": "summarize-hive",
        "4": "summarize-rest",
        "5": "full",
    }.get(choice, "discover")


def print_inventory(hs2_hms_logs: list[BundleLog], other_logs: list[BundleLog]) -> None:
    print("\nDISCOVERED LOGS")
    print("=" * 60)
    print(f"HS2/HMS logs: {len(hs2_hms_logs)}")
    for idx, log in enumerate(hs2_hms_logs[:50], 1):
        print(f"  [{idx}] {log.role} | {log.node} | {short_path(log.path)}")
    if len(hs2_hms_logs) > 50:
        print(f"  ... {len(hs2_hms_logs) - 50} more")
    print(f"Other summarizable logs: {len(other_logs)}")
    for idx, log in enumerate(other_logs[:20], 1):
        print(f"  [{idx}] {log.role} | {format_size(log.size_bytes)} | {short_path(log.path)}")
    if len(other_logs) > 20:
        print(f"  ... {len(other_logs) - 20} more")


def save_inventory(
    report_dir: Path,
    hs2_hms_logs: list[BundleLog],
    other_logs: list[BundleLog],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["DISCOVERED LOGS", "=" * 60, f"HS2/HMS logs: {len(hs2_hms_logs)}"]
    for log in hs2_hms_logs:
        lines.append(f"{log.role}\t{log.node}\t{log.path}")
    lines.append(f"\nOther summarizable logs: {len(other_logs)}")
    for log in other_logs:
        lines.append(f"{log.role}\t{format_size(log.size_bytes)}\t{log.path}")
    output = "\n".join(lines) + "\n"
    path = report_dir / "bundle_inventory.txt"
    path.write_text(output, encoding="utf-8")
    print(f"\nInventory saved to: {path}")


def run_debugger_on_hive_logs(
    debugger: HiveDebugger,
    logs: Iterable[BundleLog],
    report_dir: Path,
) -> None:
    for log in logs:
        print(f"\nDebugging {log.role} log: {short_path(log.path)}")
        diagnosis = debugger.debug_log_file(str(log.path))
        output = format_debug_result(diagnosis)
        report_path = report_dir / f"{safe_name(log.role)}_{safe_name(log.node)}_debug.txt"
        report_path.write_text(output + "\n", encoding="utf-8")
        print(output)
        print(f"Saved debug report: {report_path}")


def run_summarizer(
    debugger: HiveDebugger,
    logs: Iterable[BundleLog],
    report_dir: Path,
    label: str,
) -> None:
    for idx, log in enumerate(logs, 1):
        service = infer_service(log.path)
        print(f"\nSummarizing {label} log {idx} ({service}): {short_path(log.path)}")
        summary = debugger.summarize_log_file(str(log.path))
        output = format_log_summary(summary)
        if label == "other":
            output = output.replace("HIVE LOG SUMMARY", "SERVICE LOG SUMMARY", 1)
            output = output.replace(
                "IMPORTANT TESTS / QUERY IDS",
                "IMPORTANT IDS / APPLICATIONS",
                1,
            )
        report_path = (
            report_dir
            / (
                f"{safe_name(label)}_{idx:02d}_{safe_name(service)}_"
                f"{safe_name(log.node)}_summary.txt"
            )
        )
        report_path.write_text(output + "\n", encoding="utf-8")
        print(output)
        print(f"Saved summary report: {report_path}")


def select_rest_logs(other_logs: list[BundleLog], max_count: int) -> list[BundleLog]:
    """Pick important platform service logs before optional noisy logs."""
    selected = []
    seen = set()

    # These are the services we intentionally want to cover in bundle summaries.
    priority = [
        "YARN_RESOURCEMANAGER",
        "YARN_NODEMANAGER",
        "HDFS_NAMENODE",
        "HDFS_DATANODE",
        "HBASE_MASTER",
        "HBASE_REGIONSERVER",
        "TEZ",
        "ZOOKEEPER",
        "KAFKA",
        "ATLAS",
        "RANGER",
        "SOLR",
        "KNOX",
        "HTTPFS",
    ]

    grouped: dict[str, list[BundleLog]] = {service: [] for service in priority}
    for log in other_logs:
        service = infer_service(log.path)
        if service in grouped:
            grouped[service].append(log)

    for service in priority:
        for log in sorted(grouped[service], key=service_log_score, reverse=True):
            key = (service, log.node)
            if key in seen:
                continue
            selected.append(log)
            seen.add(key)
            break
        if len(selected) >= max_count:
            return selected

    fallback = sorted(other_logs, key=service_log_score, reverse=True)
    for log in fallback:
        key = (infer_service(log.path), log.node, log.path.name)
        if key in seen:
            continue
        if not has_failure_signal(log.path):
            continue
        selected.append(log)
        seen.add(key)
        if len(selected) >= max_count:
            break

    return selected


def infer_service(path: Path) -> str:
    text = str(path).lower()
    name = path.name.lower()
    if "resourcemanager" in text:
        return "YARN_RESOURCEMANAGER"
    if "nodemanager" in text:
        return "YARN_NODEMANAGER"
    if "namenode" in text:
        return "HDFS_NAMENODE"
    if "datanode" in text:
        return "HDFS_DATANODE"
    if "hbase" in text and "master" in text:
        return "HBASE_MASTER"
    if "hbase" in text and "regionserver" in text:
        return "HBASE_REGIONSERVER"
    if "tez" in text:
        return "TEZ"
    if "zookeeper" in text:
        return "ZOOKEEPER"
    if "kafka" in text:
        return "KAFKA"
    if "atlas" in text:
        return "ATLAS"
    if "ranger" in text:
        return "RANGER"
    if "solr" in text:
        return "SOLR"
    if "knox" in text:
        return "KNOX"
    if "httpfs" in text:
        return "HTTPFS"
    if "hadoop-cmf-" in name:
        match = re.search(r"hadoop-cmf-([A-Z0-9_]+)-", name, re.I)
        if match:
            return match.group(1).upper()
    return "OTHER"


def service_log_score(log: BundleLog) -> tuple[int, int, int]:
    text = str(log.path).lower()
    primary_cm_log = (
        10 if log.path.name.lower().startswith("hadoop-cmf-")
        and ".log.out" in log.path.name.lower()
        else 0
    )
    diagnostics_log = 5 if "diagnostics" in text else 0
    failure_signal = 20 if has_failure_signal(log.path) else 0
    return (failure_signal + primary_cm_log + diagnostics_log, log.size_bytes, -len(text))


def has_failure_signal(path: Path, max_bytes: int = 2_000_000) -> bool:
    terms = [
        " ERROR ",
        " FATAL ",
        "Exception",
        "Error:",
        "Connection refused",
        "Authentication required",
        "OutOfMemoryError",
        "timed out",
        "Failed",
        "FAILED",
    ]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            remaining = max_bytes
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                if any(term in chunk for term in terms):
                    return True
                remaining -= len(chunk)
    except OSError:
        return False
    return False


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def short_path(path: Path) -> str:
    parts = path.parts
    if len(parts) <= 5:
        return str(path)
    return str(Path(*parts[-5:]))


def format_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


if __name__ == "__main__":
    main()
