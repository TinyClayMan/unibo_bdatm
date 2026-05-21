import os
import re
import csv
import json
import time
import math
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from cafa import load_split_pt


UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def normalize_accession(x):
    return str(x).strip()


def canonical_uniprot_accession(acc):
    acc = normalize_accession(acc)
    return acc.split("-")[0]


def extract_resolution_float(text):
    if text is None:
        return None
    s = str(text).strip()
    if not s or s in {"-", "None", "null"}:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_coverage_len(chains_text):
    """
    Example values:
      "A=1-276"
      "A/C/E/G=1-26, B/D/F/H=27-149"
      "A=12-214, B=10-215"
    We just sum all interval lengths we can parse.
    """
    if not chains_text:
        return 0

    total = 0
    for start, end in re.findall(r"(\d+)\s*-\s*(\d+)", str(chains_text)):
        a, b = int(start), int(end)
        if b >= a:
            total += (b - a + 1)
    return total


def parse_pdb_crossrefs_from_uniprot_json(payload):
    refs = payload.get("uniProtKBCrossReferences", [])
    out = []

    for ref in refs:
        if ref.get("database") != "PDB":
            continue

        pdb_id = ref.get("id")
        props = {}
        for p in ref.get("properties", []):
            key = p.get("key")
            val = p.get("value")
            if key is not None:
                props[key] = val

        method = props.get("Method")
        resolution_raw = props.get("Resolution")
        chains = props.get("Chains")

        out.append({
            "pdb_id": pdb_id,
            "method": method,
            "resolution": extract_resolution_float(resolution_raw),
            "resolution_raw": resolution_raw,
            "chains": chains,
            "coverage_len": extract_coverage_len(chains),
        })

    return out


def method_rank(method):
    if not method:
        return 999
    m = method.lower()
    if "x-ray" in m or "xray" in m:
        return 0
    if "electron microscopy" in m or "cryo-em" in m or "electron micro" in m:
        return 1
    if "nmr" in m:
        return 2
    return 50


def is_acceptable_structure(hit, max_xray_resolution, max_em_resolution, allow_nmr):
    method = (hit.get("method") or "").lower()
    res = hit.get("resolution")

    if "x-ray" in method or "xray" in method:
        return res is not None and res <= max_xray_resolution

    if "electron microscopy" in method or "cryo-em" in method or "electron micro" in method:
        return res is not None and res <= max_em_resolution

    if "nmr" in method:
        return allow_nmr

    return False


def choose_best_hit(hits, max_xray_resolution, max_em_resolution, allow_nmr):
    accepted = [
        h for h in hits
        if is_acceptable_structure(h, max_xray_resolution, max_em_resolution, allow_nmr)
    ]
    if not accepted:
        return None

    # Prefer method class, then lower resolution, then larger mapped coverage
    accepted = sorted(
        accepted,
        key=lambda h: (
            method_rank(h.get("method")),
            float("inf") if h.get("resolution") is None else h.get("resolution"),
            -int(h.get("coverage_len", 0)),
            h.get("pdb_id") or "",
        )
    )
    return accepted[0]


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "experimental-pdb-downloader/1.0"})
    return s


def get_uniprot_json(session, accession, timeout=30, retries=3, sleep=1.0):
    tried = []
    for acc in [accession, canonical_uniprot_accession(accession)]:
        if acc in tried:
            continue
        tried.append(acc)

        url = UNIPROT_JSON_URL.format(accession=acc)
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=timeout)
                if r.status_code == 200:
                    return r.json(), acc
                if r.status_code == 404:
                    break
                r.raise_for_status()
            except Exception:
                if attempt == retries - 1:
                    break
                time.sleep(sleep * (attempt + 1))
    return None, None


def download_structure_file(session, pdb_id, out_dir, prefer_format="pdb", timeout=60, retries=3):
    pdb_id = str(pdb_id).upper()
    os.makedirs(out_dir, exist_ok=True)

    candidates = []
    if prefer_format == "pdb":
        candidates = [("pdb", RCSB_PDB_URL.format(pdb_id=pdb_id)),
                      ("cif", RCSB_CIF_URL.format(pdb_id=pdb_id))]
    else:
        candidates = [("cif", RCSB_CIF_URL.format(pdb_id=pdb_id)),
                      ("pdb", RCSB_PDB_URL.format(pdb_id=pdb_id))]

    for fmt, url in candidates:
        out_path = os.path.join(out_dir, f"{pdb_id}.{fmt}")
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=timeout)
                if r.status_code == 200 and r.content:
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    return out_path, fmt, url
                if r.status_code == 404:
                    break
                r.raise_for_status()
            except Exception:
                if attempt == retries - 1:
                    break
                time.sleep(1.0 * (attempt + 1))
    return None, None, None


def process_accession(
    accession,
    split_name,
    split_out_dir,
    max_xray_resolution,
    max_em_resolution,
    allow_nmr,
    prefer_format,
):
    session = make_session()

    payload, resolved_acc = get_uniprot_json(session, accession)
    result = {
        "split": split_name,
        "accession": accession,
        "resolved_accession": resolved_acc,
        "status": None,
        "num_pdb_crossrefs": 0,
        "selected_pdb_id": None,
        "selected_method": None,
        "selected_resolution": None,
        "selected_chains": None,
        "selected_coverage_len": None,
        "download_format": None,
        "download_path": None,
        "download_url": None,
        "all_hits_json": None,
        "note": None,
    }

    if payload is None:
        result["status"] = "uniprot_not_found"
        result["note"] = "UniProt accession not found via REST API"
        return result

    hits = parse_pdb_crossrefs_from_uniprot_json(payload)
    result["num_pdb_crossrefs"] = len(hits)
    result["all_hits_json"] = json.dumps(hits, ensure_ascii=False)

    if not hits:
        result["status"] = "no_pdb_crossrefs"
        result["note"] = "No PDB cross-references on UniProt entry"
        return result

    best = choose_best_hit(
        hits=hits,
        max_xray_resolution=max_xray_resolution,
        max_em_resolution=max_em_resolution,
        allow_nmr=allow_nmr,
    )

    if best is None:
        result["status"] = "no_good_experimental_hit"
        result["note"] = (
            f"No acceptable hit after quality filter "
            f"(xray<={max_xray_resolution}, em<={max_em_resolution}, allow_nmr={allow_nmr})"
        )
        return result

    result["selected_pdb_id"] = best.get("pdb_id")
    result["selected_method"] = best.get("method")
    result["selected_resolution"] = best.get("resolution")
    result["selected_chains"] = best.get("chains")
    result["selected_coverage_len"] = best.get("coverage_len")

    accession_dir = os.path.join(split_out_dir, "structures")
    out_path, fmt, url = download_structure_file(
        session=session,
        pdb_id=best["pdb_id"],
        out_dir=accession_dir,
        prefer_format=prefer_format,
    )

    if out_path is None:
        result["status"] = "download_failed"
        result["note"] = "Could not download selected structure from RCSB"
        return result

    result["status"] = "downloaded"
    result["download_format"] = fmt
    result["download_path"] = out_path
    result["download_url"] = url
    return result


def save_manifest(rows, out_csv, out_json):
    fieldnames = [
        "split",
        "accession",
        "resolved_accession",
        "status",
        "num_pdb_crossrefs",
        "selected_pdb_id",
        "selected_method",
        "selected_resolution",
        "selected_chains",
        "selected_coverage_len",
        "download_format",
        "download_path",
        "download_url",
        "note",
        "all_hits_json",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def dedupe_keep_order(ids):
    seen = set()
    out = []
    for x in ids:
        x = normalize_accession(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split_embed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--valid_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)

    parser.add_argument("--max_xray_resolution", type=float, default=2.5)
    parser.add_argument("--max_em_resolution", type=float, default=3.5)
    parser.add_argument("--allow_nmr", action="store_true")
    parser.add_argument("--prefer_format", type=str, default="pdb", choices=["pdb", "cif"])

    parser.add_argument("--max_workers", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    valid_pt = args.valid_pt or os.path.join(args.split_embed_dir, "valid_embeddings.pt")
    test_pt = args.test_pt or os.path.join(args.split_embed_dir, "test_embeddings.pt")

    _, _, valid_ids = load_split_pt(valid_pt)
    _, _, test_ids = load_split_pt(test_pt)

    valid_ids = dedupe_keep_order(valid_ids)
    test_ids = dedupe_keep_order(test_ids)

    print(f"valid ids: {len(valid_ids)}")
    print(f"test  ids: {len(test_ids)}")

    all_results = []

    for split_name, ids in [("valid", valid_ids), ("test", test_ids)]:
        split_out_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        rows = []
        futures = []

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            for accession in ids:
                futures.append(
                    ex.submit(
                        process_accession,
                        accession,
                        split_name,
                        split_out_dir,
                        args.max_xray_resolution,
                        args.max_em_resolution,
                        args.allow_nmr,
                        args.prefer_format,
                    )
                )

            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                rows.append(row)
                if i % 25 == 0 or i == len(futures):
                    print(f"[{split_name}] done {i}/{len(futures)}")

        rows = sorted(rows, key=lambda r: r["accession"])

        save_manifest(
            rows,
            out_csv=os.path.join(split_out_dir, f"{split_name}_experimental_structures.csv"),
            out_json=os.path.join(split_out_dir, f"{split_name}_experimental_structures.json"),
        )

        n_ok = sum(r["status"] == "downloaded" for r in rows)
        print(f"{split_name}: downloaded {n_ok}/{len(rows)}")
        all_results.extend(rows)

    summary = {
        "valid_total": sum(r["split"] == "valid" for r in all_results),
        "valid_downloaded": sum(r["split"] == "valid" and r["status"] == "downloaded" for r in all_results),
        "test_total": sum(r["split"] == "test" for r in all_results),
        "test_downloaded": sum(r["split"] == "test" and r["status"] == "downloaded" for r in all_results),
        "params": {
            "max_xray_resolution": args.max_xray_resolution,
            "max_em_resolution": args.max_em_resolution,
            "allow_nmr": args.allow_nmr,
            "prefer_format": args.prefer_format,
            "max_workers": args.max_workers,
        }
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()