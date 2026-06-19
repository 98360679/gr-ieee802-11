#!/usr/bin/env python3
"""
exp3_ber_compare.py — clean-vs-attacked BER/FER comparison across RX captures.

Decodes each raw-IQ capture (complex64, from the wifi_rx file_sink) with
exp3_ber_eval.py in an ISOLATED subprocess and tabulates BER / FER / error-free
rate, so you can read off how much the adversary perturbation degraded the link.

NB: do NOT use parse_mac's "instantaneous fer" for this rig — frame.bin has a
constant sequence number, which makes that metric read ~1.0 regardless. This tool
compares the actual decoded payload bits (the known 'x' fill), which is valid.

Workflow:
  1. adversary OFF:  exp3_adversary_tx.py --clean ...    + capture -> clean_capture_1.bin
  2. adversary ON :  exp3_adversary_tx.py --epsilon E .. + capture -> eps_E_capture_1.bin
  3. python3 exp3_ber_compare.py clean=/…/clean_capture_1.bin eps0.05=/…/eps_0.05_capture_1.bin

  --json out.json   also dump the comparison
"""
import os, sys, json, argparse, tempfile, subprocess

BER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp3_ber_eval.py")


def decode(path, timeout=2400):
    """Run exp3_ber_eval.py on a capture in a subprocess; return its metrics dict."""
    outj = tempfile.mktemp(suffix="_ber.json")
    try:
        subprocess.run([sys.executable, BER, "--file", path, "--json", outj],
                       capture_output=True, timeout=timeout)
        if os.path.exists(outj):
            return json.load(open(outj))
        return {"decode_failed": True}
    except subprocess.TimeoutExpired:
        return {"decode_failed": True, "timeout": True}
    finally:
        if os.path.exists(outj):
            os.remove(outj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+",
                    help="label=path pairs, e.g. clean=clean_capture_1.bin eps0.05=eps_0.05_capture_1.bin")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print(f"{'label':>12} {'frames':>7} {'err-free':>9} {'BER':>10} {'FER':>6}  {'vs clean':>9}")
    print("-" * 62)
    rows, clean_fer = [], None
    for c in a.captures:
        label, path = c.split("=", 1) if "=" in c else (os.path.basename(c), c)
        if not os.path.exists(path):
            print(f"{label:>12}   MISSING: {path}"); continue
        r = decode(path)
        if r.get("decode_failed"):
            print(f"{label:>12}   DECODE-FAIL")
            rows.append({"label": label, "file": path, "decode_failed": True}); continue
        n = r.get("frames_ours_aligned", 0)
        ef = r.get("frames_error_free", 0)
        ber = r.get("ber", float("nan"))
        fer = r.get("fer", float("nan"))
        if clean_fer is None:
            clean_fer = fer
        delta = "" if (clean_fer is None or fer != fer) else f"{(fer - clean_fer):+.3f}"
        print(f"{label:>12} {n:7d} {ef/max(n,1)*100:8.1f}% {ber:10.2e} {fer:6.3f}  {delta:>9}")
        rows.append({"label": label, "file": path, "n_frames": n, "error_free": ef,
                     "ber": ber, "fer": fer})
    if a.json:
        json.dump({"rows": rows}, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
