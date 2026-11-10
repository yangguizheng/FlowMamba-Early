"""Byte content of the first five packets of every test flow; no model is run.

For each dataset: checks of the 80/240-byte layout, the protocol, TCP flags and payload length
of the first packet, the positions of TLS ClientHello and ServerHello messages, and, for the
TLS datasets, the packet in which the server_name extension first appears, whether the host
name reveals the class label, and whether any copy of the host name remains after mask_sni.
Writes OUTPUT_ROOT/results/sni_audit.json and a readable summary, sni_audit.md.

    python src/sni_audit.py
"""
import json
import os
from collections import Counter

import numpy as np

import experiment as L
import masks as M
from paths import RESULTS_OUT


def tls_kind(payload):
    if len(payload) >= 6 and payload[0] == 0x16 and payload[1] == 0x03:
        return {1: "client_hello", 2: "server_hello"}.get(payload[5], "handshake_other")
    if len(payload) >= 3 and 20 <= payload[0] <= 23 and payload[1] == 0x03:
        return "tls_record_other"
    return None


def bin_len(n):
    if n is None:
        return "unknown"
    return "0" if n == 0 else ("1-99" if n < 100 else ("100-239" if n < 240 else ">=240"))


def label_in_host(host, name):
    h = host.decode(errors="ignore").lower().rstrip(".")
    return h == name or h.endswith("." + name)


def audit(dataset):
    sp = L.load_split(dataset, "test")
    meta = json.load(open(f"{L.DATA_ROOT}/{dataset}/metadata.json"))
    names = {v: k.lower() for k, v in meta["name_to_idx"].items()}
    B = sp["bytes"]
    layout, proto1, flags1, plen1, hello, kind1 = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    first_sni, direction, host_match, residual = [], [], Counter(), Counter()
    examples = []
    for i in range(len(B)):
        fs, dirn, host = 0, "unknown", None
        for k in range(5):
            rec = B[i, k * 320:(k + 1) * 320]
            if not rec.any():
                continue
            lay = M.ip_layout(rec)
            if lay is not None and lay[1] is not None and lay[3] is not None:
                hl = lay[1] + lay[3]
                if hl <= 80:
                    layout["header_padding_zero" if not rec[hl:80].any() else "header_padding_nonzero"] += 1
                else:
                    layout["header_longer_than_slot"] += 1
                pl = lay[4]
                if pl is not None and pl < 240:
                    layout["payload_padding_zero" if not rec[80 + pl:].any() else "payload_padding_nonzero"] += 1
                if lay[2] == 6 and pl:
                    layout["tcp_payload_starts_with_tls_record" if tls_kind(bytes(rec[80:86])) else "tcp_payload_other_start"] += 1
            else:
                layout["ip_or_transport_not_parsed"] += 1
            payload = bytes(rec[80:80 + M.payload_len(rec)])
            kind = tls_kind(payload)
            if kind in ("client_hello", "server_hello"):
                hello[f"{kind}@packet{k + 1}"] += 1
                if dirn == "unknown":
                    dirn = "client" if kind == "client_hello" else "server"
            status, _, _, h = M.client_hello_sni(payload)
            if status in ("complete", "partial") and not fs:
                fs, host = k + 1, h
            if k == 0:
                proto1[("ipv6-" if lay and lay[0] == 6 else "") + str(lay[2] if lay else "unparsed")] += 1
                if lay and lay[2] == 6 and lay[1] is not None:
                    flags1[f"0x{int(rec[lay[1] + 13]):02x}"] += 1
                plen1[bin_len(lay[4] if lay else None)] += 1
                kind1[kind or "no_tls_record"] += 1
                if len(examples) < 4 and status == "complete" and dataset in ("cstnet-tls1.3", "cipherspectrum"):
                    after = rec.copy()
                    M.mask_sni_record(after)
                    s0 = 80 + M.client_hello_sni(payload)[1]
                    examples.append({"flow": i, "label": names[int(sp["labels"][i])], "host": h.decode(errors="ignore") if h else None,
                                     "header_hex": " ".join(f"{x:02x}" for x in rec[:60]),
                                     "sni_before_hex": " ".join(f"{x:02x}" for x in rec[s0:s0 + 40]),
                                     "sni_after_hex": " ".join(f"{x:02x}" for x in after[s0:s0 + 40])})
        first_sni.append(fs)
        direction.append(dirn)
        if host:
            host_match["label_in_host" if label_in_host(host, names[int(sp["labels"][i])]) else "label_not_in_host"] += 1
            flow = B[i].copy()[None, :]
            M.apply(flow, "mask_sni")
            residual["host_still_present_after_mask_sni" if host in bytes(flow[0]) else "host_removed"] += 1
    first_sni, direction = np.array(first_sni), np.array(direction)
    npkt = sp["npkt"]
    sni_by_exit = {}
    for n in L.EXITS:
        elig = npkt >= n
        row = {"eligible_flows": int(elig.sum()),
               "share_with_sni_in_first_n": float(((first_sni >= 1) & (first_sni <= n))[elig].mean())}
        for dname in ["client", "server", "unknown"]:
            m = elig & (direction == dname)
            row[f"{dname}_flows"] = int(m.sum())
            row[f"{dname}_share_with_sni"] = float(((first_sni >= 1) & (first_sni <= n))[m].mean()) if m.any() else None
        sni_by_exit[str(n)] = row
    return {"test_flows": int(len(B)), "layout": dict(layout), "first_packet_protocol": dict(proto1),
            "first_packet_tcp_flags": dict(flags1), "first_packet_payload_length": dict(plen1),
            "first_packet_tls": dict(kind1), "hello_positions": dict(hello),
            "direction": dict(Counter(direction.tolist())), "first_sni_packet": dict(Counter(first_sni.tolist())),
            "sni_by_exit": sni_by_exit, "host_vs_label": dict(host_match), "mask_sni_residual": dict(residual),
            "examples": examples}


def main():
    stats = {d: audit(d) for d in L.DATASETS}
    os.makedirs(RESULTS_OUT, exist_ok=True)
    json.dump(stats, open(f"{RESULTS_OUT}/sni_audit.json", "w"), indent=2, default=int)
    lines = ["# Byte content of the first five packets (test flows)", "",
             "Generated by src/sni_audit.py from the test splits; no model is run. Full counts are in sni_audit.json.", ""]
    for d, s in stats.items():
        lines += [f"## {L.NAMES[d]} ({s['test_flows']} test flows)", "",
                  f"- Layout checks over non-empty packet records: {s['layout']}",
                  f"- First packet protocol: {s['first_packet_protocol']}; TCP flags: {s['first_packet_tcp_flags']}",
                  f"- First packet payload length: {s['first_packet_payload_length']}; TLS content: {s['first_packet_tls']}",
                  f"- ClientHello/ServerHello by packet: {s['hello_positions']}",
                  f"- Flow direction from the handshake type: {s['direction']}",
                  f"- Packet with the first server_name extension (0 = none in five packets): {s['first_sni_packet']}",
                  f"- Host name vs class label: {s['host_vs_label']}; after mask_sni: {s['mask_sni_residual']}", "",
                  "| Exit N | Eligible flows | Share with SNI in first N packets | Client flows (share) | Server flows (share) | Unknown flows (share) |",
                  "|---:|---:|---:|---|---|---|"]
        for n, r in s["sni_by_exit"].items():
            fmt = lambda k: f"{r[k + '_flows']} ({r[k + '_share_with_sni']:.3f})" if r[k + "_share_with_sni"] is not None else f"{r[k + '_flows']} (-)"
            lines.append(f"| {n} | {r['eligible_flows']} | {r['share_with_sni_in_first_n']:.3f} | {fmt('client')} | {fmt('server')} | {fmt('unknown')} |")
        for ex in s["examples"]:
            lines += ["", f"Example flow {ex['flow']} (label {ex['label']}, SNI {ex['host']}), packet 1:",
                      f"    header slot: {ex['header_hex']}", f"    SNI before:  {ex['sni_before_hex']}",
                      f"    SNI after:   {ex['sni_after_hex']}"]
        lines.append("")
    open(f"{RESULTS_OUT}/sni_audit.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
