"""Byte masks for the 320-byte packet records of the processed release.

Each packet record holds an 80-byte header slot (the IP header followed by the TCP or UDP
header, zero padded or truncated) and a 240-byte payload slot (the first 240 payload bytes,
zero padded). In the NetMamba+ preprocessing the header slot is the IP packet with its
scapy Raw payload removed. Link-layer bytes after the IP packet, such as Ethernet padding
and a captured frame check sequence, therefore stay in the header slot, and payloads that
scapy decodes, such as DNS, stay in the header slot while the payload slot is empty.

mask_hdr      zeroes the IPv4 identification, TTL and header checksum, the IPv6 flow label
              and hop limit, the TCP sequence and acknowledgment numbers, checksum and
              options, and the UDP checksum; length fields and flags are kept
mask_sni      zeroes the whole server_name extension of a TLS ClientHello, including its
              type and length fields, as far as it lies inside the payload slot
mask_trailer  zeroes the link-layer bytes after the IP packet in both slots
mask_both     mask_hdr and mask_sni
mask_all      mask_trailer, mask_hdr and mask_sni (the combined mask)

Run this file to execute the unit tests and print before/after examples in hex:

    python src/masks.py
"""
import sys
from collections import Counter

import numpy as np

REC, HDR = 320, 80


def _u16(b, i):
    return (int(b[i]) << 8) | int(b[i + 1])


def ip_layout(rec):
    """Return (version, transport_offset, proto, tcp_header_len, payload_len) or None."""
    v = int(rec[0]) >> 4
    if v == 4:
        ihl = (int(rec[0]) & 0xF) * 4
        if ihl < 20:
            return None
        proto, frag = int(rec[9]), _u16(rec, 6) & 0x1FFF
        if frag:
            return 4, None, proto, None, None
        total = _u16(rec, 2)
        t = ihl
    elif v == 6:
        proto, total, t = int(rec[6]), _u16(rec, 4) + 40, 40
    else:
        return None
    if proto == 6 and t + 13 <= HDR:
        th = (int(rec[t + 12]) >> 4) * 4
        return v, t, proto, th, max(total - t - th, 0)
    if proto == 17 and t + 8 <= HDR:
        return v, t, proto, 8, max(total - t - 8, 0)
    return v, t, proto, None, None


def mask_hdr_record(rec):
    lay = ip_layout(rec)
    if lay is None:
        return "not_ip"
    v, t, proto, th, _ = lay
    if v == 4:
        rec[4:6] = 0
        rec[8] = 0
        rec[10:12] = 0
    else:
        rec[1] &= 0xF0
        rec[2:4] = 0
        rec[7] = 0
    if t is None:
        return f"ipv{v}_fragment"
    if proto == 6 and th is not None:
        rec[t + 4:min(t + 12, HDR)] = 0
        rec[t + 16:min(t + 18, HDR)] = 0
        if th > 20:
            rec[t + 20:min(t + th, HDR)] = 0
        return f"ipv{v}_tcp" + ("" if t + th <= HDR else "_options_truncated")
    if proto == 17 and th is not None:
        rec[t + 6:t + 8] = 0
        return f"ipv{v}_udp"
    return f"ipv{v}_other_proto"


def client_hello_sni(payload):
    """Locate the server_name extension in a ClientHello at the start of `payload`.
    Returns (status, start, end, host) with offsets relative to the payload; end is
    clipped to the payload length."""
    n = len(payload)
    if n < 6 or payload[0] != 0x16 or payload[1] != 0x03 or payload[5] != 0x01:
        return "no_client_hello", None, None, None
    p = 5 + 4 + 2 + 32
    if p + 1 > n:
        return "truncated", None, None, None
    p += 1 + payload[p]
    if p + 2 > n:
        return "truncated", None, None, None
    p += 2 + _u16(payload, p)
    if p + 1 > n:
        return "truncated", None, None, None
    p += 1 + payload[p]
    if p + 2 > n:
        return "truncated", None, None, None
    end_ext = p + 2 + _u16(payload, p)
    p += 2
    while p + 4 <= min(n, end_ext):
        etype, elen = _u16(payload, p), _u16(payload, p + 2)
        if etype == 0x0000:
            end = p + 4 + elen
            host = None
            if p + 9 <= n:
                hlen = _u16(payload, p + 7)
                if p + 9 + hlen <= n:
                    host = bytes(payload[p + 9:p + 9 + hlen])
            return ("complete" if end <= n else "partial"), p, min(end, n), host
        p += 4 + elen
    return ("absent" if p >= end_ext else "truncated"), None, None, None


def payload_len(rec):
    lay = ip_layout(rec)
    if lay is None or lay[4] is None:
        return REC - HDR
    return min(lay[4], REC - HDR)


def ip_total(rec):
    v = int(rec[0]) >> 4
    return _u16(rec, 2) if v == 4 else _u16(rec, 4) + 40


def mask_trailer_record(rec):
    """Zero link-layer bytes after the IP packet. When the payload slot holds the
    payload, the header slot ends after the transport header; otherwise the
    decoded payload stays in the header slot and the IP packet ends at its total
    length. In the payload slot, bytes after the payload length are trailer."""
    lay = ip_layout(rec)
    if lay is None or lay[1] is None or lay[3] is None:
        return "not_parsed"
    _, t, _, th, plen = lay
    payload_in_slot = bool(rec[HDR:].any())
    end = t + th if payload_in_slot else ip_total(rec)
    done = []
    if end < HDR and rec[end:HDR].any():
        rec[end:HDR] = 0
        done.append("header_slot")
    if payload_in_slot and plen < REC - HDR and rec[HDR + plen:].any():
        rec[HDR + plen:] = 0
        done.append("payload_slot")
    return "+".join(done) or "none"


def mask_sni_record(rec):
    plen = payload_len(rec)
    status, start, end, _ = client_hello_sni(bytes(rec[HDR:HDR + plen]))
    if start is not None:
        rec[HDR + start:HDR + end] = 0
    return status


def apply(arr, mask):
    """Mask every packet record of a (flows, 1600) uint8 array in place; return counts."""
    stats = Counter()
    for i in range(arr.shape[0]):
        for k in range(arr.shape[1] // REC):
            rec = arr[i, k * REC:(k + 1) * REC]
            if not rec.any():
                stats["empty"] += 1
                continue
            if mask in ("mask_trailer", "mask_all"):
                stats["trailer:" + mask_trailer_record(rec)] += 1
            if mask in ("mask_hdr", "mask_both", "mask_all"):
                stats["hdr:" + mask_hdr_record(rec)] += 1
            if mask in ("mask_sni", "mask_both", "mask_all"):
                stats["sni:" + mask_sni_record(rec)] += 1
    return dict(stats)


# ---------------------------------------------------------------- unit tests

def _client_hello(host=b"example.com"):
    sni = b"\x00" + len(host).to_bytes(2, "big") + host
    sni_ext = b"\x00\x00" + (len(sni) + 2).to_bytes(2, "big") + (len(sni)).to_bytes(2, "big") + sni
    other = b"\x00\x0b\x00\x02\x01\x00"                     # ec_point_formats
    exts = other + sni_ext
    body = b"\x03\x03" + bytes(range(32)) + b"\x20" + bytes(range(100, 132)) + b"\x00\x04\x13\x01\x13\x02" + b"\x01\x00"
    body += len(exts).to_bytes(2, "big") + exts
    hs = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(hs).to_bytes(2, "big") + hs, len(b"\x16\x03\x01") + 2 + 4 + len(body) - len(exts) + len(other), len(sni_ext)


def _record(ip, transport, payload):
    rec = np.zeros(REC, dtype=np.uint8)
    hdr = ip + transport
    rec[:min(len(hdr), HDR)] = np.frombuffer(hdr[:HDR], dtype=np.uint8)
    rec[HDR:HDR + min(len(payload), REC - HDR)] = np.frombuffer(payload[:REC - HDR], dtype=np.uint8)
    return rec


def _ipv4(proto, total):
    return bytes([0x45, 0x11]) + total.to_bytes(2, "big") + b"\xab\xcd\x40\x00" + bytes([0x37, proto]) + b"\xbe\xef" + bytes(8)


def _tcp(doff_words):
    opts = bytes([0x01] * (doff_words * 4 - 20))
    return b"\x00\x00\x00\x00" + b"\x11\x22\x33\x44" + b"\x55\x66\x77\x88" + bytes([doff_words << 4, 0x18]) + b"\x01\xf5" + b"\xca\xfe" + b"\x00\x00" + opts


def _hex(rec, a, b):
    return " ".join(f"{x:02x}" for x in rec[a:b])


def test():
    out = []
    ch, sni_off, sni_len = _client_hello()
    tcp = _tcp(8)
    rec = _record(_ipv4(6, 20 + 32 + len(ch)), tcp, ch)
    before = rec.copy()
    status = mask_hdr_record(rec)
    assert status == "ipv4_tcp", status
    want = {4, 5, 8, 10, 11} | set(range(24, 32)) | {36, 37} | set(range(40, 52))
    changed = {int(i) for i in np.flatnonzero(before != rec)}
    assert changed == want, sorted(changed ^ want)
    assert (rec[sorted(want)] == 0).all()
    out.append(("IPv4/TCP header slot, mask_hdr", _hex(before, 0, 52), _hex(rec, 0, 52)))

    rec2 = before.copy()
    assert mask_sni_record(rec2) == "complete"
    changed = {int(i) for i in np.flatnonzero(before != rec2)}
    want = set(range(HDR + sni_off, HDR + sni_off + sni_len))
    assert changed <= want and (rec2[sorted(want)] == 0).all(), sorted(changed ^ want)
    assert b"example.com" not in bytes(rec2) and b"example.com" in bytes(before)
    out.append(("ClientHello payload slot, mask_sni", _hex(before, HDR + sni_off - 4, HDR + sni_off + sni_len + 2),
                _hex(rec2, HDR + sni_off - 4, HDR + sni_off + sni_len + 2)))

    udp = b"\x00\x00\x00\x00\x00\x24\x9a\xbc"
    rec3 = _record(_ipv4(17, 20 + 8 + 28), udp, bytes(range(1, 29)))
    b3 = rec3.copy()
    assert mask_hdr_record(rec3) == "ipv4_udp"
    changed = {int(i) for i in np.flatnonzero(b3 != rec3)}
    assert changed == {4, 5, 8, 10, 11, 26, 27}, sorted(changed)
    assert mask_sni_record(rec3) == "no_client_hello"
    out.append(("IPv4/UDP header slot, mask_hdr", _hex(b3, 0, 28), _hex(rec3, 0, 28)))

    ip6 = bytes([0x60, 0x1a, 0xbc, 0xde]) + (20 + 10).to_bytes(2, "big") + bytes([6, 0x40]) + bytes(range(1, 33))
    rec4 = _record(ip6, _tcp(5), bytes(range(10)))
    b4 = rec4.copy()
    assert mask_hdr_record(rec4) == "ipv6_tcp"
    changed = {int(i) for i in np.flatnonzero(b4 != rec4)}
    assert changed == {1, 2, 3, 7} | set(range(44, 52)) | {56, 57}, sorted(changed)
    assert rec4[0] == 0x60 and rec4[1] == 0x10
    out.append(("IPv6/TCP header slot, mask_hdr", _hex(b4, 0, 60), _hex(rec4, 0, 60)))

    arr = np.stack([np.concatenate([before, np.zeros(4 * REC, np.uint8)])])
    stats = apply(arr, "mask_both")
    assert stats == {"hdr:ipv4_tcp": 1, "sni:complete": 1, "empty": 4}, stats

    ack = _record(_ipv4(6, 40), _tcp(5) + b"P/1.1\r", b"")        # 40-byte ACK with 6 padding bytes
    b5 = ack.copy()
    assert mask_trailer_record(ack) == "header_slot"
    assert {int(i) for i in np.flatnonzero(b5 != ack)} == set(range(40, 46))
    out.append(("40-byte ACK with Ethernet padding, mask_trailer", _hex(b5, 0, 48), _hex(ack, 0, 48)))

    dns = bytes(range(1, 31))                                       # decoded UDP payload stays in the header slot
    udp_rec = _record(_ipv4(17, 20 + 8 + 30), udp + dns + b"\xde\xad\xbe\xef", b"")
    b6 = udp_rec.copy()
    assert mask_trailer_record(udp_rec) == "header_slot"
    assert {int(i) for i in np.flatnonzero(b6 != udp_rec)} == set(range(58, 62))
    out.append(("UDP with decoded payload and 4 trailer bytes, mask_trailer", _hex(b6, 26, 64), _hex(udp_rec, 26, 64)))

    tcp_rec = _record(_ipv4(6, 20 + 20 + 10), _tcp(5), bytes(range(1, 11)) + b"\x7e\x7b\xc7\x05")
    b7 = tcp_rec.copy()
    assert mask_trailer_record(tcp_rec) == "payload_slot"
    assert {int(i) for i in np.flatnonzero(b7 != tcp_rec)} == set(range(HDR + 10, HDR + 14))

    arr = np.stack([np.concatenate([before, b5, np.zeros(3 * REC, np.uint8)])])
    stats = apply(arr, "mask_all")
    assert stats["trailer:header_slot"] == 1 and stats["sni:complete"] == 1 and stats["hdr:ipv4_tcp"] == 2, stats
    for name, b, a in out:
        print(f"{name}\n  before: {b}\n  after:  {a}")
    print("all mask tests passed")


if __name__ == "__main__":
    sys.exit(test())
