#!/usr/bin/env python3
"""Render the LLMario app icon (1024x1024 PNG) with no dependencies: an indigo gradient
squircle with a white chat bubble. Shapes use signed distances for anti-aliased edges."""
import math, struct, sys, zlib

N = 1024

def rrect(px, py, x0, y0, x1, y1, r):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    dx, dy = abs(px - cx) - hx, abs(py - cy) - hy
    return math.hypot(max(dx, 0), max(dy, 0)) + min(max(dx, dy), 0) - r

def tri(px, py, pts):
    d, inside = 1e9, False
    for i in range(3):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        t = max(0, min(1, ((px - ax) * ex + (py - ay) * ey) / (ex * ex + ey * ey)))
        d = min(d, math.hypot(px - ax - t * ex, py - ay - t * ey))
        if (ay > py) != (by > py) and px < ax + (py - ay) * ex / ey:
            inside = not inside
    return -d if inside else d

def cov(d):  # coverage from signed distance (1px AA)
    return max(0.0, min(1.0, 0.5 - d))

def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

TOP, BOT = (99, 102, 241), (124, 58, 237)   # indigo-500 → violet-600
DOT = (79, 70, 229)
rows = []
for y in range(N):
    row = bytearray([0])
    for x in range(N):
        px, py = x + 0.5, y + 0.5
        a_bg = cov(rrect(px, py, 64, 64, 960, 960, 200))
        col = lerp(TOP, BOT, (px + py) / (2 * N))
        bubble = min(rrect(px, py, 232, 262, 792, 682, 130),
                     tri(px, py, [(300, 600), (430, 660), (268, 800)]))
        a_b = cov(bubble)
        col = lerp(col, (255, 255, 255), a_b)
        for cx in (382, 512, 642):
            a_d = cov(math.hypot(px - cx, py - 472) - 46) * a_b
            col = lerp(col, DOT, a_d)
        row += bytes(int(round(c)) for c in col) + bytes([int(round(255 * a_bg))])
    rows.append(bytes(row))

def chunk(t, data):
    return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", N, N, 8, 6, 0, 0, 0)) \
    + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
open(sys.argv[1] if len(sys.argv) > 1 else "icon.png", "wb").write(png)
