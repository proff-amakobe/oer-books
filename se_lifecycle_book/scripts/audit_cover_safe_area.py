#!/usr/bin/env python3
"""Audit declared cover foreground bounds against official Ingram safe boxes."""

from pathlib import Path

PT_PER_IN = 72.0

VOLUMES = {
    "Volume I": {
        "regions": {
            "Back": (222.406, 72.098, 816.406, 846.098),
            "Spine": (829.887, 72.195, 886.480, 846.000),
            "Front": (900.000, 72.098, 1494.000, 846.098),
        },
        "objects": {
            "Back": {
                "series": (246, 800, 650, 814), "headline": (246, 720, 762, 766),
                "description": (246, 430, 762, 704), "author photo": (246, 246, 354, 354),
                "author bio": (374, 246, 724, 352), "license": (246, 206, 520, 218),
                "publisher": (246, 184, 475, 196), "barcode": (676.547, 85.695, 766.907, 157.695),
            },
            "Spine": {"title": (853, 320, 863, 644), "author": (854, 120, 863, 222)},
            "Front": {
                "series": (956, 810, 1285, 824), "title": (956, 674, 1456, 756),
                "edition": (956, 510, 1045, 523), "system motif": (966, 255, 1457, 401),
                "author": (956, 135, 1140, 159), "publisher": (956, 105, 1160, 117),
            },
        },
    },
    "Volume II": {
        "regions": {
            "Back": (227.594, 72.098, 821.594, 846.098),
            "Spine": (835.070, 72.195, 886.480, 846.000),
            "Front": (900.000, 72.098, 1494.000, 846.098),
        },
        "objects": {
            "Back": {
                "series": (251, 800, 655, 814), "headline": (251, 720, 767, 766),
                "description": (251, 430, 767, 704), "author photo": (251, 246, 359, 354),
                "author bio": (379, 246, 729, 352), "license": (251, 206, 525, 218),
                "publisher": (251, 184, 480, 196), "barcode": (681.730, 85.695, 772.094, 157.695),
            },
            "Spine": {"title": (856, 318, 865, 646), "author": (856, 120, 865, 222)},
            "Front": {
                "series": (956, 810, 1290, 824), "title": (956, 680, 1456, 756),
                "edition": (956, 473, 1045, 486), "pipeline motif": (964, 244, 1442, 402),
                "author": (956, 135, 1140, 159), "publisher": (956, 105, 1160, 117),
            },
        },
    },
}


def clearance(safe, obj):
    sx0, sy0, sx1, sy1 = safe
    x0, y0, x1, y1 = obj
    return min(x0 - sx0, y0 - sy0, sx1 - x1, sy1 - y1)


def main():
    lines = ["# Cover Safe-Area Audit", "", "Measurements are calculated from the actual foreground bounding boxes declared in the vector cover sources and the official Ingram template safe rectangles. Positive clearance means the complete object is inside the official pink/red safe boundary.", ""]
    failed = False
    for volume, data in VOLUMES.items():
        lines += [f"## {volume}", "", "| Region | Critical object | Minimum clearance | Result |", "|---|---|---:|---|"]
        region_results = {}
        for region, objects in data["objects"].items():
            region_ok = True
            for name, box in objects.items():
                points = clearance(data["regions"][region], box)
                ok = points > 0
                region_ok &= ok
                failed |= not ok
                lines.append(f"| {region} | {name} | {points / PT_PER_IN:.3f} in | {'PASS' if ok else 'FAIL'} |")
            region_results[region] = region_ok
        lines += ["", "### Region result", ""]
        for region in ("Front", "Back", "Spine"):
            lines.append(f"- {volume} {region}: **{'PASS' if region_results[region] else 'FAIL'}**")
        lines.append("")
    lines += ["## Final result", "", f"**{'PASS' if not failed else 'FAIL'}**", ""]
    target = Path(__file__).resolve().parents[1] / "editorial" / "cover-safe-area-audit.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
