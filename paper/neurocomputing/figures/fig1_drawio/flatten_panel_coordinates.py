#!/usr/bin/env python3
"""Flatten panel-local draw.io coordinates for reliable static validation.

Run once after authoring a panel-parented XML file. The panel rectangles remain
editable semantic containers, while all content cells become page-level cells.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def shift_number(raw: str | None, delta: float) -> str | None:
    if raw is None:
        return None
    value = float(raw) + delta
    return str(int(value)) if value.is_integer() else str(value)


def main() -> int:
    path = Path(sys.argv[1])
    tree = ET.parse(path)
    root = tree.getroot()

    offsets: dict[str, tuple[float, float]] = {}
    for cell in root.iter("mxCell"):
        cell_id = cell.get("id", "")
        if not cell_id.startswith("panel_"):
            continue
        geometry = cell.find("mxGeometry")
        if geometry is None:
            continue
        offsets[cell_id] = (
            float(geometry.get("x", "0")),
            float(geometry.get("y", "0")),
        )

    for cell in root.iter("mxCell"):
        panel = cell.get("parent")
        if panel not in offsets:
            continue
        dx, dy = offsets[panel]
        geometry = cell.find("mxGeometry")
        if geometry is not None and cell.get("vertex") == "1":
            geometry.set("x", shift_number(geometry.get("x"), dx) or "0")
            geometry.set("y", shift_number(geometry.get("y"), dy) or "0")
        if geometry is not None and cell.get("edge") == "1":
            for point in geometry.iter("mxPoint"):
                point.set("x", shift_number(point.get("x"), dx) or "0")
                point.set("y", shift_number(point.get("y"), dy) or "0")
        cell.set("parent", "1")

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
