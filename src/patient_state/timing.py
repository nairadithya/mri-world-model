"""Verified elapsed-time joins for the new structured forecast development.

The original nominal-week forecasting protocol remains in temporal.py.
"""
from collections import defaultdict
import re
import xml.etree.ElementTree as ET
import zipfile


def mu_clinical_days(path, allowed):
    """Read only identifiers and six MRI-day fields for allowed development IDs."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    columns = ("BQ", "BR", "BS", "BT", "BU", "BV")
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(x.itertext()) for x in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        tree = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        def value(cell):
            v = cell.find("s:v", ns)
            text = v.text if v is not None else "".join(cell.itertext())
            return strings[int(text)] if cell.get("t") == "s" else text
        days, issues, headers = {}, [], {}
        for row in tree.findall("s:sheetData/s:row", ns):
            cells = {re.sub(r"\d+", "", c.attrib["r"]): c for c in row}
            if row.attrib["r"] == "1":
                headers = {c: value(cells[c]) for c in columns}
                if not all("diagnosis" in h.lower() and "mri" in h.lower() for h in headers.values()):
                    raise ValueError("Clinical timing headers changed")
                if not all(f'timepoint_{i}' in headers[c].lower() for i,c in enumerate(columns,1)):
                    raise ValueError('Clinical columns no longer match exact timepoint identifiers')
                continue
            patient = value(cells["A"]) if "A" in cells else ""
            if patient not in allowed:
                continue
            if patient in days:
                raise ValueError("Duplicate clinical patient")
            times = {}
            for i, c in enumerate(columns, 1):
                text = value(cells[c]).strip() if c in cells else ""
                if text in ("", "N/A", "NA"):
                    continue
                try:
                    day = float(text)
                    if not day.is_integer():
                        raise ValueError()
                    times[f"Timepoint_{i}"] = int(day)
                except ValueError:
                    issues.append({"patient": patient, "visit": f"Timepoint_{i}", "reason": "invalid day"})
            ordered = list(times.values())
            if any(b <= a for a, b in zip(ordered, ordered[1:])):
                issues.append({"patient": patient, "reason": "nonincreasing clinical dates; whole trajectory excluded"})
                times = {}
            days[patient] = times
    return days, issues, headers


def elapsed_weeks(row):
    if row.get("dataset", "LUMIERE") == "LUMIERE":
        return float(row["visit_order"][0])
    if row.get("time_unit") != "diagnosis_relative_days":
        raise ValueError("MU requires verified clinical elapsed time")
    return row["diagnosis_relative_day"] / 7.


def timed_pairs(rows, horizons=(12, 24), tolerance=4):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient"]].append(row)
    pairs = []
    for patient, visits in sorted(grouped.items()):
        visits.sort(key=lambda r: r["visit_order"])
        # Do not bridge a released observation with absent/invalid clinical timing.
        history = []
        for i, row in enumerate(visits):
            try:
                now = elapsed_weeks(row)
            except ValueError:
                history = []
                continue
            if history and now < elapsed_weeks(history[-1]):
                raise ValueError("Nonmonotone recorded chronology")
            history.append(row)
            if len(history) < 2:
                continue
            future = []
            for target in visits[i+1:]:
                try:
                    gap = elapsed_weeks(target)-now
                except ValueError:
                    break
                if gap > 0:
                    future.append((gap, target))
            for horizon in horizons:
                if not future:
                    continue
                gap, target = min(future, key=lambda v: (abs(v[0]-horizon), v[1]["visit_order"]))
                if abs(gap-horizon) <= tolerance:
                    pairs.append({"patient": patient, "dataset": row.get("dataset", "LUMIERE"),
                                  "history": list(history), "target": target, "horizon": horizon,
                                  "target_offset": gap-horizon})
    return pairs
