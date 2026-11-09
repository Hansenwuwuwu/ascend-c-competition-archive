"""Independent static checks of already-read key lists and sorting networks.

Only parses archived text. It never imports, compiles or runs archived code.
This does not emulate the CANN key encoder or validate device instruction APIs.
"""
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "extracted/SplineConv/rank3_submission_1177413/code/op_kernel"


def main():
    source = (KERNEL / "tiling_key_spline_conv.h").read_text(encoding="utf-8-sig")
    groups = re.findall(r"ASCENDC_TPL_ARGS_SEL\((.*?)\n    \)", source, re.S)
    fields = ("MODE", "DIM", "DEG", "CI", "CO", "ES", "NN", "INPUT_T")
    seen, by_type, by_mode, details = set(), {}, {}, []
    for number, group in enumerate(groups, 1):
        choices = {}
        for _, name, values in re.findall(r"ASCENDC_TPL_(UINT|DATATYPE)_SEL\((\w+),\s*([^)]*)\)", group):
            items = [v.strip() for v in values.split(",") if v.strip() != "ASCENDC_TPL_UI_LIST"]
            choices[name] = items
        assert set(choices) == set(fields), (number, choices)
        count = math.prod(len(choices[k]) for k in fields)
        for item in itertools.product(*(choices[k] for k in fields)):
            assert item not in seen, ("Duplicate selection", item)
            seen.add(item)
            by_type[item[-1]] = by_type.get(item[-1], 0) + 1
            by_mode[item[0]] = by_mode.get(item[0], 0) + 1
        details.append({"group": number, "count": count, "fields": choices})
    assert len(seen) == 462 and by_type == {"C_DT_FLOAT": 252, "C_DT_BF16": 210}, by_type
    lines = (KERNEL / "spline_conv.cpp").read_text(encoding="utf-8-sig").splitlines()
    # Exact contiguous source intervals already read in G3, including branches.
    intervals = {2: (3094, 3097), 4: (3099, 3118), 8: (3120, 3195)}
    networks = {}
    for width, (start, end) in intervals.items():
        section = "\n".join(lines[start - 1:end])
        pairs = [(int(a), int(b)) for a, b in re.findall(r"lo = z(\d) < z(\d) \?", section)]
        permutations = 0
        for original in itertools.permutations(range(width)):
            values = list(original)
            for a, b in pairs:
                values[a], values[b] = min(values[a], values[b]), max(values[a], values[b])
            assert values == list(range(width)), (width, original, values)
            permutations += 1
        networks[width] = {"source_lines": [start, end], "comparators": pairs,
                           "distinct_permutations_checked": permutations,
                           "meaning": "Sorting values only; duplicates follow the same min/max order property. No NPU executed."}
    result = {"operator": "SplineConv", "rank": 3, "key_groups": len(groups), "expanded_selections": len(seen),
              "selections_by_dtype": by_type, "selections_by_mode": by_mode, "group_details": details,
              "sorting_networks": networks,
              "sha256": {name: hashlib.sha256((KERNEL / name).read_bytes()).hexdigest()
                         for name in ("tiling_key_spline_conv.h", "spline_conv.cpp")},
              "limits": "Not a Host-route enumeration, compiler validation, mathematical FP32 error proof or NPU benchmark."}
    (ROOT / "deep_audit/SPLINE_STATIC_TABLE_CHECKS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS", len(groups), "SEL groups;", len(seen), "unique combinations;", by_type)
    print("PASS networks", {n: v["distinct_permutations_checked"] for n, v in networks.items()})


if __name__ == "__main__":
    main()
