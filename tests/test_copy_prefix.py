"""Authored controls for the short-suffix family, admission and leakage boundaries."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from certificate import check
from run_copy_prefix import generate, disjoint, library

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def staged(rows, pairs):
    return {"tok": "".join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p, t in enumerate(r["ctx"])),
            "sample": "".join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
            "ref": "".join(f"{r['id']}\t{r['expected']}\n" for r in rows),
            "pair": "".join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
            "edit": "".join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p["positions"])}


def test_fresh_generator_and_declared_interventions():
    rows, pairs = generate({1000, 1001}, groups=1)
    stats = disjoint(rows, {1000, 1001})
    assert {s: x['rows'] for s, x in stats.items()} == {"train": 78, "validation": 78}
    by_id = {r['id']: r for r in rows}
    for p in pairs:
        a, b = by_id[p['a']], by_id[p['b']]
        assert [i for i, (t, u) in enumerate(zip(a['ctx'], b['ctx'])) if t != u] == p['positions']
        assert a['ctx'][-1] == b['ctx'][-1]
    leaked = [dict(r) for r in rows]
    leaked[-1]['ctx'] = leaked[0]['ctx']
    with pytest.raises(ValueError, match="overlap"):
        disjoint(leaked, set())


def test_authored_copy_admits_greatest_coverage_and_rejects_corrupt_references(tmp_path):
    rows, pairs = generate(set(), groups=1)
    circuit = tmp_path / 'library.dl'
    circuit.write_text(library())
    facts = staged(rows, pairs)
    verifier = ROOT / 'dl/copy_prefix_select.dl'
    result = check(verifier, circuit, facts, {})
    assert result['certified'], result.get('error')
    assert result['relations']['cp_selected'] == [['1', '1', '0']]
    assert not result['relations']['baseline_overlap']
    # One deliberately wrong reference per firing stratum rejects every candidate.
    # Do this on the longest context so the >=32 candidates are tested too.
    target = next(r for r in rows if r['prefix'] == 2 and r['layout'] == 'noise32' and not r['negative'])
    bad = dict(facts, ref=facts['ref'].replace(f"{target['id']}\t{target['expected']}\n", f"{target['id']}\t999999\n"))
    failed = check(verifier, circuit, bad, {})
    assert failed['certified']  # input integrity still holds; scientific admission fails
    assert failed['relations']['cp_selected'] == []
    assert len(failed['relations']['rejected']) == 6
    missing = dict(facts, ref=facts['ref'].replace(f"{target['id']}\t{target['expected']}\n", ''))
    assert not check(verifier, circuit, missing, {})['certified']


def test_repeated_variants_cannot_replace_independent_causal_groups(tmp_path):
    rows, pairs = generate(set(), groups=1)
    for row in rows:
        row['group'] = row['split'] + '-one-group'
    for p in pairs:
        p['group'] = rows[p['a']]['group']
    circuit = tmp_path / 'library.dl'
    circuit.write_text(library())
    result = check(ROOT / 'dl/copy_prefix_select.dl', circuit, staged(rows, pairs), {})
    assert result['certified'], result.get('error')
    assert not result['relations']['cp_selected']


def test_token_only_guards_abstain_on_conflicts_repeated_targets_and_long_matches(tmp_path):
    contexts = [
        [10, 11, 12, 13, 10, 11],  # unique two-token source, target 12
        [10, 11, 12, 10, 11, 13, 10, 11],  # conflicting successors
        [10, 11, 12, 13, 12, 10, 11],  # successor occurs twice
        [9, 10, 11, 12, 13, 9, 10, 11],  # a complete three-token match exists
        [10, 11, 12, 13],  # no earlier suffix
    ]
    facts = {"tok": "".join(f"{i}\t{p}\t{t}\n" for i, ctx in enumerate(contexts) for p, t in enumerate(ctx)),
             "sample": "".join(f"{i}\ttest\tg{i}\n" for i in range(len(contexts)))}
    circuit = tmp_path / 'library.dl'
    circuit.write_text(library())
    result = check(ROOT / 'dl/copy_prefix_domains.dl', circuit, facts, {})
    assert result['certified'], result.get('error')
    assert result['relations']['firing'] == [['1', '0', '12'], ['4', '0', '12']]
    assert result['relations']['incremental'] == [['1', '0'], ['4', '0']]
