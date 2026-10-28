"""Full inventory factors, paired edits and lossless model-free score replay."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from run_qwen35_copy_inventory import generate
from run_copy_prefix import disjoint
from qwen35_reference_evidence import extract,replay_archive


def test_inventory_keeps_single_and_repeated_sources_and_all_edits():
    pools={'query':list(range(1,513)),'target':list(range(1000,2024)),
        'decoy':list(range(3000,4024)),'background':list(range(5000,13192))}
    rows,pairs,used=generate(pools)
    assert len(rows)==300 and len(pairs)==144
    assert len(disjoint(rows,set()))==2
    assert {r['exposures'] for r in rows if not r['negative']}=={1,2,3}
    assert sum(r['negative'] for r in rows)==12
    for p in pairs:
        a,b=rows[p['a']],rows[p['b']]
        assert len(p['positions'])==a['exposures']
        assert [j for j,(t,u) in enumerate(zip(a['ctx'],b['ctx'])) if t!=u]==p['positions']
        assert a['ctx'][-a['prefix']:]==b['ctx'][-b['prefix']:]


def test_reference_archive_keeps_complete_score_witness_and_tie_break(tmp_path):
    class Scores(list):
        def tolist(self):return list(self)
    path=tmp_path/'ref.tar.gz';result=extract(Scores([0.,2.,2.]),path,17)
    assert result['query']['certified'] and result['query']['relations']['ref']==[['1']]
    again=replay_archive(path,result['sha256'],row_id=17)
    assert again['relations']==result['query']['relations']
    with pytest.raises(ValueError):replay_archive(path,result['sha256'],row_id=18)
    with pytest.raises(ValueError):replay_archive(path,'bad-digest')
