"""Byte metadata agrees with ByteLevel decoding; diagnostic classes stay distinct."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from qwen_token_bytes import byte_decoder


def test_byte_alphabet_matches_bytelevel_decoder():
    decoders=pytest.importorskip('tokenizers.decoders')
    reverse=byte_decoder();forward={b:c for c,b in reverse.items()}
    assert len(reverse)==256 and set(reverse.values())==set(range(256))
    for text in ('Hello world','世界🙂','café\n\t'):
        encoded=''.join(forward[b] for b in text.encode('utf-8'))
        assert bytes(reverse[c] for c in encoded)==text.encode('utf-8')
        assert decoders.ByteLevel().decode([encoded])==text


def test_prefix_diagnostic_distinguishes_fusion_split_and_other_outputs(tmp_path):
    m=ROOT/'dl/qwen35_copy_bytes.dl'
    expected=[1,2,2,2,2,2,2];predicted=[1,3,4,5,6,7,8]
    output={1:b'a',3:b'abc',4:b'a',5:b'z',6:b'STOP',8:b'ab'}
    facts={'target':''.join(f'{i}\t{t}\n' for i,t in enumerate(expected)),
        'ref':''.join(f'{i}\t{t}\n' for i,t in enumerate(predicted)),
        'token_len':'1\t1\n2\t2\n3\t3\n4\t1\n5\t1\n6\t4\n8\t2\n',
        'special':'6\n','source':''.join(f'{i}\t0\n' for i in range(7)),
        'tail_byte':''.join(f'{i}\t0\t{j}\t{b}\n' for i in range(7) for j,b in enumerate(b'abc')),
        'output_byte':''.join(f'{i}\t{j}\t{b}\n' for i,t in enumerate(predicted) for j,b in enumerate(output.get(t,b'')))}
    x=check(m,m,facts,{},evidence_dir=tmp_path/'diagnosis')
    assert x['certified'],x
    assert dict(x['relations']['class'])=={'0':'literal_id','1':'fused_source_tokens','2':'split_source_token',
        '3':'not_source_byte_prefix','4':'special_output','5':'unassigned_output','6':'same_bytes_other_id'}
