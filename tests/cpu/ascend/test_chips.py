"""Chip data: the extracted YAML is loadable and complete enough for kernel generation."""
from infrarsi.ascend import chips


def test_chips_load():
    cards = chips.all_cards()
    assert cards, 'infrarsi/ascend/data/chips.yaml is missing or empty'
    assert '910B2' in cards
    assert '950PR-Server' in cards or '950PR' in cards


def test_910b2_constraints():
    c = chips.get('Ascend910B2')
    assert c is not None
    assert c.npu_arch == 'dav-2201'
    assert c.memory_gib == 64
    con = chips.constraints(c)
    assert con['dma_alignment_bytes'] == 32
    assert con['buffers_kb']['l1_kb'] == 512
    assert con['buffers_kb']['ub_kb'] == 192
    assert con['buffers_kb']['l0c_kb'] == 128
    assert con['cube_mac'] == '16x16x16'


def test_prompt_block_mentions_arch_and_sources():
    c = chips.get('910B2')
    text = chips.prompt_block(c)
    assert 'dav-2201' in text
    assert 'L1=512' in text or 'l1_kb=512' in text
    assert 'npu-hardware-params.md' in text


def test_unknown_card_is_none():
    assert chips.get('not-a-card') is None
