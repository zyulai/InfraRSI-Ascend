"""Launch-knob catalogue and registry merge."""
from infrarsi.ascend import knobs
from infrarsi.experience.registry import gpu_class, mechanisms


def test_env_choices():
    ch = knobs.env_choices()
    assert ch['TASK_QUEUE_ENABLE'] == ['0', '1', '2']
    assert 'HCCL_BUFFSIZE' in ch
    assert all(len(v) >= 2 for v in ch.values())


def test_mechanism_mapping():
    assert knobs.mechanism_of('TASK_QUEUE_ENABLE') == 'npu_task_queue'
    assert 'HCCL' in knobs.notes_of('HCCL_BUFFSIZE')


def test_registry_merges_ascend_gpus():
    assert gpu_class('Ascend910B2') == '910B2'
    assert gpu_class('910B3') == '910B3'
    assert gpu_class('Ascend910_93') == '910_93'


def test_registry_mechanisms_have_ascend_entries():
    reg = mechanisms()
    assert 'npu_task_queue' in reg
    assert reg['npu_task_queue']['states'] == ['0', '1', '2']
