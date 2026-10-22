from types import SimpleNamespace

import pytest


def test_targeted_chunks_keep_exact_update_budget_without_duplicate_resume_iteration():
    from train_v151_targeted import checkpoint_chunks
    assert list(checkpoint_chunks()) == [(25,25),(50,25),(100,50)]
    assert sum(count for _,count in checkpoint_chunks()) == 100


@pytest.mark.parametrize('override',[{'max_iterations':500},{'num_envs':1024},{'resume':False},{'max_delay_steps':0},{'seed':42}])
def test_preflight_rejects_unreviewed_runtime_budget_or_inputs(override):
    from train_v151_targeted import validate_pilot_args
    args=SimpleNamespace(max_iterations=100,num_envs=512,resume=True,max_delay_steps=3,seed=47)
    validate_pilot_args(args)
    for k,v in override.items(): setattr(args,k,v)
    with pytest.raises(ValueError): validate_pilot_args(args)


def test_terrain_restriction_preserves_rough_geometry_and_matches_evaluation_difficulty():
    from train_v151_targeted import restrict_terrain
    rough=SimpleNamespace(proportion=.1,noise_range=(.02,.10),noise_step=.02,border_width=.25)
    generator=SimpleNamespace(num_rows=10,curriculum=True,sub_terrains={'random_rough':rough,'stairs':object()})
    contract=restrict_terrain(generator)
    assert set(generator.sub_terrains)=={'flat','random_rough'}
    assert generator.sub_terrains['flat'].noise_range==(0.,0.)
    assert generator.sub_terrains['random_rough'].noise_range==(.02,.10)
    assert rough.proportion==.1  # inherited terrain object was not mutated
    assert generator.sub_terrains['flat'].proportion==generator.sub_terrains['random_rough'].proportion==.5
    assert generator.difficulty_range==(4/9,4/9)
    assert generator.num_rows==1 and generator.curriculum is False
    assert contract['stairs_validated'] is False


@pytest.mark.parametrize('kind,values',[
    ('V14LaneVelocityCommand',[[0.,.4,0.]]),
    ('UniformVelocityCommand',[[0.,.4,.1]]),
    ('UniformVelocityCommand',[[0.,0.,0.]]),
    ('UniformVelocityCommand',[[0.,float('nan'),0.]])])
def test_actual_command_contract_rejects_lane_feedback_drift_stand_or_nonfinite(kind,values):
    from train_v151_targeted import validate_actual_command
    assert validate_actual_command('UniformVelocityCommand',[[0.,.400000006,0.]])['command_class']=='UniformVelocityCommand'
    with pytest.raises(ValueError): validate_actual_command(kind,values)
