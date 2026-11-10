"""Token-level wiring with a real renderer (needs network for the public tokenizer).

Covers the path the Tinker training loop takes: ELTDatasetBuilder ->
ELTEnvGroupBuilder.make_envs -> EnvFromMessageEnv.initial_observation, plus
construction of the training Config. No Tinker or warehouse credentials needed.
"""

from __future__ import annotations

import pytest
from tinker_cookbook import tokenizer_utils

from conftest import REPO
from elt_rl.train import CLIConfig, build_config

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return tokenizer_utils.get_tokenizer(MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable (offline?): {e}")


async def test_dataset_to_token_env(tokenizer, tmp_path):
    cfg = CLIConfig(
        tasks_root=str(REPO / "tasks"), model_name=MODEL, group_size=2, groups_per_batch=1,
        max_steps=1, log_path=str(tmp_path / "run"), behavior_if_log_dir_exists="delete",
    )
    config = build_config(cfg)
    assert config.model_name == MODEL and config.max_steps == 1

    train_set, test_set = await config.dataset_builder()
    assert len(train_set) == 1 and test_set is None
    (builder,) = train_set.get_batch(0)
    envs = await builder.make_envs()
    try:
        assert len(envs) == 2
        ob, stop = await envs[0].initial_observation()
        prompt = tokenizer.decode(ob.to_ints())
        for name in ("read_file", "write_file", "sql", "run_models", "submit"):
            assert name in prompt
        assert "bash" not in [t.name for t in builder._rollouts[0].tools.tool_list()]
        assert "toy_shop" in prompt and stop
    finally:
        await builder.cleanup()
    assert not builder._rollouts


def test_rollout_script_config():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_rollout", REPO / "scripts" / "run_rollout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_rollout"] = module  # chz resolves annotations via sys.modules
    spec.loader.exec_module(module)
    cfg = module.RolloutConfig(tasks_root=str(REPO / "tasks"))
    assert cfg.group_size == 1 and cfg.out_dir == "rollouts"
