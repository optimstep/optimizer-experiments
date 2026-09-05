from pathlib import Path

from experiments.nanogpt_speedrun.run import command_for, load_config


CONFIG = Path("configs/nanogpt_speedrun/baseline.yaml")


def test_baseline_records_pinned_upstream_and_official_world_size():
    config = load_config(CONFIG)
    assert len(config["upstream"]["revision"]) == 40
    assert config["hardware"]["nproc_per_node"] == 8
    assert config["runtime"]["patch"] is None


def test_run_command_uses_configured_process_count():
    config = load_config(CONFIG)
    assert command_for(config, "run") == [
        "torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"
    ]

