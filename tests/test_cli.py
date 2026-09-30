"""Exercise the user-facing count argument without loading GPU backends."""

import signal

import pytest
import yaml

from advv import cli
from advv.reporting import records_for
from advv.storage import read_json
from conftest import FakeBackend


@pytest.fixture(autouse=True)
def restore_sigterm_handler():
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


@pytest.mark.parametrize("quantity, target", [(["3"], 3), (["--target-count", "2"], 2), ([], 2)])
def test_cli_total_count_and_resume(cfg, tmp_path, monkeypatch, quantity, target):
    cfg["run"]["target_count"] = 2  # Positional and flag forms override the YAML value.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    instances = []

    class Backend(FakeBackend):
        def __init__(self, resolved, root):
            count = resolved["run"]["target_count"]
            # One NO and one pixel-identical source must not consume the accepted quota.
            super().__init__(
                ["NO"] + ["YES"] * (2 * (count + 1)),
                [(10, 20, 30), "source"] + [(180, i + 1, 60) for i in range(count)],
            )
            instances.append(self)

    monkeypatch.setattr(cli, "LocalBackend", Backend)
    monkeypatch.setattr(cli, "preflight", lambda *args, **kwargs: {"status": "ready"})
    assert (
        cli.main(
            [
                "run",
                *quantity,
                "--config",
                str(config_path),
                "--gpus",
                "0,1",
                "--run-id",
                "cli_count",
            ]
        )
        == 0
    )
    root = tmp_path / "runs/cli_count"
    state = read_json(root / "state.json")
    assert state["status"] == "completed"
    assert state["accepted_count"] == target
    assert state["attempts"] == target + 2
    assert read_json(root / "config.json")["run"]["target_count"] == target
    assert len(list((root / "accepted").glob("*.png"))) == target
    assert all(r["backend"] == "fake" for r in records_for(root))

    assert cli.main(["run", "--resume", "--run-dir", str(root), "--gpus", "2,3"]) == 0
    assert read_json(root / "state.json")["accepted_count"] == target
    assert instances[-1].generations == [] and instances[-1].calls == []


@pytest.mark.parametrize(
    "quantity, message",
    [
        (["0"], "positive integer"),
        (["-1"], "positive integer"),
        (["1.5"], "positive integer"),
        (["abc"], "positive integer"),
        (["--target-count", "0"], "positive integer"),
        (["--target-count", "-1"], "positive integer"),
        (["--target-count", "1.5"], "positive integer"),
        (["2", "--target-count", "3"], "not both"),
        (["2", "--target-count", "2"], "not both"),
        (["2", "--resume"], "saved target count"),
        (["--target-count", "2", "--resume"], "saved target count"),
    ],
)
def test_bad_count_fails_before_loading_resources(monkeypatch, capsys, quantity, message):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before config/model/run access")

    for name in ("load_config", "read_json", "preflight", "create_run", "LocalBackend"):
        monkeypatch.setattr(cli, name, forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *quantity, "--gpus", "0,1"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
