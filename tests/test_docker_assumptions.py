"""Docker-related assumptions WITHOUT requiring Docker in CI.
The actual container build/run smoke test lives in scripts/docker_smoke.sh
and is executed manually when a Docker daemon exists."""
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_pyproject_has_runtime_deps():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "fastapi" in text and "httpx" in text
    assert "[project]" in text


def test_app_importable_and_health_offline():
    from fastapi.testclient import TestClient
    from finance_controller.api import create_app
    c = TestClient(create_app())
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_clean_env_install_and_run(tmp_path):
    """Install into an isolated venv from pyproject.toml, then verify
    import + health + deterministic pipeline without credentials."""
    env_dir = tmp_path / "cleanenv"
    venv.create(env_dir, with_pip=True)
    pip = env_dir / ("Scripts" if os.name == "nt" else "bin") / "pip"
    py = env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.run([str(pip), "install", "--quiet", str(REPO)],
                   check=True, capture_output=True)
    code = (
        "import json,os;"
        "os.environ.pop('FINANCE_LLM_API_KEY',None);"
        "from fastapi.testclient import TestClient;"
        "from finance_controller.api import create_app;"
        "c=TestClient(create_app());"
        "assert c.get('/health').json()=={'status':'ok'};"
        "r=c.post('/pipeline/run',json={});"
        "assert r.status_code==200 and r.json()['case_count']==122;"
        "print('CLEAN_ENV_OK')")
    out = subprocess.run([str(py), "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CLEAN_ENV_OK" in out.stdout


def test_dockerfile_exists_non_root_and_no_secrets():
    df = (REPO / "Dockerfile").read_text()
    assert "USER appuser" in df          # non-root
    assert "EXPOSE 8000" in df
    assert "finance_controller.api:app" in df
    for bad in ("sk-", ".env", "API_KEY="):
        assert bad not in df


def test_compose_no_hardcoded_keys():
    dc = (REPO / "docker-compose.yml").read_text()
    assert "sk-" not in dc
    assert "${FINANCE_LLM_API_KEY:-}" in dc   # env passthrough only


def test_cli_still_works_offline(capsys):
    from finance_controller.cli import main
    assert main(["run"]) == 0
    assert "Investigation cases    : 122" in capsys.readouterr().out
