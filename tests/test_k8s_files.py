"""Validate Kubernetes manifests and GitHub Actions CI workflow against the Stage 3 spec."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
K8S = ROOT / "k8s"


def _load_k8s(name: str) -> dict:
    return yaml.safe_load((K8S / name).read_text(encoding="utf-8"))


def test_deployment_matches_the_client_spec():
    dep = _load_k8s("deployment.yaml")
    assert dep["kind"] == "Deployment"
    assert dep["metadata"]["name"] == "mp024-api"
    assert dep["spec"]["replicas"] == 2
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp024-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000


def test_probes_use_health_and_liveness_waits_longer():
    container = _load_k8s("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]
    ready, live = container["readinessProbe"], container["livenessProbe"]
    assert ready["httpGet"]["path"] == "/health" and ready["httpGet"]["port"] == 8000
    assert live["httpGet"]["path"] == "/health" and live["httpGet"]["port"] == 8000
    assert live["initialDelaySeconds"] > ready["initialDelaySeconds"]


def test_service_selector_matches_pod_labels_and_ports():
    dep, svc = _load_k8s("deployment.yaml"), _load_k8s("service.yaml")
    assert svc["kind"] == "Service"
    assert svc["metadata"]["name"] == "mp024-svc"
    pod_labels = dep["spec"]["template"]["metadata"]["labels"]
    assert all(pod_labels.get(k) == v for k, v in svc["spec"]["selector"].items())
    assert dep["spec"]["selector"]["matchLabels"].items() <= pod_labels.items()
    port = svc["spec"]["ports"][0]
    assert port["port"] == 80 and port["targetPort"] == 8000


def test_ci_workflow_runs_pytest_on_push():
    raw = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert raw.lstrip().startswith("on:")  # keep literal `on:` key in workflow YAML
    workflow = yaml.safe_load(raw)
    triggers = workflow.get("on", workflow.get(True))
    assert "push" in triggers
    assert "requirements.txt" in raw
    assert "pytest" in raw
