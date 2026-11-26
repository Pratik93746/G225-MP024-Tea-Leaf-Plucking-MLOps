"""No cluster is needed to check that the Kubernetes files say what the client asked for."""
from pathlib import Path

import yaml

K8S = Path(__file__).resolve().parent.parent / "k8s"


def _load(name):
    return yaml.safe_load((K8S / name).read_text(encoding="utf-8"))


def test_deployment_matches_the_client_spec():
    dep = _load("deployment.yaml")
    assert dep["kind"] == "Deployment" and dep["metadata"]["name"] == "mp024-api"
    assert dep["spec"]["replicas"] == 2
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp024-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000


def test_probes_use_health_and_liveness_waits_longer():
    container = _load("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]
    ready, live = container["readinessProbe"], container["livenessProbe"]
    assert ready["httpGet"]["path"] == "/health" and live["httpGet"]["path"] == "/health"
    assert live["initialDelaySeconds"] > ready["initialDelaySeconds"]


def test_service_selector_matches_pod_labels_and_ports():
    dep, svc = _load("deployment.yaml"), _load("service.yaml")
    assert svc["metadata"]["name"] == "mp024-svc"
    pod_labels = dep["spec"]["template"]["metadata"]["labels"]
    assert all(pod_labels.get(k) == v for k, v in svc["spec"]["selector"].items())
    assert dep["spec"]["selector"]["matchLabels"].items() <= pod_labels.items()
    port = svc["spec"]["ports"][0]
    assert port["port"] == 80 and port["targetPort"] == 8000


def test_ci_workflow_runs_pytest_on_push():
    raw = (K8S.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert raw.lstrip().startswith("on:")        # the key must be literally `on:` (not `true:`)
    assert "pytest" in raw and "push" in raw
