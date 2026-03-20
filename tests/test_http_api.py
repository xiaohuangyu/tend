import pathlib
import sys
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tend.http_api import create_app
from tend.main_api import JobState, TendAPI


class TestHTTPAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TendAPI()
        self.client = TestClient(create_app(self.api))

    def tearDown(self) -> None:
        self.client.close()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> tuple[int, dict]:
        response = self.client.request(method=method, url=path, json=payload)
        return response.status_code, response.json()

    def test_datasets_route_chain(self) -> None:
        status, dataset = self._request(
            "POST",
            "/api/v1/datasets",
            {"name": "mnist", "task_type": "classification", "labels": ["digit"]},
        )
        self.assertEqual(status, 201)
        dataset_id = dataset["id"]

        status, fetched = self._request("GET", f"/api/v1/datasets/{dataset_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["name"], "mnist")

        status, version = self._request(
            "POST",
            f"/api/v1/datasets/{dataset_id}/versions",
            {
                "version": "v1",
                "manifest_uri": "minio://datasets/mnist/v1/manifest.json",
                "size_bytes": 2048,
                "checksum": "sha256:mnist",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(version["dataset_id"], dataset_id)

    def test_jobs_route_chain(self) -> None:
        _, dataset = self._request(
            "POST",
            "/api/v1/datasets",
            {"name": "imagenet", "task_type": "classification"},
        )
        _, version = self._request(
            "POST",
            f"/api/v1/datasets/{dataset['id']}/versions",
            {
                "version": "v1",
                "manifest_uri": "minio://datasets/imagenet/v1/manifest.json",
                "size_bytes": 4096,
                "checksum": "sha256:imagenet",
            },
        )
        status, job = self._request(
            "POST",
            "/api/v1/jobs",
            {
                "name": "resnet-train",
                "dataset_version_id": version["id"],
                "config": {"lr": 0.001},
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(job["state"], JobState.QUEUED.value)
        job_id = job["id"]

        _, job = self._request(
            "POST",
            f"/api/v1/jobs/{job_id}/events",
            {"event": "scheduler_tick"},
        )
        self.assertEqual(job["state"], JobState.MATCHING.value)

    def test_nodes_route_chain(self) -> None:
        status, node = self._request(
            "POST",
            "/api/v1/nodes",
            {
                "hostname": "worker-1",
                "ip": "192.168.1.101",
                "gpus": [{"index": 0, "model": "RTX4090", "vram_mb": 24576}],
            },
        )
        self.assertEqual(status, 201)
        node_id = node["id"]

        _, node = self._request(
            "POST",
            f"/api/v1/nodes/{node_id}/drain",
            {},
        )
        self.assertEqual(node["status"], "draining")

        _, node = self._request(
            "POST",
            f"/api/v1/nodes/{node_id}/resume",
            {},
        )
        self.assertEqual(node["status"], "online")

    def test_openapi_and_docs_available(self) -> None:
        status, openapi = self._request("GET", "/openapi.json")
        self.assertEqual(status, 200)
        self.assertEqual(openapi["openapi"], "3.1.0")
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
