import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tend.main_api import JobEvent, JobState, TendAPI, TendAPIError


class TestDatasetsAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TendAPI()

    def test_create_list_get_dataset(self) -> None:
        created = self.api.create_dataset(
            name="mnist",
            task_type="classification",
            labels=["digit"],
        )
        datasets = self.api.list_datasets()
        fetched = self.api.get_dataset(created["id"])
        self.assertEqual(len(datasets), 1)
        self.assertEqual(fetched["name"], "mnist")
        self.assertEqual(fetched["task_type"], "classification")

    def test_create_dataset_version(self) -> None:
        dataset = self.api.create_dataset(name="coco", task_type="detection")
        version = self.api.create_dataset_version(
            dataset_id=dataset["id"],
            version="v1",
            manifest_uri="minio://datasets/coco/v1/manifest.json",
            size_bytes=1024,
            checksum="sha256:abc",
        )
        versions = self.api.list_dataset_versions(dataset["id"])
        self.assertEqual(version["dataset_id"], dataset["id"])
        self.assertEqual(len(versions), 1)

    def test_create_dataset_version_with_missing_dataset(self) -> None:
        with self.assertRaises(TendAPIError):
            self.api.create_dataset_version(
                dataset_id="missing",
                version="v1",
                manifest_uri="minio://datasets/missing/v1/manifest.json",
                size_bytes=1,
                checksum="sha256:x",
            )


class TestJobsAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TendAPI()
        dataset = self.api.create_dataset(name="imagenet", task_type="classification")
        self.dataset_version = self.api.create_dataset_version(
            dataset_id=dataset["id"],
            version="v1",
            manifest_uri="minio://datasets/imagenet/v1/manifest.json",
            size_bytes=4096,
            checksum="sha256:job",
        )

    def test_create_and_list_jobs(self) -> None:
        job = self.api.create_job(
            name="resnet50-train",
            dataset_version_id=self.dataset_version["id"],
            config={"lr": 0.001, "batch_size": 32},
            priority=3,
        )
        jobs = self.api.list_jobs()
        self.assertEqual(job["state"], JobState.QUEUED.value)
        self.assertEqual(len(job["history"]), 1)
        self.assertEqual(len(jobs), 1)

    def test_advance_cancel_and_retry_job(self) -> None:
        job = self.api.create_job(
            name="vit-train",
            dataset_version_id=self.dataset_version["id"],
            config={"lr": 0.0003},
        )
        job = self.api.advance_job(job["id"], JobEvent.SCHEDULER_TICK)
        job = self.api.advance_job(job["id"], JobEvent.RESOURCE_FOUND)
        job = self.api.advance_job(job["id"], JobEvent.NODE_ACK)
        job = self.api.advance_job(job["id"], JobEvent.PROCESS_STARTED)
        job = self.api.advance_job(job["id"], JobEvent.RUNTIME_ERROR)
        self.assertEqual(job["state"], JobState.FAILED.value)
        job = self.api.retry_job(job["id"])
        self.assertEqual(job["state"], JobState.QUEUED.value)
        job = self.api.cancel_job(job["id"])
        self.assertEqual(job["state"], JobState.CANCELED.value)

    def test_create_job_with_missing_dataset_version(self) -> None:
        with self.assertRaises(TendAPIError):
            self.api.create_job(
                name="bad-job",
                dataset_version_id="missing",
                config={},
            )


class TestNodesAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TendAPI()

    def test_register_list_heartbeat_node(self) -> None:
        node = self.api.register_node(
            hostname="worker-1",
            ip="192.168.1.101",
            gpus=[{"index": 0, "model": "RTX4090", "vram_mb": 24576}],
            labels={"zone": "lab-a"},
        )
        nodes = self.api.list_nodes()
        heartbeated = self.api.heartbeat_node(
            node["id"],
            gpus=[{"index": 0, "model": "RTX4090", "vram_mb": 24576, "util": 55}],
        )
        self.assertEqual(len(nodes), 1)
        self.assertEqual(heartbeated["status"], "online")
        self.assertEqual(heartbeated["hostname"], "worker-1")

    def test_drain_and_resume_node(self) -> None:
        node = self.api.register_node(
            hostname="worker-2",
            ip="192.168.1.102",
            gpus=[{"index": 0, "model": "A5000", "vram_mb": 24576}],
        )
        drained = self.api.drain_node(node["id"])
        resumed = self.api.resume_node(node["id"])
        self.assertEqual(drained["status"], "draining")
        self.assertEqual(resumed["status"], "online")

    def test_node_not_found_errors(self) -> None:
        with self.assertRaises(TendAPIError):
            self.api.heartbeat_node("missing")
        with self.assertRaises(TendAPIError):
            self.api.drain_node("missing")
        with self.assertRaises(TendAPIError):
            self.api.resume_node("missing")


if __name__ == "__main__":
    unittest.main()
