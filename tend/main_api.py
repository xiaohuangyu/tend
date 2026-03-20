from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from tend.scheduler_state_machine import JobEvent, JobState, TrainingJobStateMachine


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Dataset:
    id: str
    name: str
    task_type: str
    labels: list[str]
    status: str
    created_at: str
    updated_at: str


@dataclass
class DatasetVersion:
    id: str
    dataset_id: str
    version: str
    manifest_uri: str
    size_bytes: int
    checksum: str
    created_at: str


@dataclass
class Node:
    id: str
    hostname: str
    ip: str
    gpus: list[dict[str, Any]]
    labels: dict[str, str]
    status: str
    last_heartbeat_at: str
    created_at: str


@dataclass
class Job:
    id: str
    name: str
    dataset_version_id: str
    priority: int
    config: dict[str, Any]
    state: JobState
    node_id: str | None
    created_at: str
    updated_at: str
    machine: TrainingJobStateMachine


class TendAPIError(ValueError):
    pass


class TendAPI:
    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._dataset_versions: dict[str, DatasetVersion] = {}
        self._dataset_version_index: dict[str, list[str]] = {}
        self._nodes: dict[str, Node] = {}
        self._jobs: dict[str, Job] = {}

    def create_dataset(
        self,
        name: str,
        task_type: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        dataset = Dataset(
            id=str(uuid4()),
            name=name,
            task_type=task_type,
            labels=labels or [],
            status="active",
            created_at=now,
            updated_at=now,
        )
        self._datasets[dataset.id] = dataset
        self._dataset_version_index[dataset.id] = []
        return self._to_dataset_dict(dataset)

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self._to_dataset_dict(item) for item in self._datasets.values()]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise TendAPIError(f"dataset not found: {dataset_id}")
        return self._to_dataset_dict(dataset)

    def create_dataset_version(
        self,
        dataset_id: str,
        version: str,
        manifest_uri: str,
        size_bytes: int,
        checksum: str,
    ) -> dict[str, Any]:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise TendAPIError(f"dataset not found: {dataset_id}")
        now = _utc_now_iso()
        version_item = DatasetVersion(
            id=str(uuid4()),
            dataset_id=dataset_id,
            version=version,
            manifest_uri=manifest_uri,
            size_bytes=size_bytes,
            checksum=checksum,
            created_at=now,
        )
        self._dataset_versions[version_item.id] = version_item
        self._dataset_version_index[dataset_id].append(version_item.id)
        dataset.updated_at = now
        return self._to_dataset_version_dict(version_item)

    def list_dataset_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        if dataset_id not in self._datasets:
            raise TendAPIError(f"dataset not found: {dataset_id}")
        version_ids = self._dataset_version_index.get(dataset_id, [])
        return [
            self._to_dataset_version_dict(self._dataset_versions[item_id])
            for item_id in version_ids
        ]

    def create_job(
        self,
        name: str,
        dataset_version_id: str,
        config: dict[str, Any],
        priority: int = 5,
    ) -> dict[str, Any]:
        if dataset_version_id not in self._dataset_versions:
            raise TendAPIError(f"dataset version not found: {dataset_version_id}")
        now = _utc_now_iso()
        machine = TrainingJobStateMachine()
        machine.apply(JobEvent.SUBMIT)
        job = Job(
            id=str(uuid4()),
            name=name,
            dataset_version_id=dataset_version_id,
            priority=priority,
            config=config,
            state=machine.state,
            node_id=None,
            created_at=now,
            updated_at=now,
            machine=machine,
        )
        self._jobs[job.id] = job
        return self._to_job_dict(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [self._to_job_dict(item) for item in self._jobs.values()]

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise TendAPIError(f"job not found: {job_id}")
        return self._to_job_dict(job)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._apply_job_event(job_id, JobEvent.USER_CANCEL)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        return self._apply_job_event(job_id, JobEvent.RETRY)

    def advance_job(self, job_id: str, event: JobEvent) -> dict[str, Any]:
        return self._apply_job_event(job_id, event)

    def register_node(
        self,
        hostname: str,
        ip: str,
        gpus: list[dict[str, Any]],
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        node = Node(
            id=str(uuid4()),
            hostname=hostname,
            ip=ip,
            gpus=gpus,
            labels=labels or {},
            status="online",
            last_heartbeat_at=now,
            created_at=now,
        )
        self._nodes[node.id] = node
        return self._to_node_dict(node)

    def heartbeat_node(
        self,
        node_id: str,
        gpus: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        node = self._nodes.get(node_id)
        if node is None:
            raise TendAPIError(f"node not found: {node_id}")
        node.last_heartbeat_at = _utc_now_iso()
        if gpus is not None:
            node.gpus = gpus
        if node.status != "draining":
            node.status = "online"
        return self._to_node_dict(node)

    def list_nodes(self) -> list[dict[str, Any]]:
        return [self._to_node_dict(item) for item in self._nodes.values()]

    def drain_node(self, node_id: str) -> dict[str, Any]:
        node = self._nodes.get(node_id)
        if node is None:
            raise TendAPIError(f"node not found: {node_id}")
        node.status = "draining"
        node.last_heartbeat_at = _utc_now_iso()
        return self._to_node_dict(node)

    def resume_node(self, node_id: str) -> dict[str, Any]:
        node = self._nodes.get(node_id)
        if node is None:
            raise TendAPIError(f"node not found: {node_id}")
        node.status = "online"
        node.last_heartbeat_at = _utc_now_iso()
        return self._to_node_dict(node)

    def _apply_job_event(self, job_id: str, event: JobEvent) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise TendAPIError(f"job not found: {job_id}")
        job.machine.apply(event)
        job.state = job.machine.state
        job.updated_at = _utc_now_iso()
        return self._to_job_dict(job)

    def _to_dataset_dict(self, dataset: Dataset) -> dict[str, Any]:
        return {
            "id": dataset.id,
            "name": dataset.name,
            "task_type": dataset.task_type,
            "labels": dataset.labels,
            "status": dataset.status,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
        }

    def _to_dataset_version_dict(self, version: DatasetVersion) -> dict[str, Any]:
        return {
            "id": version.id,
            "dataset_id": version.dataset_id,
            "version": version.version,
            "manifest_uri": version.manifest_uri,
            "size_bytes": version.size_bytes,
            "checksum": version.checksum,
            "created_at": version.created_at,
        }

    def _to_node_dict(self, node: Node) -> dict[str, Any]:
        return {
            "id": node.id,
            "hostname": node.hostname,
            "ip": node.ip,
            "gpus": node.gpus,
            "labels": node.labels,
            "status": node.status,
            "last_heartbeat_at": node.last_heartbeat_at,
            "created_at": node.created_at,
        }

    def _to_job_dict(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "name": job.name,
            "dataset_version_id": job.dataset_version_id,
            "priority": job.priority,
            "config": job.config,
            "state": job.state.value,
            "node_id": job.node_id,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "history": [
                {
                    "from_state": item.from_state.value,
                    "to_state": item.to_state.value,
                    "event": item.event.value,
                    "occurred_at": item.occurred_at.isoformat(),
                    "payload": item.payload,
                }
                for item in job.machine.history
            ],
        }
