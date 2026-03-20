from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from tend.main_api import JobEvent, TendAPI, TendAPIError


class DatasetCreateRequest(BaseModel):
    name: str
    task_type: str
    labels: list[str] | None = None


class DatasetVersionCreateRequest(BaseModel):
    version: str
    manifest_uri: str
    size_bytes: int
    checksum: str


class JobCreateRequest(BaseModel):
    name: str
    dataset_version_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5


class JobEventRequest(BaseModel):
    event: str


class NodeRegisterRequest(BaseModel):
    hostname: str
    ip: str
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    labels: dict[str, str] | None = None


class NodeHeartbeatRequest(BaseModel):
    gpus: list[dict[str, Any]] | None = None


def create_app(api: TendAPI | None = None) -> FastAPI:
    service = api or TendAPI()
    app = FastAPI(title="tend api", version="0.1.0")

    @app.exception_handler(TendAPIError)
    async def tend_api_error_handler(
        request: Request,
        exc: TendAPIError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        if exc.status_code == 404:
            return JSONResponse(
                status_code=404,
                content={"error": f"route not found: {request.url.path}"},
            )
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.post("/api/v1/datasets", status_code=201)
    async def create_dataset(payload: DatasetCreateRequest) -> dict[str, Any]:
        return service.create_dataset(
            name=payload.name,
            task_type=payload.task_type,
            labels=payload.labels,
        )

    @app.get("/api/v1/datasets")
    async def list_datasets() -> dict[str, Any]:
        return {"items": service.list_datasets()}

    @app.get("/api/v1/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str) -> dict[str, Any]:
        return service.get_dataset(dataset_id)

    @app.post("/api/v1/datasets/{dataset_id}/versions", status_code=201)
    async def create_dataset_version(
        dataset_id: str,
        payload: DatasetVersionCreateRequest,
    ) -> dict[str, Any]:
        return service.create_dataset_version(
            dataset_id=dataset_id,
            version=payload.version,
            manifest_uri=payload.manifest_uri,
            size_bytes=payload.size_bytes,
            checksum=payload.checksum,
        )

    @app.get("/api/v1/datasets/{dataset_id}/versions")
    async def list_dataset_versions(dataset_id: str) -> dict[str, Any]:
        return {"items": service.list_dataset_versions(dataset_id)}

    @app.post("/api/v1/jobs", status_code=201)
    async def create_job(payload: JobCreateRequest) -> dict[str, Any]:
        return service.create_job(
            name=payload.name,
            dataset_version_id=payload.dataset_version_id,
            config=payload.config,
            priority=payload.priority,
        )

    @app.get("/api/v1/jobs")
    async def list_jobs() -> dict[str, Any]:
        return {"items": service.list_jobs()}

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return service.get_job(job_id)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        return service.cancel_job(job_id)

    @app.post("/api/v1/jobs/{job_id}/retry")
    async def retry_job(job_id: str) -> dict[str, Any]:
        return service.retry_job(job_id)

    @app.post("/api/v1/jobs/{job_id}/events")
    async def advance_job(job_id: str, payload: JobEventRequest) -> dict[str, Any]:
        return service.advance_job(job_id, JobEvent(payload.event))

    @app.post("/api/v1/nodes", status_code=201)
    async def register_node(payload: NodeRegisterRequest) -> dict[str, Any]:
        return service.register_node(
            hostname=payload.hostname,
            ip=payload.ip,
            gpus=payload.gpus,
            labels=payload.labels,
        )

    @app.get("/api/v1/nodes")
    async def list_nodes() -> dict[str, Any]:
        return {"items": service.list_nodes()}

    @app.post("/api/v1/nodes/{node_id}/heartbeat")
    async def heartbeat_node(
        node_id: str,
        payload: NodeHeartbeatRequest,
    ) -> dict[str, Any]:
        return service.heartbeat_node(node_id, gpus=payload.gpus)

    @app.post("/api/v1/nodes/{node_id}/drain")
    async def drain_node(node_id: str) -> dict[str, Any]:
        return service.drain_node(node_id)

    @app.post("/api/v1/nodes/{node_id}/resume")
    async def resume_node(node_id: str) -> dict[str, Any]:
        return service.resume_node(node_id)

    return app


def run_http_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
