from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class JobState(str, Enum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    MATCHING = "MATCHING"
    ALLOCATED = "ALLOCATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"


class JobEvent(str, Enum):
    SUBMIT = "submit"
    SCHEDULER_TICK = "scheduler_tick"
    RESOURCE_FOUND = "resource_found"
    NO_FIT_RETRY = "no_fit_retry"
    NODE_ACK = "node_ack"
    PROCESS_STARTED = "process_started"
    EXIT_SUCCESS = "exit_success"
    RUNTIME_ERROR = "runtime_error"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    USER_CANCEL = "user_cancel"
    RETRY = "retry"


@dataclass(frozen=True)
class Transition:
    from_state: JobState
    event: JobEvent
    to_state: JobState


@dataclass(frozen=True)
class StateEvent:
    from_state: JobState
    event: JobEvent


@dataclass(frozen=True)
class TransitionRecord:
    from_state: JobState
    to_state: JobState
    event: JobEvent
    occurred_at: datetime
    payload: dict[str, Any]


DEFAULT_TRANSITIONS: tuple[Transition, ...] = (
    Transition(JobState.DRAFT, JobEvent.SUBMIT, JobState.QUEUED),
    Transition(JobState.QUEUED, JobEvent.SCHEDULER_TICK, JobState.MATCHING),
    Transition(JobState.MATCHING, JobEvent.RESOURCE_FOUND, JobState.ALLOCATED),
    Transition(JobState.MATCHING, JobEvent.NO_FIT_RETRY, JobState.QUEUED),
    Transition(JobState.ALLOCATED, JobEvent.NODE_ACK, JobState.STARTING),
    Transition(JobState.STARTING, JobEvent.PROCESS_STARTED, JobState.RUNNING),
    Transition(JobState.RUNNING, JobEvent.EXIT_SUCCESS, JobState.SUCCEEDED),
    Transition(JobState.RUNNING, JobEvent.RUNTIME_ERROR, JobState.FAILED),
    Transition(JobState.RUNNING, JobEvent.DEADLINE_EXCEEDED, JobState.TIMEOUT),
    Transition(JobState.FAILED, JobEvent.RETRY, JobState.QUEUED),
    Transition(JobState.TIMEOUT, JobEvent.RETRY, JobState.QUEUED),
    Transition(JobState.QUEUED, JobEvent.USER_CANCEL, JobState.CANCELED),
    Transition(JobState.MATCHING, JobEvent.USER_CANCEL, JobState.CANCELED),
    Transition(JobState.ALLOCATED, JobEvent.USER_CANCEL, JobState.CANCELED),
    Transition(JobState.STARTING, JobEvent.USER_CANCEL, JobState.CANCELED),
    Transition(JobState.RUNNING, JobEvent.USER_CANCEL, JobState.CANCELED),
)


class InvalidTransitionError(ValueError):
    pass


@dataclass
class TrainingJobStateMachine:
    state: JobState = JobState.DRAFT

    def __post_init__(self) -> None:
        self._transitions = {
            StateEvent(item.from_state, item.event): item.to_state
            for item in DEFAULT_TRANSITIONS
        }
        self.history: list[TransitionRecord] = []

    def can_apply(self, event: JobEvent) -> bool:
        return StateEvent(self.state, event) in self._transitions

    def apply(self, event: JobEvent, payload: dict[str, Any] | None = None) -> JobState:
        key = StateEvent(self.state, event)
        if key not in self._transitions:
            raise InvalidTransitionError(
                f"invalid transition: {self.state.value} --{event.value}--> ?"
            )
        next_state = self._transitions[key]
        record = TransitionRecord(
            from_state=self.state,
            to_state=next_state,
            event=event,
            occurred_at=datetime.now(UTC),
            payload=payload or {},
        )
        self.history.append(record)
        self.state = next_state
        return self.state

