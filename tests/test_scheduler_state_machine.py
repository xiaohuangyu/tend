import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tend.scheduler_state_machine import (
    InvalidTransitionError,
    JobEvent,
    JobState,
    TrainingJobStateMachine,
)


class TestTrainingJobStateMachine(unittest.TestCase):
    def test_happy_path_to_succeeded(self) -> None:
        machine = TrainingJobStateMachine()
        machine.apply(JobEvent.SUBMIT)
        machine.apply(JobEvent.SCHEDULER_TICK)
        machine.apply(JobEvent.RESOURCE_FOUND)
        machine.apply(JobEvent.NODE_ACK)
        machine.apply(JobEvent.PROCESS_STARTED)
        machine.apply(JobEvent.EXIT_SUCCESS)
        self.assertEqual(machine.state, JobState.SUCCEEDED)
        self.assertEqual(len(machine.history), 6)

    def test_cancel_from_running(self) -> None:
        machine = TrainingJobStateMachine()
        machine.apply(JobEvent.SUBMIT)
        machine.apply(JobEvent.SCHEDULER_TICK)
        machine.apply(JobEvent.RESOURCE_FOUND)
        machine.apply(JobEvent.NODE_ACK)
        machine.apply(JobEvent.PROCESS_STARTED)
        machine.apply(JobEvent.USER_CANCEL)
        self.assertEqual(machine.state, JobState.CANCELED)

    def test_retry_after_failed(self) -> None:
        machine = TrainingJobStateMachine()
        machine.apply(JobEvent.SUBMIT)
        machine.apply(JobEvent.SCHEDULER_TICK)
        machine.apply(JobEvent.RESOURCE_FOUND)
        machine.apply(JobEvent.NODE_ACK)
        machine.apply(JobEvent.PROCESS_STARTED)
        machine.apply(JobEvent.RUNTIME_ERROR)
        self.assertEqual(machine.state, JobState.FAILED)
        machine.apply(JobEvent.RETRY)
        self.assertEqual(machine.state, JobState.QUEUED)

    def test_retry_after_timeout(self) -> None:
        machine = TrainingJobStateMachine()
        machine.apply(JobEvent.SUBMIT)
        machine.apply(JobEvent.SCHEDULER_TICK)
        machine.apply(JobEvent.RESOURCE_FOUND)
        machine.apply(JobEvent.NODE_ACK)
        machine.apply(JobEvent.PROCESS_STARTED)
        machine.apply(JobEvent.DEADLINE_EXCEEDED)
        self.assertEqual(machine.state, JobState.TIMEOUT)
        machine.apply(JobEvent.RETRY)
        self.assertEqual(machine.state, JobState.QUEUED)

    def test_no_fit_goes_back_to_queue(self) -> None:
        machine = TrainingJobStateMachine()
        machine.apply(JobEvent.SUBMIT)
        machine.apply(JobEvent.SCHEDULER_TICK)
        machine.apply(JobEvent.NO_FIT_RETRY)
        self.assertEqual(machine.state, JobState.QUEUED)

    def test_invalid_transition_raises(self) -> None:
        machine = TrainingJobStateMachine()
        with self.assertRaises(InvalidTransitionError):
            machine.apply(JobEvent.PROCESS_STARTED)

    def test_can_apply(self) -> None:
        machine = TrainingJobStateMachine()
        self.assertTrue(machine.can_apply(JobEvent.SUBMIT))
        self.assertFalse(machine.can_apply(JobEvent.EXIT_SUCCESS))


if __name__ == "__main__":
    unittest.main()
