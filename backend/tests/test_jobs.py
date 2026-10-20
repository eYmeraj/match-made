import pytest

from matchmade.jobs import failure_message


@pytest.mark.parametrize(
    ("exc", "stage", "expected"),
    [
        (
            "Work-horse terminated unexpectedly; waitpid returned 9 (signal 9); ",
            "block",
            "The worker ran out of memory while finding candidate pairs.",
        ),
        (
            "rq.timeouts.JobTimeoutException: Task exceeded maximum timeout",
            "score",
            "The run hit its time limit while scoring.",
        ),
        (None, None, "The run stopped unexpectedly."),
    ],
)
def test_failure_message(exc, stage, expected):
    assert failure_message(exc, stage).startswith(expected)
