class AnswerOutputLimitError(RuntimeError):
    """A user-facing answer was cut short by the model output limit."""

    def __init__(self, partial_answer: str) -> None:
        super().__init__("Agent answer reached output token limit")
        self.partial_answer = partial_answer
