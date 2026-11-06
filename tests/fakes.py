"""Test doubles shared across tests."""


class FakeLLM:
    """Returns queued python objects validated through the requested schema; records messages."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        self.seen.append((stage, messages, vote_index, temperature))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return schema.model_validate(r)
