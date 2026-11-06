"""Reader interface: any show format -> list[Episode] + parse health."""
from dataclasses import dataclass, field
from promofinder.models import Episode, ParseHealth


class ReaderError(Exception):
    pass


@dataclass
class ReadResult:
    episodes: list[Episode]
    health: list[ParseHealth]
    warnings: list[str] = field(default_factory=list)
