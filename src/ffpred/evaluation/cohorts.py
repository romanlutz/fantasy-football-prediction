"""Named evaluation cohorts retained from the original experiment."""

from ffpred.domain.identifiers import PlayerId

LEGACY_2014_QUARTERBACKS = frozenset(
    PlayerId(player_id)
    for player_id in (
        "00-0029263",
        "00-0023459",
        "00-0026143",
        "00-0020531",
        "00-0026158",
        "00-0027973",
        "00-0024226",
        "00-0023436",
        "00-0029701",
        "00-0019596",
        "00-0031280",
        "00-0022924",
        "00-0026625",
        "00-0021678",
        "00-0027974",
        "00-0010346",
        "00-0029668",
        "00-0026498",
        "00-0022803",
        "00-0022942",
        "00-0027939",
        "00-0031237",
        "00-0031407",
        "00-0023541",
    )
)
