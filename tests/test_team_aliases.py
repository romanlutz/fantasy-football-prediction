from ffpred.acquisition.contracts import RELOCATED_TEAM_CODES, normalize_team_code


def test_normalize_team_code_maps_every_known_relocation() -> None:
    assert normalize_team_code("STL") == "LA"
    assert normalize_team_code("SD") == "LAC"
    assert normalize_team_code("OAK") == "LV"


def test_normalize_team_code_is_identity_for_unrelocated_teams() -> None:
    assert normalize_team_code("KC") == "KC"
    assert normalize_team_code("SEA") == "SEA"
    # Already-current codes for relocated franchises must also pass through
    # unchanged, since schedules for recent seasons already report them.
    assert normalize_team_code("LA") == "LA"
    assert normalize_team_code("LAC") == "LAC"
    assert normalize_team_code("LV") == "LV"


def test_relocated_team_codes_map_to_distinct_current_codes() -> None:
    assert len(set(RELOCATED_TEAM_CODES.values())) == len(RELOCATED_TEAM_CODES)
