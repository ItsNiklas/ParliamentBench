from typing import Optional

LIBERAL_POLICIES = 6
FASCIST_POLICIES = 11
LIBERAL_POLICIES_TO_WIN = 5
FASCIST_POLICIES_TO_WIN = 6
PLAYER_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
]

PLAYERS: dict[int, tuple[int, int, list[Optional[str]]]] = {
    5: (3, 1, [None, None, "policy", "kill", "kill", None]),
    6: (4, 1, [None, None, "policy", "kill", "kill", None]),
    7: (4, 2, [None, "inspect", "choose", "kill", "kill", None]),
    8: (5, 2, [None, "inspect", "choose", "kill", "kill", None]),
    9: (5, 3, ["inspect", "inspect", "choose", "kill", "kill", None]),
    10: (6, 3, ["inspect", "inspect", "choose", "kill", "kill", None]),
}


class Role(object):
    """
    Parent role. Inherited by three subclasses: Liberal, Fascist, Hitler.
    """

    def __init__(self) -> None:
        self.party_membership = ""
        self.role = ""

    def __repr__(self) -> str:
        return self.role.title()


class LiberalRole(Role):
    def __init__(self) -> None:
        super(LiberalRole, self).__init__()
        self.party_membership = "liberal"
        self.role = "liberal"


class FascistRole(Role):
    def __init__(self) -> None:
        super(FascistRole, self).__init__()
        self.party_membership = "fascist"
        self.role = "fascist"


class LeaderRole(Role):
    def __init__(self) -> None:
        super(LeaderRole, self).__init__()
        self.party_membership = "fascist"
        self.role = "hitler"


class Policy(object):
    def __init__(self, type: str) -> None:
        self.type = type

    def __repr__(self) -> str:
        return self.type.title()

    def __str__(self) -> str:
        return self.type.title()


class LiberalPolicy(Policy):
    def __init__(self) -> None:
        super(LiberalPolicy, self).__init__("liberal")


class FascistPolicy(Policy):
    def __init__(self) -> None:
        super(FascistPolicy, self).__init__("fascist")


class Vote(object):
    def __init__(self, type: bool) -> None:
        self.type = type

    def __nonzero__(self) -> bool:
        return self.type

    def __bool__(self) -> bool:
        return self.type


class Ja(Vote):
    def __init__(self) -> None:
        super(Ja, self).__init__(True)


class Nein(Vote):
    def __init__(self) -> None:
        super(Nein, self).__init__(False)
