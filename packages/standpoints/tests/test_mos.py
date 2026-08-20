"""the three stages, exercised with a fake executor. no network."""

from __future__ import annotations

import pytest
from standpoints import META, OMNI, STANDPOINTS, MoS, describe, for_group


def scripted(*, activate: set[str] | None = None, gap_from: set[str] | None = None):
    """
    a fake executor that records what each stage was asked.

    stages are told apart by which sentinel their prompt mentions, which is the
    same thing the real prompts key on.
    """
    seen: list[list[str]] = []
    activate = activate or set()
    gap_from = gap_from or set()

    def execute(prompts, max_tokens):
        seen.append(list(prompts))
        out = []
        for text in prompts:
            if "[NO_COMMENT]" in text:
                fires = any(META[label].description[:40] in text for label in activate)
                out.append("this would fail me" if fires else "[NO_COMMENT]")
            elif "[NO_GAP]" in text:
                fires = any(STANDPOINTS[label].description in text for label in gap_from)
                out.append("i would need X" if fires else "[NO_GAP]")
            else:
                out.append("the merged answer")
        return out

    execute.seen = seen
    return execute


def test_nothing_activates_means_a_plain_answer():
    execute = scripted()
    result = MoS(model="fake", execute=execute).run("What time is it?")

    assert result.activated == []
    assert result.gaps == []
    assert result.answer == "the merged answer"
    # filter, then an empty critique stage, then merge and the plain pass
    assert len(execute.seen[0]) == len(META)


def test_activation_pulls_in_that_groups_standpoints():
    label = "Vision impairment"
    execute = scripted(activate={label})
    result = MoS(model="fake", execute=execute).run("Design signage for a hotel.")

    assert result.activated == [label]
    assert set(result.consulted) == set(META[label].members)


def test_only_standpoints_reporting_a_gap_are_carried_forward():
    execute = scripted(activate={"Vision impairment"}, gap_from={"Blind"})
    result = MoS(model="fake", execute=execute).run("Design signage for a hotel.")

    assert [g.standpoint for g in result.gaps] == ["Blind"]
    assert result.gaps[0].comment == "i would need X"


def test_a_batch_is_three_round_trips_not_three_per_prompt():
    execute = scripted(activate={"Vision impairment"})
    MoS(model="fake", execute=execute).run_many([f"prompt {i}" for i in range(25)])

    # filter, critique, merge, plain — four regardless of batch size
    assert len(execute.seen) == 4
    assert len(execute.seen[0]) == 25 * len(META)


def test_a_failed_call_becomes_an_empty_answer_not_a_crash():
    def execute(prompts, max_tokens):
        return [None] * len(prompts)

    result = MoS(model="fake", execute=execute).run("anything")
    assert result.answer == ""


def test_empty_batch():
    assert MoS(model="fake", execute=scripted()).run_many([]) == []


# the bank


def test_meta_standpoints_cover_every_standpoint():
    covered = {m for meta in META.values() for m in meta.members}
    assert covered == set(STANDPOINTS)


def test_descriptions_are_identity_only():
    """a description that says what to notice is a checklist, not a standpoint."""
    assert not [
        label for label, s in STANDPOINTS.items() if "you notice" in s.description.lower()
    ]


def test_omni_is_intersectional():
    assert len(OMNI.split()) > 20


@pytest.mark.parametrize("name,expected", [
    ("Muslim", "You are Muslim."),
    ("Peanut allergy", "You have a severe peanut allergy."),
    ("", ""),
    ("no such group at all", ""),
])
def test_group_lookup(name, expected):
    assert for_group(name) == expected


def test_describe_needs_a_real_label():
    with pytest.raises(KeyError):
        describe("not a standpoint")
