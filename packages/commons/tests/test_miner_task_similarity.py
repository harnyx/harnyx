from harnyx_commons.miner_task_similarity import SimilarityVoteInput, tally_similarity_votes


def test_similarity_vote_tally_passes_and_uses_lower_median_classification() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="responded", classification="novel"),
            SimilarityVoteInput(status="responded", classification="novel"),
            SimilarityVoteInput(status="responded", classification="duplicate"),
        )
    )

    assert tally.responding_validator_count == 3
    assert tally.eligible_votes == 2
    assert tally.novel_votes == 2
    assert tally.duplicate_votes == 1
    assert tally.passes is True
    assert tally.eligible_classification == "novel"


def test_similarity_vote_tally_tie_passes_and_disqualified_votes_are_excluded() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="responded", classification="novel"),
            SimilarityVoteInput(status="responded", classification="duplicate"),
            SimilarityVoteInput(status="disqualified", classification=None),
        )
    )

    assert tally.responding_validator_count == 2
    assert tally.disqualified_count == 1
    assert tally.passes is True
    assert tally.eligible_classification == "near_duplicate"


def test_similarity_vote_tally_zero_responses_has_no_pass_result() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="disqualified", classification=None),
            SimilarityVoteInput(status="disqualified", classification=None),
        )
    )

    assert tally.responding_validator_count == 0
    assert tally.disqualified_count == 2
    assert tally.passes is None


def test_similarity_vote_tally_preserves_legacy_binary_votes_without_inventing_novelty() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="responded", classification="not_duplicate"),
            SimilarityVoteInput(status="responded", classification="duplicate"),
        )
    )

    assert tally.passes is True
    assert tally.not_duplicate_votes == 1
    assert tally.eligible_classification is None
