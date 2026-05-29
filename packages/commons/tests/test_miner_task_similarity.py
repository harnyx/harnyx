from harnyx_commons.miner_task_similarity import SimilarityVoteInput, tally_similarity_votes


def test_similarity_vote_tally_passes_on_majority_not_duplicate_votes() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="responded", verdict="not_duplicate"),
            SimilarityVoteInput(status="responded", verdict="not_duplicate"),
            SimilarityVoteInput(status="responded", verdict="duplicate"),
        )
    )

    assert tally.responding_validator_count == 3
    assert tally.not_duplicate_votes == 2
    assert tally.duplicate_votes == 1
    assert tally.passes is True


def test_similarity_vote_tally_tie_passes_and_disqualified_votes_are_excluded() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="responded", verdict="not_duplicate"),
            SimilarityVoteInput(status="responded", verdict="duplicate"),
            SimilarityVoteInput(status="disqualified", verdict=None),
        )
    )

    assert tally.responding_validator_count == 2
    assert tally.disqualified_count == 1
    assert tally.passes is True


def test_similarity_vote_tally_zero_responses_has_no_pass_result() -> None:
    tally = tally_similarity_votes(
        (
            SimilarityVoteInput(status="disqualified", verdict=None),
            SimilarityVoteInput(status="disqualified", verdict=None),
        )
    )

    assert tally.responding_validator_count == 0
    assert tally.disqualified_count == 2
    assert tally.passes is None
