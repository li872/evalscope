"""pass@k and pass^k count an attempt as correct only when its score reaches 1.0.

Summing raw values would turn several partially correct attempts into "correct" ones,
so four attempts scored 0.5 used to report pass@4 == 1.0.
"""
from typing import Any, List, Optional

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators import MeanPassAtK, MeanPassHatK


def make_group(values: List[float], group_id: str = 'g0') -> List[SampleScore]:
    return [
        SampleScore(
            score=Score(value={'acc': value}),
            sample_id=f'{group_id}-{index}',
            group_id=group_id,
            generation_index=index,
        ) for index, value in enumerate(values)
    ]


def scores_by_k(agg_scores: List[Any], aggregation: str) -> dict:
    return {agg.dimensions['k']: agg.score for agg in agg_scores if agg.aggregation == aggregation}


def find_mean(agg_scores: List[Any]) -> Optional[float]:
    return next(agg.score for agg in agg_scores if agg.aggregation == 'mean')


@pytest.mark.parametrize('values', [[0.5, 0.5, 0.5, 0.5], [0.9, 0.9, 0.9]])
def test_pass_at_k_ignores_partial_scores(values):
    result = MeanPassAtK()(make_group(values))
    assert find_mean(result) == pytest.approx(sum(values) / len(values))
    assert set(scores_by_k(result, 'pass_at_k').values()) == {0.0}


@pytest.mark.parametrize('values', [[0.5, 0.5, 0.5, 0.5], [0.9, 0.9, 0.9]])
def test_pass_hat_k_ignores_partial_scores(values):
    result = MeanPassHatK()(make_group(values))
    assert set(scores_by_k(result, 'pass_hat_k').values()) == {0.0}


def test_partial_scores_do_not_add_up_to_a_correct_attempt():
    values = [1.0, 0.0, 0.6, 0.6]
    pass_at = scores_by_k(MeanPassAtK()(make_group(values)), 'pass_at_k')
    pass_hat = scores_by_k(MeanPassHatK()(make_group(values)), 'pass_hat_k')
    # One correct attempt out of four.
    assert pass_at == pytest.approx({1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0})
    assert pass_hat == pytest.approx({1: 0.25, 2: 0.0, 3: 0.0, 4: 0.0})


def test_scores_above_one_count_once():
    pass_hat = scores_by_k(MeanPassHatK()(make_group([2.0, 0.0])), 'pass_hat_k')
    assert pass_hat == pytest.approx({1: 0.5, 2: 0.0})


@pytest.mark.parametrize(
    'values, expected_pass_at, expected_pass_hat',
    [
        ([1.0, 0.0, 1.0, 0.0], {1: 0.5, 2: 5 / 6, 3: 1.0, 4: 1.0}, {1: 0.5, 2: 1 / 6, 3: 0.0, 4: 0.0}),
        ([1, 1, 1], {1: 1.0, 2: 1.0, 3: 1.0}, {1: 1.0, 2: 1.0, 3: 1.0}),
        ([0, 0], {1: 0.0, 2: 0.0}, {1: 0.0, 2: 0.0}),
        ([True, False], {1: 0.5, 2: 1.0}, {1: 0.5, 2: 0.0}),
    ],
)
def test_binary_scores_unchanged(values, expected_pass_at, expected_pass_hat):
    assert scores_by_k(MeanPassAtK()(make_group(values)), 'pass_at_k') == pytest.approx(expected_pass_at)
    assert scores_by_k(MeanPassHatK()(make_group(values)), 'pass_hat_k') == pytest.approx(expected_pass_hat)
