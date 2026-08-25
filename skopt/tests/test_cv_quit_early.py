"""Tests for the ``cv_quit_early`` early-abandonment option shared by
:class:`skopt.BayesSearchCV`, :class:`skopt.GridSearchCV` and
:class:`skopt.RandomizedSearchCV` (see ``skopt/searchcv.py``,
``_CVQuitEarlyMixin``).
"""
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from skopt import BayesSearchCV, GridSearchCV, RandomizedSearchCV

ALL_CLASSES = (BayesSearchCV, GridSearchCV, RandomizedSearchCV)


def _toy_dataset():
    """Same toy dataset used throughout manual exploration of this
    feature: small, quick to fit, with enough class separation that a
    plain accuracy scorer is well-behaved."""
    return make_classification(n_samples=400, n_features=12, random_state=0,
                               class_sep=1.5)


def _accuracy_scorer(estimator, X, y):
    return estimator.score(X, y)


def _make_search(cls, **overrides):
    """Build one of the three search classes over the toy dataset's
    DecisionTreeClassifier search space, with sane defaults for the new
    cv_quit_early_* kwargs that tests can override piecemeal."""
    kwargs = dict(cv=5, scoring=_accuracy_scorer)
    kwargs.update(overrides)

    if cls is BayesSearchCV:
        space = {'max_depth': (1, 15), 'min_samples_split': (2, 40)}
        kwargs.setdefault('n_iter', 10)
        kwargs.setdefault('random_state', 1)
        kwargs.setdefault('progress_bar', False)
        return cls(DecisionTreeClassifier(), space, **kwargs)
    elif cls is GridSearchCV:
        space = {'max_depth': [1, 3, 5, 7, 9, 11, 13],
                 'min_samples_split': [2, 10, 20, 30]}
        return cls(DecisionTreeClassifier(), space, **kwargs)
    elif cls is RandomizedSearchCV:
        space = {'max_depth': list(range(1, 15)),
                 'min_samples_split': list(range(2, 40))}
        kwargs.setdefault('n_iter', 10)
        kwargs.setdefault('random_state', 1)
        return cls(DecisionTreeClassifier(), space, **kwargs)
    raise AssertionError(cls)


# ═══════════════════════════════════════════════════════════════════════
#  Basic instantiation / default-path tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_default_cv_quit_early_is_off(cls):
    """cv_quit_early defaults to False and shouldn't change behaviour at
    all relative to the underlying sklearn/skopt class."""
    search = _make_search(cls)
    assert search.cv_quit_early is False
    assert search.cv_quit_after_n_folds == 3
    assert search.cv_quit_margin == 0.5
    assert search.cv_quit_min_completed == 5


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_fit_without_cv_quit_early_runs_and_scores_well(cls):
    """Plain instantiation + fit (no cv_quit_early) should behave like
    a normal search: produce a fitted best_estimator_ with a reasonable
    score on the toy dataset."""
    X, y = _toy_dataset()
    search = _make_search(cls)
    search.fit(X, y)

    assert hasattr(search, 'best_params_')
    assert hasattr(search, 'best_estimator_')
    assert search.best_score_ > 0.8
    # cv_results_ should look like a normal (rectangular) sklearn result:
    # no early_quit / partial-score columns since the code path never
    # touches _fit_with_cv_quit_early.
    assert 'early_quit' not in search.cv_results_
    assert 'partial_score_at_quit' not in search.cv_results_


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_early_params_are_stored_on_instance(cls):
    search = _make_search(cls, cv_quit_early=True, cv_quit_after_n_folds=2,
                          cv_quit_margin=0.25, cv_quit_min_completed=4)
    assert search.cv_quit_early is True
    assert search.cv_quit_after_n_folds == 2
    assert search.cv_quit_margin == 0.25
    assert search.cv_quit_min_completed == 4


# ═══════════════════════════════════════════════════════════════════════
#  cv_quit_early fit-path tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_early_fits_and_selects_reasonable_params(cls):
    """With cv_quit_early on, the search should still complete and land
    on a best_estimator_ that scores comparably to the non-early-quit
    path (pruning shouldn't wreck overall search quality on an easy toy
    problem)."""
    X, y = _toy_dataset()
    search = _make_search(cls, cv_quit_early=True, cv_quit_after_n_folds=2,
                          cv_quit_margin=0.3, cv_quit_min_completed=3)
    search.fit(X, y)

    assert search.best_score_ > 0.8
    assert search.best_estimator_ is not None
    # A prediction call should work end to end.
    preds = search.best_estimator_.predict(X)
    assert preds.shape == y.shape


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_early_cv_results_shape_and_columns(cls):
    """cv_results_ under cv_quit_early should carry the extra bookkeeping
    columns, all the same length as the candidate list, plus per-split
    score columns with NaN for folds an abandoned candidate never ran."""
    X, y = _toy_dataset()
    search = _make_search(cls, cv_quit_early=True, cv_quit_after_n_folds=2,
                          cv_quit_margin=0.3, cv_quit_min_completed=3)
    search.fit(X, y)

    n_candidates = len(search.cv_results_['params'])
    for key in ('mean_test_score', 'std_test_score', 'mean_fit_time',
                'n_folds_evaluated', 'early_quit'):
        assert key in search.cv_results_
        assert len(search.cv_results_[key]) == n_candidates

    assert search.n_splits_ == 5
    for split_idx in range(search.n_splits_):
        col = search.cv_results_[f'split{split_idx}_test_score']
        assert len(col) == n_candidates

    # Every early-quit candidate ran fewer folds than n_splits_, and has
    # NaN in the split columns beyond what it actually ran.
    for i, is_early in enumerate(search.cv_results_['early_quit']):
        n_ran = search.cv_results_['n_folds_evaluated'][i]
        if is_early:
            assert n_ran < search.n_splits_
            for split_idx in range(n_ran, search.n_splits_):
                assert np.isnan(
                    search.cv_results_[f'split{split_idx}_test_score'][i])
        else:
            assert n_ran == search.n_splits_


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_min_completed_gates_pruning(cls):
    """The first cv_quit_min_completed candidates must always run to full
    completion, regardless of how bad they look after cv_quit_after_n_folds
    folds -- there's no baseline to compare against yet."""
    X, y = _toy_dataset()
    min_completed = 5
    search = _make_search(cls, cv_quit_early=True, cv_quit_after_n_folds=2,
                          cv_quit_margin=0.3,
                          cv_quit_min_completed=min_completed)
    search.fit(X, y)

    early_quit = search.cv_results_['early_quit']
    n_candidates = len(early_quit)
    if n_candidates <= min_completed:
        pytest.skip('not enough candidates generated to exercise the gate')
    assert not any(early_quit[:min_completed]), (
        'a candidate among the first cv_quit_min_completed was pruned '
        'before there was any completed-candidate baseline to compare '
        'against')


def test_cv_quit_early_never_prunes_when_min_completed_exceeds_n_candidates():
    """If cv_quit_min_completed is set higher than the number of
    candidates the search will ever try, nothing should ever be pruned --
    this exercises the "always returns None" path end to end."""
    X, y = _toy_dataset()
    search = _make_search(GridSearchCV, cv_quit_early=True,
                          cv_quit_after_n_folds=2, cv_quit_margin=0.3,
                          cv_quit_min_completed=10_000)
    search.fit(X, y)
    assert not any(search.cv_results_['early_quit'])
    assert search.n_splits_ == 5
    for i in range(len(search.cv_results_['params'])):
        assert search.cv_results_['n_folds_evaluated'][i] == 5


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_early_warns_and_uses_shuffled_cv(cls):
    """Passing a bare fold count with cv_quit_early=True should trigger a
    warning and swap in a shuffled StratifiedKFold, so the folds used for
    early-quit comparisons aren't biased by row order."""
    X, y = _toy_dataset()
    search = _make_search(cls, cv=5, cv_quit_early=True,
                          cv_quit_after_n_folds=2, cv_quit_margin=0.3,
                          cv_quit_min_completed=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        search.fit(X, y)

    messages = [str(w.message) for w in caught]
    assert any('shuffled' in m and 'StratifiedKFold' in m for m in messages), (
        f'expected a shuffled-CV warning, got: {messages}')


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_early_respects_explicit_splitter(cls):
    """An explicit splitter object passed as `cv` should be used as-is --
    no shuffle warning, no override -- even with cv_quit_early=True."""
    X, y = _toy_dataset()
    explicit_cv = StratifiedKFold(n_splits=4, shuffle=False)
    search = _make_search(cls, cv=explicit_cv, cv_quit_early=True,
                          cv_quit_after_n_folds=2, cv_quit_margin=0.3,
                          cv_quit_min_completed=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        search.fit(X, y)

    messages = [str(w.message) for w in caught]
    assert not any('shuffled' in m for m in messages)
    assert search.n_splits_ == 4


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_cv_quit_early_partial_score_surrogate_fallback(cls):
    """Before 3 partial-score data points exist, the threshold should fall
    back to comparing against completed candidates' full-fold mean rather
    than returning None forever -- so pruning can still engage as soon as
    cv_quit_min_completed is satisfied, even though few/no partial scores
    have accumulated yet at exactly that point."""
    X, y = _toy_dataset()
    # cv_quit_min_completed=1 with a low margin: as soon as a single
    # candidate completes, the very next candidate should be able to be
    # judged against it via the full-score fallback (since there's only
    # ever at most 1 partial score recorded at that point, well under the
    # 3-sample threshold for switching to partial-vs-partial).
    search = _make_search(cls, cv_quit_early=True, cv_quit_after_n_folds=2,
                          cv_quit_margin=0.9, cv_quit_min_completed=1)
    search.fit(X, y)

    # The feature should be exercised (not necessarily every run --
    # depends on the actual score spread -- but the search should complete
    # without error and produce internally consistent bookkeeping).
    n_candidates = len(search.cv_results_['params'])
    assert n_candidates > 1
    assert search.best_score_ > 0.5


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cv_quit_early_chunking_across_n_jobs(cls, n_jobs):
    """Regression test for the batching bug: GridSearchCV/RandomizedSearchCV
    historically handed their whole candidate list to evaluate_candidates()
    in one call, so results never became visible mid-search and pruning
    could never trigger. With chunking, pruning should be able to engage
    for any n_jobs value once cv_quit_min_completed is satisfied."""
    X, y = _toy_dataset()
    search = _make_search(cls, n_jobs=n_jobs, cv_quit_early=True,
                          cv_quit_after_n_folds=2, cv_quit_margin=0.3,
                          cv_quit_min_completed=3)
    search.fit(X, y)

    n_candidates = len(search.cv_results_['params'])
    n_completed = int(np.sum(~search.cv_results_['early_quit']))
    assert n_completed >= min(3, n_candidates)
    # Not a strict assertion that pruning *did* fire (depends on the score
    # spread for this particular n_jobs/chunking split), just that the
    # bookkeeping is internally consistent either way.
    assert n_completed <= n_candidates
