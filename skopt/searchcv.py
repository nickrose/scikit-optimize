import warnings

try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized

import numpy as np
from copy import deepcopy
from scipy.stats import rankdata

from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import GridSearchCV as _SkGridSearchCV
from sklearn.model_selection import RandomizedSearchCV as _SkRandomizedSearchCV
from sklearn.utils import check_random_state

from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.metrics import check_scoring
except ImportError:
    from sklearn.metrics.scorer import check_scoring

from . import Optimizer
from .utils import point_asdict, dimensions_aslist, eval_callbacks
from .space import check_dimension
from .callbacks import check_callback


class _CVQuitEarlyMixin:
    """Shared ``cv_quit_early`` fit path for any ``BaseSearchCV`` subclass
    whose ``_run_search(evaluate_candidates)`` calls
    ``evaluate_candidates(candidate_params)`` with plain parameter dicts --
    true of :class:`BayesSearchCV`, sklearn's ``GridSearchCV`` and
    ``RandomizedSearchCV`` alike. See :class:`BayesSearchCV` for parameter
    docs (``cv_quit_early``, ``cv_quit_after_n_folds``, ``cv_quit_margin``,
    ``cv_quit_min_completed``).
    """

    def _fit_with_cv_quit_early(self, X, y=None, groups=None, **fit_params):
        """Custom fit loop used when ``cv_quit_early=True``.

        Bypasses ``BaseSearchCV.fit``'s all-folds-always evaluation (which
        always scores every candidate against every CV fold) and instead
        scores each candidate's folds sequentially, abandoning the remaining
        folds for a candidate once ``cv_quit_after_n_folds`` folds look
        clearly worse than other candidates' scores *at that same fold
        count* (a median-stopping-rule style comparison — see
        ``cv_quit_margin``). Pruning only activates once
        ``cv_quit_min_completed`` candidates have run to completion, so a
        handful of early, possibly-unrepresentative trials can't distort the
        threshold before the search has any real sense of the score spread.
        Does not support ``return_train_score`` or sample-aligned fit params
        (sample_weight etc.) — add if/when needed.
        """
        from sklearn.model_selection._validation import _fit_and_score
        from sklearn.model_selection import check_cv, StratifiedKFold, KFold
        from sklearn.base import clone, is_classifier
        from sklearn.utils.validation import indexable
        from joblib import Parallel, delayed

        estimator = self.estimator
        scorer = check_scoring(estimator, scoring=self.scoring)
        self.scorer_ = scorer

        X, y = indexable(X, y)

        # A candidate's first `cv_quit_after_n_folds` folds stand in for the
        # whole CV score when deciding whether to prune it -- that's only a
        # fair sample if fold order isn't correlated with anything (row
        # order, label order, ...). Force a shuffled splitter whenever the
        # caller left `cv` as a bare fold count (or default), so early-quit
        # comparisons aren't biased by unshuffled fold order. A `cv` that's
        # already an explicit splitter/iterable is left alone.
        cv_arg = self.cv
        if isinstance(cv_arg, (int, np.integer)) or cv_arg is None:
            n_splits_req = 5 if cv_arg is None else int(cv_arg)
            seed = getattr(self, 'random_state', None)
            if is_classifier(estimator):
                cv_arg = StratifiedKFold(n_splits=n_splits_req, shuffle=True,
                                         random_state=seed)
            else:
                cv_arg = KFold(n_splits=n_splits_req, shuffle=True,
                               random_state=seed)
            warnings.warn(
                f'cv_quit_early=True: cv={self.cv!r} replaced with a shuffled '
                f'{cv_arg.__class__.__name__}(n_splits={n_splits_req}, '
                f'shuffle=True) so the first cv_quit_after_n_folds folds used '
                f'for early-quit comparisons are a representative sample '
                f'rather than being biased by row order.')

        cv_orig = check_cv(cv_arg, y, classifier=is_classifier(estimator))
        splits = list(cv_orig.split(X, y, groups=groups))
        n_splits = len(splits)
        n_quit = min(self.cv_quit_after_n_folds, n_splits)
        can_quit_early = n_quit < n_splits
        min_completed = max(1, getattr(self, 'cv_quit_min_completed', 5))

        all_candidate_params = []
        all_rows = []
        results = {}

        def _current_quit_threshold():
            # Require a real warm-up of fully-evaluated candidates before
            # trusting any spread estimate -- with too few completions the
            # observed spread is itself noise, and pruning off that noise
            # can knock out good regions before the search has explored
            # enough to know better.
            n_completed = sum(1 for r in all_rows if not r['early_quit'])
            if n_completed < min_completed:
                return None
            # Compare against other candidates' scores at the *same* fold
            # count (partial-vs-partial), not fully-evaluated candidates'
            # full-fold mean -- comparing a 3-fold partial mean to a 5-fold
            # full mean is apples-to-oranges whenever folds differ in
            # difficulty, and systematically over-prunes. But partial scores
            # only exist for candidates that already reached the
            # cv_quit_after_n_folds mark, so early on there may not be
            # enough of them yet even once cv_quit_min_completed is
            # satisfied (e.g. cv_quit_min_completed=1). Rather than block
            # pruning indefinitely on "not enough partial history", fall
            # back to the (noisier, apples-to-oranges) full-fold-mean
            # comparison until at least 3 partial scores have accumulated,
            # then switch over to the fairer partial-vs-partial comparison.
            partial_scores = [r['partial_score_at_quit'] for r in all_rows
                              if r['partial_score_at_quit'] is not None]
            if len(partial_scores) >= 3:
                scores = partial_scores
            else:
                scores = [r['mean_test_score'] for r in all_rows
                         if not r['early_quit']]
            if len(scores) < 1:
                return None
            best = max(scores)
            spread = (max(scores) - min(scores)) if len(scores) > 1 \
                else max(abs(best), 1e-6)
            return best - self.cv_quit_margin * spread

        def _eval_one(parameters):
            split_scores = [np.nan] * n_splits
            fit_times = [np.nan] * n_splits
            early_quit = False
            partial_score_at_quit = None
            for split_idx, (train, test) in enumerate(splits):
                est = clone(estimator)
                est.set_params(**parameters)
                res = _fit_and_score(
                    est, X, y, scorer=scorer, train=train, test=test,
                    verbose=self.verbose, parameters=parameters,
                    fit_params=fit_params, score_params={},
                    return_times=True, error_score=self.error_score)
                split_scores[split_idx] = res['test_scores']
                fit_times[split_idx] = res['fit_time']

                if can_quit_early and (split_idx + 1) == n_quit:
                    partial_score_at_quit = float(np.nanmean(split_scores[:n_quit]))
                    threshold = _current_quit_threshold()
                    if threshold is not None and partial_score_at_quit < threshold:
                        early_quit = True
                        break

            return {
                'params': parameters,
                'split_scores': split_scores,
                'mean_test_score': float(np.nanmean(split_scores)),
                'std_test_score': float(np.nanstd(split_scores)),
                'mean_fit_time': float(np.nanmean(fit_times)),
                'n_folds_evaluated': (n_quit if early_quit else n_splits),
                'early_quit': early_quit,
                'partial_score_at_quit': partial_score_at_quit,
            }

        # GridSearchCV/RandomizedSearchCV hand the *entire* candidate list to
        # evaluate_candidates() in one call (unlike BayesSearchCV, which asks
        # for n_points at a time) -- dispatched to Parallel() as a single
        # batch, every worker would see `all_rows` exactly as it was before
        # the call started, so early-quit could never trigger past the first
        # batch. Chunk internally so results become visible (and pruning can
        # kick in) between chunks, while still parallelising within a chunk.
        if isinstance(self.n_jobs, int) and self.n_jobs > 0:
            chunk_size = self.n_jobs
        elif self.n_jobs in (None, 0):
            chunk_size = 1
        else:
            import os
            chunk_size = os.cpu_count() or 4

        def evaluate_candidates(candidate_params):
            nonlocal results
            candidate_params = list(candidate_params)

            for start in range(0, len(candidate_params), chunk_size):
                chunk = candidate_params[start:start + chunk_size]
                rows = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)(
                    delayed(_eval_one)(p) for p in chunk)
                all_candidate_params.extend(chunk)
                all_rows.extend(rows)

            results = {
                'params': list(all_candidate_params),
                'mean_test_score': np.array([r['mean_test_score'] for r in all_rows]),
                'std_test_score': np.array([r['std_test_score'] for r in all_rows]),
                'mean_fit_time': np.array([r['mean_fit_time'] for r in all_rows]),
                'n_folds_evaluated': np.array([r['n_folds_evaluated'] for r in all_rows]),
                'early_quit': np.array([r['early_quit'] for r in all_rows]),
            }
            for split_idx in range(n_splits):
                results[f'split{split_idx}_test_score'] = np.array(
                    [r['split_scores'][split_idx] for r in all_rows])
            return results

        self._run_search(evaluate_candidates)

        if not all_rows:
            raise ValueError('No fits were performed. Was the CV iterator '
                             'empty? Were there no candidates?')

        n_early_quit = int(np.sum(results['early_quit']))
        if n_early_quit:
            logger_msg = (f'cv_quit_early: {n_early_quit}/{len(all_rows)} '
                         f'candidates abandoned after {n_quit}/{n_splits} '
                         f'folds.')
            warnings.warn(logger_msg)

        self.cv_results_ = results
        self.n_splits_ = n_splits
        self.best_index_ = int(np.argmax(results['mean_test_score']))
        self.best_score_ = float(results['mean_test_score'][self.best_index_])
        self.best_params_ = results['params'][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(estimator)
            self.best_estimator_.set_params(**self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        return self


class BayesSearchCV(_CVQuitEarlyMixin, BaseSearchCV):
    """Bayesian optimization over hyper parameters.

    BayesSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    Parameters are presented as a list of skopt.space.Dimension objects.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    search_spaces : dict, list of dict or list of tuple containing (dict, int).
        One of these cases:
        1. dictionary, where keys are parameter names (strings)
        and values are skopt.space.Dimension instances (Real, Integer
        or Categorical) or any other valid value that defines skopt
        dimension (see skopt.Optimizer docs). Represents search space
        over parameters of the provided estimator.
        2. list of dictionaries: a list of dictionaries, where every
        dictionary fits the description given in case 1 above.
        If a list of dictionary objects is given, then the search is
        performed sequentially for every parameter space with maximum
        number of evaluations set to self.n_iter.
        3. list of (dict, int > 0): an extension of case 2 above,
        where first element of every tuple is a dictionary representing
        some search subspace, similarly as in case 2, and second element
        is a number of iterations that will be spent optimizing over
        this subspace.

    n_iter : int, default=50
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution. Consider increasing
        ``n_points`` if you want to try more parameter settings in
        parallel.

    optimizer_kwargs : dict, optional
        Dict of arguments passed to :class:`Optimizer`.  For example,
        ``{'base_estimator': 'RF'}`` would use a Random Forest surrogate
        instead of the default Gaussian Process.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel. At maximum there are
        ``n_points`` times ``cv`` jobs available during each iteration.

    n_points : int, default=1
        Number of parameter settings to sample in parallel. If this does
        not align with ``n_iter``, the last iteration will sample less
        points. See also :func:`~Optimizer.ask`

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    progress_bar : boolean, default=True
        If ``True``, a progress bar is shown during the search.

    cv_quit_early : bool, default=False
        If True, evaluate each candidate's CV folds sequentially and
        abandon the remaining folds once ``cv_quit_after_n_folds`` have
        been scored and the partial score is judged "bad" relative to
        other candidates' scores *at that same fold count* (see
        ``cv_quit_margin``). Skipped folds are recorded as NaN in
        ``cv_results_`` and the candidate's ``mean_test_score`` is the
        mean of the folds that *were* run, so early-quit candidates are
        still comparable (if pessimistic) for the optimizer's ``tell()``.
        Only takes effect once ``cv_quit_min_completed`` candidates have
        run to completion — see that parameter. Also forces a *shuffled*
        splitter (``StratifiedKFold``/``KFold`` with ``shuffle=True``)
        whenever ``cv`` is left as a bare fold count or ``None``, so the
        first ``cv_quit_after_n_folds`` folds used for pruning decisions
        are a representative sample rather than being biased by row order
        (logged via a ``warnings.warn`` the first time it happens; an
        explicit splitter/iterable passed as ``cv`` is left untouched).

    cv_quit_after_n_folds : int, default=3
        Number of folds to run before checking whether to quit early.
        Ignored if ``cv_quit_early`` is False, or if the CV splitter
        produces this many folds or fewer (nothing to quit early from).

    cv_quit_margin : float, default=0.5
        Tolerance band, as a fraction of the observed spread of other
        candidates' scores. Once at least 3 candidates have a recorded
        partial score at the same ``cv_quit_after_n_folds`` fold count, the
        comparison uses those partial scores (partial-vs-partial — comparing
        a 3-fold partial mean to a 5-fold full mean is apples-to-oranges
        whenever folds differ in difficulty, and would systematically
        over-prune). Before that many partial scores exist, falls back to
        comparing against completed candidates' full-fold ``mean_test_score``
        instead, so pruning isn't blocked from ever engaging just because
        partial-score history hasn't accumulated yet. A candidate quits
        early when its partial score falls below
        ``best_score - cv_quit_margin * score_spread`` under whichever
        basis is active (scores are "higher is better", matching sklearn
        scorer convention — e.g. ``neg_brier_score``). Larger values prune
        more aggressively.

    cv_quit_min_completed : int, default=5
        Minimum number of candidates that must have run to full completion
        (all folds) before early-quitting is allowed to trigger for any
        candidate. With too few completions, the observed score spread is
        itself noisy, and pruning based on it risks throwing away a
        genuinely good but not-yet-explored region of the search space
        before the search has any real sense of where "bad" begins.
        Ignored if ``cv_quit_early`` is False.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=False
        If ``'True'``, the ``cv_results_`` attribute will include training
        scores.

    Examples
    --------

    >>> from skopt import BayesSearchCV
    >>> # parameter ranges are specified by one of below
    >>> from skopt.space import Real, Categorical, Integer
    >>>
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     train_size=0.75,
    ...                                                     random_state=0)
    >>>
    >>> # log-uniform: understand as search over p = exp(x) by varying x
    >>> opt = BayesSearchCV(
    ...     SVC(),
    ...     {
    ...         'C': Real(1e-6, 1e+6, prior='log-uniform'),
    ...         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    ...         'degree': Integer(1,8),
    ...         'kernel': Categorical(['linear', 'poly', 'rbf']),
    ...     },
    ...     n_iter=32,
    ...     random_state=0
    ... )
    >>>
    >>> # executes bayesian optimization
    >>> _ = opt.fit(X_train, y_train)
    >>>
    >>> # model can be saved, used for predictions or scoring
    >>> print(opt.score(X_test, y_test))
    0.973...

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    optimizer_results_ : list of `OptimizeResult`
        Contains a `OptimizeResult` for each search space. The search space
        parameter are sorted by its name.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    """

    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                 n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                 n_points=1, iid='deprecated', refit=True, cv=None,
                 verbose=0, progress_bar=True,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False,
                 cv_quit_early=False, cv_quit_after_n_folds=3,
                 cv_quit_margin=0.5, cv_quit_min_completed=5):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        # Temporary fix for compatibility with sklearn 0.20 and 0.21
        # See scikit-optimize#762
        # To be consistent with sklearn 0.21+, fit_params should be deprecated
        # in the constructor and be passed in ``fit``.
        self.fit_params = fit_params

        if iid != "deprecated":
            warnings.warn("The `iid` parameter has been deprecated "
                          "and will be ignored.")
        self.iid = iid  # For sklearn repr pprint

        self.cv_quit_early = cv_quit_early
        self.cv_quit_after_n_folds = cv_quit_after_n_folds
        self.cv_quit_margin = cv_quit_margin
        self.cv_quit_min_completed = cv_quit_min_completed

        super(BayesSearchCV, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)
        self.progress_bar = progress_bar

    def _check_search_space(self, search_space):
        """Checks whether the search space argument is correct"""

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, int), got %s" % elem
                        )
                    subspace, n_iter = elem

                    if (not isinstance(n_iter, int)) or n_iter < 0:
                        raise ValueError(
                            "Number of iterations in search space should be"
                            "positive integer, got %s in tuple %s " %
                            (n_iter, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, int), got %s" % elem)

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for k, v in subspace.items():
                    check_dimension(v)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space)

    @property
    def optimizer_results_(self):
        check_is_fitted(self, '_optim_results')
        return self._optim_results

    def _make_optimizer(self, params_space):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """

        kwargs = self.optimizer_kwargs_.copy()
        kwargs['dimensions'] = dimensions_aslist(params_space)
        optimizer = Optimizer(**kwargs)
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = list(sorted(
                params_space.keys()))[i]

        return optimizer

    def _step(self, search_space, optimizer, evaluate_candidates, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel.
        """
        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_points)

        # convert parameters to python native types
        params = [[np.array(v).item() for v in p] for p in params]

        # make lists into dictionaries
        params_dict = [point_asdict(search_space, p) for p in params]

        all_results = evaluate_candidates(params_dict)
        # Feed the point and objective value back into optimizer
        # Optimizer minimizes objective, hence provide negative score
        local_results = all_results["mean_test_score"][-len(params):]
        return optimizer.tell(params, [-score for score in local_results])

    @property
    def total_iterations(self):
        """
        Count total iterations that will be taken to explore
        all subspaces with `fit` method.

        Returns
        -------
        max_iter: int, total number of iterations to explore
        """
        total_iter = 0

        for elem in self.search_spaces:

            if isinstance(elem, tuple):
                space, n_iter = elem
            else:
                n_iter = self.n_iter

            total_iter += n_iter

        return total_iter

    # TODO: Accept callbacks via the constructor?
    def fit(self, X, y=None, *, groups=None, callback=None, **fit_params):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        callback: [callable, list of callables, optional]
            If callable then `callback(res)` is called after each parameter
            combination tested. If list of callables, then each callable in
            the list is called.
        """
        self._callbacks = check_callback(callback)

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)

        if self.cv_quit_early:
            self._fit_with_cv_quit_early(X, y=y, groups=groups, **fit_params)
        else:
            super().fit(X=X, y=y, groups=groups, **fit_params)

        # BaseSearchCV never ranked train scores,
        # but apparently we used to ship this (back-compat)
        if self.return_train_score and not self.cv_quit_early:
            self.cv_results_["rank_train_score"] = \
                rankdata(-np.array(self.cv_results_["mean_train_score"]),
                         method='min').astype(int)
        return self

    def _run_search(self, evaluate_candidates):
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        callbacks = deepcopy(self._callbacks)

        random_state = check_random_state(self.random_state)
        self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        optimizers = []
        for search_space in search_spaces:
            if isinstance(search_space, tuple):
                search_space = search_space[0]
            optimizers.append(self._make_optimizer(search_space))
        self.optimizers_ = optimizers  # will save the states of the optimizers

        self._optim_results = []

        n_points = self.n_points

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        for spc_idx, (search_space, optimizer) in enumerate(zip(search_spaces, optimizers)):
            # if not provided with search subspace, n_iter is taken as
            # self.n_iter
            if isinstance(search_space, tuple):
                search_space, n_iter = search_space
            else:
                n_iter = self.n_iter

            if self.progress_bar and self.verbose == 0:
                if (tqdm is None):
                    progress = None
                    warnings.warn('progress_bar request, but `tqdm` package is not installed.')
                else:
                    progress = tqdm(total=n_iter, 
                        desc=f"{self.__class__.__name__} across {n_iter} test points...")
                    
                    if spc_idx == 0:
                        def on_step(optim_result):
                            score = -optim_result['fun']
                            med = np.median(-optim_result['func_vals'])
                            progress.set_description(f"{self.__class__.__name__} across {n_iter} "
                                                     "(remaining) test points "
                                                     f"[best score: {score:.3g}, median: {med:.3g}]")
                            # if score >= 0.99:
                            #     # progress.close()
                            #     print('Interrupting!')
                            return False
                        callbacks.append(on_step)
            else:
                progress = None

            # do the optimization for particular search space
            while n_iter > 0:
                # when n_iter < n_points points left for evaluation
                n_points_adjusted = min(n_iter, n_points)

                optim_result = self._step(
                    search_space, optimizer,
                    evaluate_candidates, n_points=n_points_adjusted
                )
                n_iter -= n_points
                if progress is not None:
                    progress.update(n_points_adjusted)

                if eval_callbacks(callbacks, optim_result):
                    break
            self._optim_results.append(optim_result)
            if progress is not None:
                progress.close()

class GridSearchCV(_CVQuitEarlyMixin, _SkGridSearchCV):
    """``sklearn.model_selection.GridSearchCV`` with the same
    ``cv_quit_early`` early-abandonment option as :class:`BayesSearchCV`.
    See :class:`BayesSearchCV` for the four added parameters' docs.
    """

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False,
                 cv_quit_early=False, cv_quit_after_n_folds=3,
                 cv_quit_margin=0.5, cv_quit_min_completed=5):
        self.cv_quit_early = cv_quit_early
        self.cv_quit_after_n_folds = cv_quit_after_n_folds
        self.cv_quit_margin = cv_quit_margin
        self.cv_quit_min_completed = cv_quit_min_completed
        super().__init__(
            estimator=estimator, param_grid=param_grid, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        if self.cv_quit_early:
            return self._fit_with_cv_quit_early(X, y=y, groups=groups, **fit_params)
        return super().fit(X, y=y, groups=groups, **fit_params)


class RandomizedSearchCV(_CVQuitEarlyMixin, _SkRandomizedSearchCV):
    """``sklearn.model_selection.RandomizedSearchCV`` with the same
    ``cv_quit_early`` early-abandonment option as :class:`BayesSearchCV`.
    See :class:`BayesSearchCV` for the four added parameters' docs.
    """

    def __init__(self, estimator, param_distributions, *, n_iter=10,
                 scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=False,
                 cv_quit_early=False, cv_quit_after_n_folds=3,
                 cv_quit_margin=0.5, cv_quit_min_completed=5):
        self.cv_quit_early = cv_quit_early
        self.cv_quit_after_n_folds = cv_quit_after_n_folds
        self.cv_quit_margin = cv_quit_margin
        self.cv_quit_min_completed = cv_quit_min_completed
        super().__init__(
            estimator=estimator, param_distributions=param_distributions,
            n_iter=n_iter, scoring=scoring, n_jobs=n_jobs, refit=refit,
            cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            random_state=random_state, error_score=error_score,
            return_train_score=return_train_score)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        if self.cv_quit_early:
            return self._fit_with_cv_quit_early(X, y=y, groups=groups, **fit_params)
        return super().fit(X, y=y, groups=groups, **fit_params)
