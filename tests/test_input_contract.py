import pickle

import pytest
import numpy as np
import pandas as pd
import onnx
import onnxruntime as rt
from interpret.glassbox import ExplainableBoostingClassifier

import ebm2onnx


def _fit_input_regression(boolean, excluded):
    row = np.arange(128)
    a = ((row % 16) - 8).astype(np.float32)
    b = row % 3 == 0 if boolean else ((row // 16) - 4).astype(np.float32)
    x = pd.DataFrame({'a': a, 'b': b})
    y = ((a > 0) ^ (b if boolean else b > 0)).astype(np.int64)
    permutation = np.random.RandomState(1729).permutation(len(row))
    train, test = permutation[:96], permutation[96:]
    excluded_index = 1 if boolean else 0
    model = ExplainableBoostingClassifier(
        feature_names=list(x.columns),
        feature_types=['continuous', 'nominal' if boolean else 'continuous'],
        interactions=[(0, 1)],
        exclude=[excluded_index] if excluded else None,
        random_state=1729, n_jobs=1, outer_bags=1, inner_bags=0,
        max_rounds=24, max_bins=8, max_interaction_bins=8,
        validation_size=0, early_stopping_rounds=0,
        smoothing_rounds=0, interaction_smoothing_rounds=0,
    )
    model.fit(x.iloc[train], y[train])
    assert (0, 1) in model.term_features_
    assert ((excluded_index,) not in model.term_features_) == excluded
    if boolean:
        assert set(b[train]) == set(b[test]) == {False, True}
    return model, x.iloc[test].copy()


def _assert_input_regression(model, x, dtype, feed):
    expected_labels = model.predict(x)
    expected_probabilities = model.predict_proba(x)
    bins_before = pickle.dumps(model.bins_, protocol=4)
    try:
        converted = ebm2onnx.to_onnx(model, dtype=dtype, predict_proba=True)
        names = [value.name for value in converted.graph.input]
        assert len(names) == len(feed)
        assert set(names) == set(feed)
        onnx.checker.check_model(converted)
        rt.disable_telemetry_events()
        options = rt.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        session = rt.InferenceSession(
            converted.SerializeToString(), sess_options=options,
            providers=['CPUExecutionProvider'],
        )
        names = [value.name for value in session.get_inputs()]
        assert len(names) == len(feed)
        assert set(names) == set(feed)
        labels, probabilities = session.run(['prediction', 'probabilities'], feed)
        assert labels.shape == expected_labels.shape
        assert probabilities.shape == expected_probabilities.shape
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_allclose(
            probabilities, expected_probabilities, rtol=1e-5, atol=1e-6,
        )
    finally:
        assert pickle.dumps(model.bins_, protocol=4) == bins_before


@pytest.mark.parametrize('excluded', [False, True])
def test_boolean_inputs_with_excluded_main_effect(excluded):
    model, x = _fit_input_regression(boolean=True, excluded=excluded)
    _assert_input_regression(
        model, x, dtype={'a': 'float', 'b': 'bool'},
        feed={'a': x['a'].to_numpy(dtype=np.float32),
              'b': x['b'].to_numpy(dtype=np.bool_)},
    )


@pytest.mark.parametrize('excluded', [False, True])
def test_tensor_inputs_with_excluded_main_effect(excluded):
    model, x = _fit_input_regression(boolean=False, excluded=excluded)
    values = x.to_numpy(dtype=np.float32)
    assert values.shape == (32, len(model.feature_names_in_))
    assert list(x.columns) == list(model.feature_names_in_)
    _assert_input_regression(model, x, dtype=('data', 'float'), feed={'data': values})
