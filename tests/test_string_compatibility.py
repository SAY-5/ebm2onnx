import pickle

import pytest
import numpy as np
import pandas as pd
import onnx
import onnxruntime as rt
from interpret.glassbox import ExplainableBoostingClassifier

import ebm2onnx


@pytest.fixture
def infer_string():
    try:
        pd.get_option('future.infer_string')
    except KeyError:
        pytest.skip('pandas does not provide future.infer_string')
    with pd.option_context('future.infer_string', True):
        yield


def test_inferred_string_dtype(infer_string):
    frame = pd.DataFrame({'category': ['Graduate', 'College', 'Unknown']})
    assert isinstance(frame['category'].dtype, pd.StringDtype)
    assert ebm2onnx.get_dtype_from_pandas(frame) == {'category': 'str'}


def test_object_string_dtype():
    frame = pd.DataFrame({'category': ['Graduate', 'College', 'Unknown']}, dtype=object)
    assert ebm2onnx.get_dtype_from_pandas(frame) == {'category': 'str'}


def _string_labels():
    a = (np.arange(128) % 16) - 8
    return np.where(a < -2, '$40K - $60K', np.where(a > 2, '$120K +', 'Unknown'))


def _assert_class_labels(y, label_type):
    row = np.arange(128)
    x = pd.DataFrame({
        'a': ((row % 16) - 8).astype(np.float32),
        'b': ((row // 16) - 4).astype(np.float32),
    })
    permutation = np.random.RandomState(1729).permutation(len(row))
    train, test = permutation[:96], permutation[96:]
    model = ExplainableBoostingClassifier(
        feature_names=list(x.columns), feature_types=['continuous', 'continuous'],
        interactions=0, random_state=1729, n_jobs=1, outer_bags=1, inner_bags=0,
        max_rounds=24, max_bins=8, max_interaction_bins=8,
        validation_size=0, early_stopping_rounds=0,
        smoothing_rounds=0, interaction_smoothing_rounds=0,
    )
    model.fit(x.iloc[train], y.iloc[train] if hasattr(y, 'iloc') else y[train])
    x = x.iloc[test]
    expected_labels = model.predict(x)
    expected_probabilities = model.predict_proba(x)
    before = pickle.dumps((model.bins_, model.classes_), protocol=4)
    try:
        converted = ebm2onnx.to_onnx(
            model, dtype={'a': 'float', 'b': 'float'}, predict_proba=True,
        )
        onnx.checker.check_model(converted)
        outputs = {value.name: value for value in converted.graph.output}
        assert outputs['prediction'].type.tensor_type.elem_type == label_type
        rt.disable_telemetry_events()
        options = rt.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        session = rt.InferenceSession(
            converted.SerializeToString(), sess_options=options,
            providers=['CPUExecutionProvider'],
        )
        feed = {column: x[column].to_numpy(dtype=np.float32) for column in x.columns}
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
        assert pickle.dumps((model.bins_, model.classes_), protocol=4) == before


@pytest.mark.parametrize('as_frame', [False, True])
def test_pandas_string_class_labels(infer_string, as_frame):
    labels = pd.Series(_string_labels(), name='income')
    assert isinstance(labels.dtype, pd.StringDtype)
    _assert_class_labels(labels.to_frame() if as_frame else labels, onnx.TensorProto.STRING)


@pytest.mark.parametrize('dtype', [np.str_, object])
def test_numpy_string_class_labels(dtype):
    _assert_class_labels(_string_labels().astype(dtype), onnx.TensorProto.STRING)


def test_integer_class_labels():
    labels = (np.arange(128) % 16 > 8).astype(np.int64)
    _assert_class_labels(labels, onnx.TensorProto.INT64)
