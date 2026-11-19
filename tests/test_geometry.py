import numpy as np
from ssod_research.geometry import project_boxes, pairwise_iou, greedy_class_matches


def test_translation():
    boxes, valid = project_boxes([[1, 2, 3, 4]], [[1, 0, 10], [0, 1, 5], [0, 0, 1]])
    assert valid.tolist() == [True]
    np.testing.assert_allclose(boxes, [[11, 7, 13, 9]])


def test_rotation_all_corners():
    # A quarter-turn about origin changes min/max placement; all corners required.
    boxes, valid = project_boxes([[1, 2, 3, 5]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert valid[0]
    np.testing.assert_allclose(boxes, [[-5, 1, -2, 3]])


def test_empty_and_degenerate():
    assert pairwise_iou([], [[0, 0, 1, 1]]).shape == (0, 1)
    b, ok = project_boxes([[3, 3, 3, 5]], np.eye(3))
    assert not ok[0] and np.all(b == 0)


def test_iou_and_unique_matching():
    assert pairwise_iou([[0, 0, 2, 2]], [[1, 1, 3, 3]])[0, 0] == 1 / 7
    m = greedy_class_matches([[0, 0, 5, 5], [0, 0, 5, 5]], [1, 2],
                             [[0, 0, 5, 5]], [1])
    assert len(m) == 1 and m[0][0] == 0
