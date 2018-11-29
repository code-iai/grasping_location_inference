from grasping_position_inference.inference.probability_grid import ProbabilityGrid, _steps


def test_should_return_correct_probability_based_on_grid_cell():
    probability_grid = ProbabilityGrid()

    assert probability_grid[1, 2] == 0.5

def test_should_return_correct_steps_numbers():
    assert _steps(-0.8, -0.8) == 0
    assert _steps(-0.8, 0.8) == 160
    assert _steps(-0.8, -0.7) == 10
    assert _steps(-0.7, -0.8) == 10
    assert _steps(-0.7, 0) == 70
    assert _steps(-0.7, 0.1) == 80
    assert _steps(0.7, 0.1) == 60
    assert _steps(0.7, -0.1) == 80