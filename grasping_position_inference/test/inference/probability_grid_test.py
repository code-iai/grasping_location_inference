from grasping_position_inference.inference.probability_grid import ProbabilityGrid


def test_should_return_correct_probability_based_on_grid_cell():
    probability_grid = ProbabilityGrid()

    assert probability_grid[1, 2] == 0.5