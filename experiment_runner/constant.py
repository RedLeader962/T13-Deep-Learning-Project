# coding=utf-8
import os

EXPERIMENT_DIR: str = 'experiment'
CHERYPICKED: str = 'cherypicked'

EXPERIMENT_RUN_DIR: str = os.path.join(EXPERIMENT_DIR, 'run')
TEST_EXPERIMENT_RUN_DIR: str = os.path.join('tests', EXPERIMENT_DIR, 'run')

TRAJECTORIES_OPTIMAL = 'trajectories_optimal'
TRAJECTORIES_SUBOPTIMAL = 'trajectories_suboptimal'
