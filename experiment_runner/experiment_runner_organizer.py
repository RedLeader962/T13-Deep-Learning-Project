# coding=utf-8
import os
import uuid
from datetime import datetime

from experiment_runner.constant import EXPERIMENT_DIR, EXPERIMENT_RUN_DIR
from experiment_runner.experiment_spec import ExperimentSpec
from experiment_runner.test_related_utils import is_automated_test
#
#
# def format_unique_dir_name(cleaned_exp_tag: str, type: str = 'run') -> str:
#     date_now = datetime.now()
#     experiment_uuid = uuid.uuid1().int.__str__()
#     unique_dir_name = "{}-{}-{}h{}--{}-{}-{}--{}".format(type, cleaned_exp_tag,
#                                                          date_now.hour, date_now.minute,
#                                                          date_now.day, date_now.month, date_now.year,
#                                                          experiment_uuid)
#     return unique_dir_name
#
#
# def clean_tag(tag: str):
#     cleaned_tag = ''
#     if tag:
#         if len(tag) != len(tag.replace(' ', '')):
#             for x in tag.split(' '):
#                 cleaned_tag += x.capitalize()
#         else:
#             cleaned_tag = tag
#     return cleaned_tag
#
#
# def get_spec_run_dir(spec: ExperimentSpec) -> str:
#     if spec.spec.experiment_path is None:
#         cleaned_exp_tag = clean_tag(spec.spec.experiment_tag)
#
#         if spec.spec.is_batch_spec:
#             exp_dir = "run-{}-{}".format(cleaned_exp_tag, spec.spec.spec_idx)
#         else:
#             exp_dir = format_unique_dir_name(cleaned_exp_tag, type='run')
#
#         spec.spec.experiment_path = exp_dir
#
#     return spec.spec.experiment_path
#
#
# def get_batch_run_dir(spec: ExperimentSpec) -> str:
#     if (spec.spec.batch_dir is None) and spec.spec.is_batch_spec:
#         batch_tag = clean_tag(spec.spec.batch_tag)
#
#         batch_dir = format_unique_dir_name(batch_tag, type='batch')
#
#         spec.spec.batch_dir = batch_dir
#
#     return spec.spec.batch_dir
#
#
# def reset_and_get_spec_run_path(spec: ExperimentSpec) -> os.PathLike:
#     spec.spec.experiment_path = None
#     return spec.get_spec_run_path()
#
#
# def get_spec_run_path(spec: ExperimentSpec) -> os.PathLike:
#     if spec.spec.experiment_path is None:
#         root_path = os.path.relpath(EXPERIMENT_RUN_DIR)
#
#         if spec.spec.batch_dir:
#             root_path = os.path.join(root_path, spec.spec.batch_dir)
#
#         spec_run_path = os.path.join(root_path, spec.get_spec_run_dir(spec.spec))
#         spec.spec.experiment_path = spec_run_path
#
#     return spec.spec.experiment_path
#
#
# def setup_run_dir(spec: ExperimentSpec) -> None:
#     exp_path = spec.get_spec_run_path(spec.spec)
#
#     if not os.path.exists(exp_path):
#         os.makedirs(exp_path)
#
#     return None
