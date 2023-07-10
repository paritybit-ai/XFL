# Copyright 2022 The XFL Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import sys

import client
import scheduler_run
import trainer_run
from common.utils.logger import logger


def add_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--scheduler", action="store_true",
                       help="run scheduler server")
    group.add_argument("-a", "--assist_trainer", action="store_true",
                       help='run assist trainer server')
    group.add_argument("-t", "--trainer", type=str, metavar="1",
                       default="trainer", const="trainer", nargs="?",
                       help='run trainer server')
    group.add_argument("-c", "--client", type=str, metavar="start",
                       choices=["start", "stop", "status", "algo", "stage"],
                       help="run command line client")
    parser.add_argument("--bar", action="store_true",
                        help="display a progress bar on scheduler")
    parser.add_argument("--config_path", type=str,
                        default="/opt/config", metavar="/opt/config", nargs="?",
                        help="config file path")
    return parser


def main():
    parser = add_args(argparse.ArgumentParser(description="XFL - BaseBit Federated Learning"))
    args = parser.parse_args()

    if args.scheduler:
        scheduler_run.main(args.config_path, args.bar)
    elif args.client:
        # client.main(args.client)
        client.main(args.client, args.config_path)
    elif args.assist_trainer:
        # trainer_run.main("assist_trainer", "assist_trainer")
        trainer_run.main("assist_trainer", "assist_trainer", config_path=args.config_path)
    elif args.trainer:
        # trainer_run.main("trainer", args.trainer)
        trainer_run.main("trainer", args.trainer, config_path=args.config_path)


def check_version():
    version = "3.9"
    current_version = '.'.join([str(sys.version_info.major), str(sys.version_info.minor)])
    if version != current_version:
        logger.error("Python Version is not: " + version)
        sys.exit(-1)


if __name__ == "__main__":
    # check_version()
    main()
