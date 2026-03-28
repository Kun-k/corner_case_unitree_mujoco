import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import argparse

from deploy_mujoco.terrain_g1.g1_controller import G1Controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs="?", default="g1.yaml", help="Config filename under terrain_g1/configs")
    args = parser.parse_args()

    controller = G1Controller(args.config_file)
    controller.run()


if __name__ == "__main__":
    main()

