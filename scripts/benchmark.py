#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import czifile
import numpy as np
import psutil
from quilt3 import Package
from tqdm import tqdm

import aicspylibczi as aics

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

CLUSTER_CONFIGS = [
    {"name": "small-local-cluster-replica", "per_worker_cores": 2, "workers": 4},
    {"name": "large-local-cluster-replica", "per_worker_cores": 2, "workers": 16},
    {"name": "small-worker-distributed-cluster", "per_worker_cores": 1, "workers": 32},
    {
        "name": "many-small-worker-distributed-cluster",
        "per_worker_cores": 1,
        "workers": 128,
    },
    {
        "name": "many-standard-worker-distributed-cluster",
        "per_worker_cores": 4,
        "workers": 32,
    },
]

###############################################################################

# Args


class Args(argparse.Namespace):
    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="benchmark_aics",
            description=(
                "Run read time benchmarks for aicspylibczi against other common image "
                "readers. The benchmark dataset can be downloaded using the "
                "download_test_resources script with the specific hash: "
                "5e665ed66c1b373a84002227044c7a12a2ecc506b84a730442a5ed798428e26a"
            ),
        )

        # Arguments
        p.add_argument(
            "--save_path",
            default="benchmark/results.json",
            type=Path,
            help="Path to save the generated benchmark JSON file.",
        )
        p.add_argument(
            "--upload",
            action="store_true",
            help="Should the results be uploaded to Quilt.",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Show traceback if the script were to fail.",
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################


def _run_benchmark(
        resources_dir: Path,
        extensions: List[str],
        non_aics_reader: List[Callable],
        iterations: int = 3,
):
    # Collect files matching the extensions provided
    files = []
    for ext in extensions:
        files += list(resources_dir.glob(ext))
        files += list(Path("/Users/jamies/Data").glob(ext))

    # Run reads for each file and store details in results
    aics_czi_reader = lambda file: ( aics.CziFile(file).read_image(S=0) )
    results = []
    for file in files:
        print(file)
        yx_planes = np.prod(aics.CziFile(file).size[:-2]) #info_read.size("STCZ"))
        for reader in [aics_czi_reader, non_aics_reader]:
            reader_path = f"{reader.__module__}.{reader.__name__}"
            for i in tqdm(range(iterations), desc=f"{reader_path}: {file.name}"):
                reader(str(file))  # warmup run
                start = time.perf_counter()
                reader(str(file))
                results.append(
                    {
                        "file_name": file.name,
                        "file_size_gb": file.stat().st_size / 10e8,
                        "reader": (
                            "other" if "czifile" in reader_path else "aics"
                        ),
                        "yx_planes": int(yx_planes),
                        "read_duration": time.perf_counter() - start,
                    }
                )

    return results


def _run_benchmark_suite(resources_dir: Path):
    # Default reader / imageio imread tests

    # CZI reader / czifile imread tests
    czi_reader_results = _run_benchmark(
        resources_dir=resources_dir,
        extensions=["*.czi"],
        non_aics_reader=czifile.imread,
    )

    return [
        *czi_reader_results,
    ]


def run_benchmarks(args: Args):
    # Results are stored as they are returned
    all_results = {}

    # Try running the benchmarks
    #try:
    # Get benchmark resources dir
    resources_dir = Path().resolve().parent / "aicspylibczi" / "tests" / "resources"
    print(resources_dir.resolve())
    # Store machine config
    _ = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "cpu_total_count": psutil.cpu_count(),
        "cpu_current_utilization": psutil.cpu_percent(),
        "memory_total_gb": psutil.virtual_memory().total / 10e8,
        "memory_available_gb": psutil.virtual_memory().available / 10e8,
    }

    # Store python config
    pyversion = sys.version_info
    _ = {
        "python_version": f"{pyversion.major}.{pyversion.minor}.{pyversion.micro}",
        "aicspylibczi": aics.__version__,
        "czifile": czifile.__version__,
    }

    # Run tests
    #######################################################################

    log.info(f"Running tests: no cluster...")
    log.info(f"=" * 80)

    all_results["no-cluster"] = _run_benchmark_suite(resources_dir=resources_dir)

    #######################################################################

        # Create or get log dir
        # Do not include ms
    log_dir_name = datetime.now().isoformat().split(".")[0]
    log_dir = Path(f".dask_logs/{log_dir_name}").expanduser()
    # Log dir settings
    log_dir.mkdir(parents=True, exist_ok=True)

    #######################################################################

    log.info(f"Completed all tests")
    log.info(f"=" * 80)

    # Ensure save dir exists and save results
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w") as write_out:
        json.dump(all_results, write_out)

    # # Construct and push package
    # if args.upload:
    #     p = Package()
    #     p.set("results.json", args.save_path)
    #     p.push(
    #         "aics/benchmarks",
    #         "s3://aics-modeling-packages-test-resources",
    #         message=f"aics version: {aics.__version__}",
    #     )

    # Catch any exception
# except Exception as e:
#     log.error("=============================================")
#     if args.debug:
#         log.error("\n\n" + traceback.format_exc())
#         log.error("=============================================")
#     log.error("\n\n" + str(e) + "\n")
#     log.error("=============================================")
#     sys.exit(1)


###############################################################################
# Runner


def main():
    args = Args()
    run_benchmarks(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
