"""Fit all quasars in a miniJPAS-like catalogue.

Usage::

    python scripts/run_catalog.py CATALOG OUTPUT.npy [--profile gaussian] [--seed 0]

    # in parallel (requires mpi4py); each rank writes OUTPUT_<rank>.npy
    mpirun -n 8 python scripts/run_catalog.py CATALOG OUTPUT.npy --mpi

The output is the structured array returned by ``quasar_lines.fit_catalog``.
"""

import argparse
from pathlib import Path

import numpy as np

import quasar_lines as ql


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog", help="whitespace-separated flux catalogue")
    parser.add_argument("output", help="output .npy file")
    parser.add_argument("--profile", default="gaussian", choices=["gaussian", "nagao06"])
    parser.add_argument("--calibration", default="chaves-montero22", choices=list(ql.CALIBRATIONS))
    parser.add_argument("--z-column", default="zspec")
    parser.add_argument("--no-contamination", action="store_true",
                        help="do not correct for [OIII]/FeII contamination (for mocks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mpi", action="store_true", help="split the catalogue among MPI ranks")
    args = parser.parse_args()

    rank, size = 0, 1
    if args.mpi:
        from mpi4py import MPI

        rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()

    cat = ql.read_minijpas_catalog(args.catalog, z_column=args.z_column)
    filters = ql.load_jpas_filters(cat.band_names)
    ind = np.array_split(np.arange(len(cat)), size)[rank]

    out = ql.fit_catalog(
        cat.flux[ind], cat.flux_err[ind], cat.z[ind], filters,
        profile=args.profile, calibration=args.calibration,
        correct_contamination=not args.no_contamination,
        rng=args.seed + rank, verbose=rank == 0,
    )

    output = Path(args.output)
    if args.mpi:
        output = output.with_name(f"{output.stem}_{rank}{output.suffix}")
    np.save(output, out)


if __name__ == "__main__":
    main()
