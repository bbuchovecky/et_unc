"""
dask_cluster.py
===============
Start and stop a dask PBS cluster on NCAR's Casper or Derecho
(https://ncar.github.io/dask-tutorial/notebooks/05-dask-hpc.html).

Example
-------
>>> import dask_cluster as dc
>>> client_cluster = dc.create_dask_cluster(account="UWAS0155", nworkers=4, ncores=4, nmem="32GB")
>>> ...
>>> dc.close_dask_cluster(client_cluster)
"""

from __future__ import annotations

from glob import glob
import os
import platform
import time

from dask.distributed import Client
from dask_jobqueue import PBSCluster


def create_dask_cluster(
    account: str,
    nworkers: int,
    ncores: int = 1,
    nmem: str = "5GB",
    walltime: str = "01:00:00",
    queue: str | None = None,
    print_dash: bool = True,
    **kwargs,
) -> tuple[Client, PBSCluster]:
    """
    Create and scale a dask cluster on either Casper or Derecho.

    Parameters
    ----------
    account : str
        Account to charge core hours for dask workers.
    nworkers : int
        Number of workers to scale up.
    ncores : int
        Requested number of cores.
    nmem : str
        Requested amount of memory, in the form "XGB".
    walltime : str
        Requested walltime, in the form "00:00:00".
    queue : str, optional
        "casper" or "derecho" to override the machine detected from the hostname.
    print_dash : bool
        Whether to print instructions to access the dask dashboard.
    **kwargs
        Arguments to pass to PBSCluster.

    Returns
    -------
    (client, cluster)
    """
    node = platform.node()
    if "crlogin" in node or queue == "casper":
        node = "casper"
        queue = "casper"
        interface = "ext"
    elif "derecho" in node or queue == "derecho":
        node = "derecho"
        queue = "develop"
        interface = "hsn0"
    else:
        raise KeyError('must be on "casper" or "derecho", other machines not implemented')

    print(f"account:  {account}")
    print(f"nworkers: {nworkers}")
    print(f"ncores:   {ncores}")
    print(f"nmemory:  {nmem}")
    print(f"walltime: {walltime}")

    cluster = PBSCluster(
        cores=ncores,
        processes=ncores,
        memory=nmem,
        queue=queue,
        interface=interface,
        resource_spec=f"select=1:ncpus={str(ncores)}:mem={nmem}",
        account=account,
        walltime=walltime,
        **kwargs,
    )
    client = Client(cluster)
    cluster.scale(nworkers)
    time.sleep(5)

    print(cluster.workers)

    # SSH tunnel to view the dask dashboard locally
    if print_dash:
        user = os.environ.get("USER")
        port = cluster.dashboard_link.split(":")[2].split("/")[0]
        address = cluster.dashboard_link.split(":")[1][2:]
        print("\nTo view the dask dashboard")
        print("Run the following command in your local terminal:")
        print(f"> ssh -N -L {port}:{address}:{port} {user}@{node}.hpc.ucar.edu")
        print("Open the following link in your local browser:")
        print(f"> http://localhost:{port}/status")

    return (client, cluster)


def close_dask_cluster(client_cluster: tuple[Client, PBSCluster], remove_std_files: bool = True) -> None:
    """Close the dask client and cluster, and remove the workers' dask-worker.* logs."""
    client, cluster = client_cluster
    client.close()
    cluster.close()
    if remove_std_files:
        for f in glob("dask-worker.*"):
            os.remove(f)
