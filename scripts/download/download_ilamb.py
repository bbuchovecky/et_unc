"""
Download formatted datasets from ILAMB into OUTPUT_ROOT (run as a script:
`python scripts/download/download_ilamb.py`; importing it downloads nothing).
"""

from __future__ import annotations

from collections import namedtuple
import subprocess
from pathlib import Path
import xarray as xr


Dataset = namedtuple("Dataset", ["url", "variable", "cf"])

LV = 2.5e6  # J/kg

ILAMB_ROOT_URL = Path("https://www.ilamb.org/ILAMB-Data/DATA")
ILAMB_DATASETS = {
    # "et": {
    #     "FLUXNET2015": Dataset(ILAMB_ROOT_URL / "hfls" / "hfls.nc", "hfls", 1/LV),  # "W m-2" -> "kg/m2/s"
    #     "FLUXCOM": Dataset(ILAMB_ROOT_URL / "hfls" / "le.nc", "le", 1/LV),          # "watt/m2" -> "kg/m2/s"
    #     "DOLCE": Dataset(ILAMB_ROOT_URL / "evspsbl" / "DOLCE.nc", "hfls", 1/LV),    # "W m-2" -> "kg/m2/s"
    #     "CLASS": Dataset(ILAMB_ROOT_URL / "hfls" / "hfls.nc", "hfls", 1/LV),        # "W m-2" -> "kg/m2/s"
    #     "WECANN": Dataset(ILAMB_ROOT_URL / "hfls" / "hfls.nc", "hfls", 1/LV),       # "W/m^2" -> "kg/m2/s"
    #     "GLEAMv3.3a": Dataset(ILAMB_ROOT_URL / "evspsbl" / "et.nc", "et", 1),       # "kg/m2/s"
    #     "MODIS": Dataset(ILAMB_ROOT_URL / "evspsbl" / "et_0.5x0.5.nc", "et", 1),    # "kg/m2/s"
    #     "MOD16A2": Dataset(ILAMB_ROOT_URL / "evspsbl" / "et.nc", "et", 1),          # "kg/m2/s"
    # },
    # "lai": {
    #     "AVHRR": Dataset(ILAMB_ROOT_URL / "lai" / "lai_0.5x0.5.nc", "lai", 1),
    #     "AVH15C1": Dataset(ILAMB_ROOT_URL / "lai" / "lai.nc", "lai", 1),
    #     "MODIS": Dataset(ILAMB_ROOT_URL / "lai" / "lai_0.5x0.5.nc", "lai", 1),
    #     "GIMMS_LAI4g": Dataset(ILAMB_ROOT_URL / "lai" / "cao2023_lai.nc", "lai", 1),
    # },
    "rns": {
        "CERESed4.2": Dataset(ILAMB_ROOT_URL / "rns" / "CERESed4.2" / "rns.nc", "rns", 1),
        "FLUXNET2015": Dataset(ILAMB_ROOT_URL / "rns" / "FLUXNET2015" / "rns.nc", "rns", 1),
        "GEWEX.SRB": Dataset(ILAMB_ROOT_URL / "rns" / "GEWEX.SRB" / "rns_0.5x0.5.nc", "rns", 1),
        "WRMC.BSRN": Dataset(ILAMB_ROOT_URL / "rns" / "WRMC.BSRN" / "rns", "rns", 1),
        "CLASS": Dataset(ILAMB_ROOT_URL / "rns" / "CLASS" / "rns", "rns", 1),
    },
    "pr": {
        "CMAPv1904": Dataset(ILAMB_ROOT_URL / "pr" / "CMAPv1904" / "pr.nc", "pr", 1),
        "FLUXNET2015": Dataset(ILAMB_ROOT_URL / "pr" / "FLUXNET2015" / "pr.nc", "pr", 1),
        "GPCCv2018": Dataset(ILAMB_ROOT_URL / "pr" / "GPCCv2018" / "pr.nc", "pr", 1),
        "GPCPv2.3": Dataset(ILAMB_ROOT_URL / "pr" / "GPCPv2.3" / "pr.nc", "pr", 1),
        "CLASS": Dataset(ILAMB_ROOT_URL / "pr" / "CLASS" / "pr.nc", "pr", 1),
    }
}

OUTPUT_ROOT = Path("/glade/campaign/univ/uwas0155/obs")


def main():
    for variable, dataset_dict in ILAMB_DATASETS.items():
        output_path = OUTPUT_ROOT / variable
        output_path.mkdir(parents=True, exist_ok=True)

        for name, (subdir, file, var) in dataset_dict.items():
            data_url = f"{ILAMB_ROOT_URL}/{subdir}/{name}/{file}"
            tmp = output_path / f"{variable}_{name}_tmp.nc"

            print(f"\n\n{name}: {data_url} -> {tmp}")
            subprocess.run(["wget", "-O", tmp, data_url ])

            ds = xr.open_dataset(tmp, decode_timedelta=False)
            ds = ds.rename({var: variable})
            print(f"units: {ds[variable].attrs.get('units', 'no units')}")


if __name__ == "__main__":
    main()


# ds_dict = {}

# for obs, tup in ILAMB_DATASETS["et"].items():
#     path = ROOT / "et" / f"{tup.url.parent.stem}_{obs}_tmp.nc"
#     print(path)
#     ds = xr.open_dataset(path)
#     ds = ds.rename({tup.variable: "et"})
#     ds["et"] = ds["et"] * tup.cf
#     ds["et"].attrs["long_name"] = "evapotranspiration"
#     ds["et"].attrs["units"] = "kg/m2/s"

#     if f"{tup.variable}_bnds" in ds.data_vars:
#         ds = ds.rename({f"{tup.variable}_bnds": "et_bnds"})
#         ds["et_bnds"].attrs["units"] = "kg/m2/s"

#     ds.to_netcdf(ROOT / "et" / f"et_{obs}.nc")
#     subprocess.run(["rm", path])
