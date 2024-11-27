# Getting Started

## Important Note
The script `soil_public/scripts/get_s2/get_sentinel2_timeseries.py` will not be able to query the Google Earth Engine (GEE) servers without a service account json key (declare once and forget) and an Earth Engine Cloud enabled project string. You can obtain both of these following the guide at <https://developers.google.com/earth-engine/guides/service_account>

You can set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable as the path to the service account key like this:
```{bash}
EXPORT GOOGLE_APPLICATION_CREDENTIALS="path_to_service_account_key.json"
```

## Running the Script

Run the script as follows, assuming you cloned the repo into home (Linux specific terminology).
```{bash}
cd "$HOME/soil_public/scripts/get_s2" && ./main.sh "/path/to/logs_dir" "ee-project-name" "/path/to/AOI" "YYYY-MM-DD" "YYYY-MM-DD" "/path/to/out_dir"
```

The `main.sh` file prepares the directories, invokes a conda environment and passes the arguments from main to `python`.