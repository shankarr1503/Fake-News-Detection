import sys
import urllib.request

from common.paths import DATA_DIR, RAW_DATA

URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Download to a temp file and rename, so a failed download never leaves a partial news.csv.
    tmp_path = RAW_DATA.with_suffix(".csv.part")
    print(f"Downloading {URL} to {RAW_DATA}...")
    try:
        urllib.request.urlretrieve(URL, tmp_path)
        tmp_path.replace(RAW_DATA)
    except OSError as e:  # includes URLError and HTTPError
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"Error downloading file: {e}")
    print("Download complete!")


if __name__ == "__main__":
    main()
