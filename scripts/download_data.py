import urllib.request
import os

url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
output_path = "data/news.csv"

if not os.path.exists("data"):
    os.makedirs("data")

print(f"Downloading {url} to {output_path}...")
try:
    urllib.request.urlretrieve(url, output_path)
    print("Download complete!")
except Exception as e:
    print(f"Error downloading file: {e}")
