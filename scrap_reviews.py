import csv
import ssl

from google_play_scraper import Sort, reviews_all

ssl._create_default_https_context = ssl._create_unverified_context

# provide application name to get all reviews
# Example: "com.google.android.apps.maps" for Google Maps
APP_NAME = "com.example.app"  # Replace with actual app package name

if __name__ == "__main__":
    result = reviews_all(
        APP_NAME,
        sleep_milliseconds=0,  # defaults to 0
        sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
    )
    # open the file for writing in binary mode
    if result:
        # the filename to use for the CSV file
        filename = "data/all_reviews.csv"

        # open the file for writing in text mode
        with open(filename, "w", newline="") as f:
            # create a CSV writer object
            writer = csv.DictWriter(f, fieldnames=result[0].keys())

            # write the header row
            writer.writeheader()

            # write the data rows
            for row in result:
                writer.writerow(row)
        print(len(result))
