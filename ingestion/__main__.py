from ingestion.fetch import fetch_all
from ingestion.load import load_all


def main():
    finished = fetch_all()
    if finished:
        load_all()
    else:
        print("Fetch did not finish successfully. Not loading data.")


if __name__ == "__main__":
    main()