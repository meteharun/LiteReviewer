import csv
from pathlib import Path

csv_path = Path(r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\experiments\gemma_few_shot_results.csv")

def show_rows_one_by_one(csv_file: Path):
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            comment = row.get("comment", "")
            created = row.get("comment_created", "")

            print("\n---------------------------")
            print("COMMENT:")
            print(comment)
            print("\nCREATED AT:")
            print(created)
            print("---------------------------")

            input("\nPress ENTER for next row...")

    print("\nEnd of file reached.")

if __name__ == "__main__":
    show_rows_one_by_one(csv_path)
