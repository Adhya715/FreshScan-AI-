from pathlib import Path
dataset_path = Path("Fruit Freshness Dataset/data/raw")
for fruit_folder in dataset_path.iterdir():
    if fruit_folder.is_dir():

        print(f"\n{fruit_folder.name}")

        for freshness_folder in fruit_folder.iterdir(): #Proceed only if this item is a directory (folder).

            if freshness_folder.is_dir():
                image_count = len(list(freshness_folder.glob("*"))) #Give me everything inside this folder, len() can count

                print(f"   {freshness_folder.name}: {image_count} images")
