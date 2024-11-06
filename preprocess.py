import os
import shutil

class Preprocess:
    def rename_files(self, folder_path):
        # Сортируем файлы для переименования
        jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(folder_path, f))])
        xml_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xml') and os.path.isfile(os.path.join(folder_path, f))])

        for i, (jpg, xml) in enumerate(zip(jpg_files, xml_files), start=1):
            new_name = f"{i:05d}"
            
            old_jpg_path = os.path.join(folder_path, jpg)
            old_xml_path = os.path.join(folder_path, xml)
            new_jpg_path = os.path.join(folder_path, f"{new_name}.jpg")
            new_xml_path = os.path.join(folder_path, f"{new_name}.xml")
            
            os.rename(old_jpg_path, new_jpg_path)
            os.rename(old_xml_path, new_xml_path)

    def move_files(self, folder_path):
        # Создаем папки, если они не существуют
        os.makedirs(os.path.join(folder_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "annots"), exist_ok=True)

        # Фильтруем файлы, чтобы перемещать только из корневого каталога, не затрагивая подпапки
        jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(folder_path, f))])
        xml_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xml') and os.path.isfile(os.path.join(folder_path, f))])

        for file in jpg_files:
            shutil.move(os.path.join(folder_path, file), os.path.join(folder_path, "images", file))

        for file in xml_files:
            shutil.move(os.path.join(folder_path, file), os.path.join(folder_path, "annots", file))


# Пример использования
preprocess = Preprocess()

# Сначала переименуем файлы
preprocess.rename_files("train")

# Затем переместим файлы в папки images и annots
preprocess.move_files("train")
