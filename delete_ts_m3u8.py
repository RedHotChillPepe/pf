import os

def delete_files_with_extensions(folder_path, extensions):
    if not os.path.isdir(folder_path):
        print(f"Папка {folder_path} не существует.")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Удален файл: {file_path}")
                    except Exception as e:
                        print(f"Ошибка при удалении файла {file_path}: {e}")

if __name__ == "__main__":
    # Укажите путь к папке, где нужно искать файлы
    folder_path = './stream'

    # Укажите расширения файлов, которые нужно удалить
    extensions_to_delete = ['.ts', '.m3u8']

    # Вызов функции для удаления файлов
    delete_files_with_extensions(folder_path, extensions_to_delete)
    print("Кэш-файлы видеопотока успешно удалены")