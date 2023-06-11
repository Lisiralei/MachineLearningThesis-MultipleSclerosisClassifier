
import os
from datetime import datetime

import pandas as pd
import dicom2jpg


def data_to_png(source_directory, target_directory):
    source_path = source_directory if source_directory else 'D:\\Machine Learning Data\\Multiple Sclerosis\\'
    export_path = target_directory if target_directory else 'D:\\Machine Learning Data\\Multiple Sclerosis pngs\\'
    machines = os.listdir(source_path)
    for machine in machines:
        machine_source = source_path + machine
        for patient in os.listdir(machine_source):
            patient_source = machine_source + '\\' + patient
            print('patient: ', patient)
            if not os.path.exists(export_path + patient):
                os.mkdir(export_path + patient)

            for patient_item in os.listdir(patient_source):
                patient_item_source = patient_source + '\\' + patient_item
                print(patient_item)
                print('\tfiles in directory: ', os.listdir(patient_item_source))
                dicom2jpg.dicom2png(patient_item_source, target_root=export_path + '\\' + patient, anonymous=True)


# Usage:
# data_manipulation.data_to_png('D:\\Machine Learning Data\\Multiple Sclerosis\\', 'D:\\Machine Learning Data\\Multiple Sclerosis pngs\\')


def denest(source_directory, start=None):
    if not start:
        start = source_directory
    first_root = os.listdir(source_directory)

    print('in root: ', first_root)
    if not first_root:
        return
    for item in first_root:
        item_path = os.path.join(source_directory, item)
        if os.path.isfile(item_path):
            print('\t\titem:', item, "item's path:", item_path)
            print('\t\t\tsource: ', start)

            destination = os.path.join(start, item)
            print('\t\t\t\titem will have a path:', destination)
            os.rename(item_path, destination)
        else:
            print('\tfolder:', item)
            denest(item_path, start)


def denest_all(source_path, to_source=False):
    target_folder = source_path if to_source else None
    folders = os.listdir(source_path)
    for folder in folders:
        folder_path = os.path.join(source_path, folder)
        denest(folder_path, target_folder)

# Usage:
# data_manipulation.denest_all('D:\\Machine Learning Data\\Multiple Sclerosis pngs')
# data_manipulation.denest_all('D:\\Machine Learning Data\\MultipleSclerosisTest', to_source=True)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def cleanup(path_source):
    first_root = os.listdir(path_source)
    print('in root:', first_root)
    for folder in first_root:
        folder_path = os.path.join(path_source, folder)
        print('current folder: ', folder_path)
        remove_empty_folders(folder_path)

# Usage:
# data_manipulation.cleanup('D:\\Machine Learning Data\\Multiple Sclerosis pngs')


def set_flag(directory, target_directory, count_seed=0):
    images = os.listdir(directory)
    count = count_seed
    for image in images:
        image_path = os.path.join(directory, image)

        unique_id = str(count)

        new_image_name = os.path.join(target_directory, unique_id + '.png')
        print("Image:", image_path, "\twill be renamed to:", new_image_name)
        os.rename(image_path, new_image_name)
        count += 1

    return count
# Usage:
# last_count = data_manipulation.set_flag('D:\\Machine Learning Data\\MultipleSclerosisTest\\yes', 0,  'yes')


def set_flag_for_all(directory, flag_source, classes=('no', 'yes')):
    for class_name in classes:
        if not os.path.exists(os.path.join(directory, class_name)):
            os.mkdir(os.path.join(directory, class_name))
    clients = os.listdir(directory)
    class_count = dict([(class_name, 0) for class_name in classes])
    for client_folder in clients:
        if client_folder in flag_source:
            print(f'Folder: {client_folder}, label: {flag_source[client_folder]}')
            client_folder_path = os.path.join(directory, client_folder)
            class_folder = 'no' if flag_source[client_folder] == 0 else 'yes'
            print(flag_source[client_folder] == 0)
            class_folder_path = os.path.join(directory, class_folder)
            print(f'Folder will be unpacked into class {class_folder} at {class_folder_path}')
            current_count = set_flag(client_folder_path, class_folder_path, class_count[class_folder])
            class_count[class_folder] = current_count


def excel_to_label(excel_file_path):
    labels = dict()
    label_excel = pd.read_excel(excel_file_path)
    #print(label_excel)
    for index, row in label_excel.iterrows():
        #print(f'Name: {row[0]}, label: {row[1]}')
        labels[str(row[0])] = int(row[1])


    return labels

