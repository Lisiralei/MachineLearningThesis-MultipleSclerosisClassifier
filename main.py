
import data_manipulation
from models import sclerosis_cnn


def dataset_manipulation_snippet():
    excel_file = 'D:\\Machine Learning Data\\MultipleSclerosisTest\\labels.xlsx'
    dataset_path_for_label = 'D:\\Machine Learning Data\\Multiple Sclerosis pngs'

    folder_labels = data_manipulation.excel_to_label(excel_file)
    print('folder labels:', folder_labels)
    data_manipulation.set_flag_for_all(dataset_path_for_label, flag_source=folder_labels, classes=('no', 'yes'))
    data_manipulation.remove_empty_folders(dataset_path_for_label)


if __name__ == '__main__':

    #dataset_manipulation_snippet()

    dataset_path = 'D:\\Machine Learning Data\\MultipleSclerosisTest'
    print(sclerosis_cnn.run_cnn(dataset_path, epochs=5, batch_size=18, save_frequency=5, verbosity=2))

