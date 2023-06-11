import data_manipulation
from models import sclerosis_cnn




if __name__ == '__main__':
    #data_manipulation.cleanup('D:\\Machine Learning Data\\MultipleSclerosisTest')

    dataset_path = 'D:\\Machine Learning Data\\MultipleSclerosisTest'
    print(sclerosis_cnn.run_cnn(dataset_path, epochs=5, batch_size=18, save_frequency=5, verbosity=2))
