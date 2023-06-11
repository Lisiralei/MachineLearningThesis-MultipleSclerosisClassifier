import data_manipulation
from models import sclerosis_cnn




if __name__ == '__main__':
    #data_manipulation.cleanup('D:\\Machine Learning Data\\MultipleSclerosisTest')

    print(sclerosis_cnn.run_cnn(epochs=10, batch_size=16, save_frequency=5, verbosity=2))
