import pandas as pd


def read_file(data_file):
    # Read a single trajectory file from taxi, walk, etc. folder
    points = pd.read_csv(data_file, skiprows=0, header=None)

    return points


if __name__ == '__main__':
    df1 = read_file('motor_data.csv')
    df2 = read_file('train_data.csv')
    df3 = read_file('walk_data.csv')
    df4 = read_file('bike_data.csv')
    frames = [df1, df2, df3, df4]
    data = pd.concat(frames)
    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv('data.csv', index=False)
