# r'C:\Users\darao\Desktop\College\Connect\Geolife Trajectories 1.3\Data'

import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os


def read_plt(plt_file):
    # This function reads the .plt file and returns it as a pandas dataframe
    # Skips the first 6 rows and reads the date in the first row and uses the same format for all the other rows
    points = pd.read_csv(plt_file, skiprows=6, header=None, parse_dates={'time': [5, 6]}, infer_datetime_format=True)
    # Rename the columns
    points.rename(inplace=True, columns={0: 'lat', 1: 'lon', 4: 'sec'})
    # Remove the unwanted columns (All zero column and altitude)
    points.drop(inplace=True, columns=[2, 3])

    return points


def read_labels(labels_file):
    # This function reads the labels file and returns it as a pandas dataframe
    # Skip the first row, parse the dates like plt file and use any white space as sep
    labels = pd.read_csv(labels_file, skiprows=1, header=None, parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, sep=r"\s+")
    # Name the columns
    labels.columns = ['start_time', 'end_time', 'label']

    return labels


def apply_labels(points, labels):
    # This function returns the labels to the points dataframe
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    # Need to add id so individual routes can be determined
    points['ID'] = labels.iloc[indices].index
    points['label'][no_label] = 0

# Global variable that increments for file naming.
taxi_id = 1


def create_taxi_route(route):
    global taxi_id
    ids = route.ID.unique()
    for ride_id in ids:
        taxi_route = route[route.ID == ride_id]
        drop_columns = taxi_route[['sec', 'lat', 'lon']]
        drop_columns.to_csv(os.path.join(r'Motor', str(taxi_id)+'.csv'))
        taxi_id += 1
        print(drop_columns)


walk_id = 1


def create_walk_route(route):
    global walk_id
    ids = route.ID.unique()
    for id in ids:
        walk_route = route[route.ID == id]
        drop_columns = walk_route[['sec', 'lat', 'lon']]
        drop_columns.to_csv(os.path.join(r'Walk', str(walk_id)+'.csv'))
        walk_id += 1
        # print(drop_columns)


train_id = 1


def create_train_route(route):
    global train_id
    ids = route.ID.unique()
    for id in ids:
        train_route = route[route.ID == id]
        drop_columns = train_route[['sec', 'lat', 'lon']]
        drop_columns.to_csv(os.path.join(r'Train', str(train_id)+'.csv'))
        train_id += 1
        # print(drop_columns)

bike_id = 1


def create_bike_route(route):
    global bike_id
    ids = route.ID.unique()
    for id in ids:
        bike_route = route[route.ID == id]
        drop_columns = bike_route[['sec', 'lat', 'lon']]
        drop_columns.to_csv(os.path.join(r'Bike', str(bike_id)+'.csv'))
        bike_id += 1


bus_id = 1


def create_bus_route(route):
    global bus_id

    drop_columns = route[['sec', 'lat', 'lon']]
    drop_columns.to_csv(os.path.join(r'All_Data', str(bus_id)+'.csv'))
    bus_id += 1


#
def read_user(user_folder):
    labels = None
    # Find all .plt file in the trajectory file of the user
    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    # Read all the files in plt_files and concatenate to the one dataframe
    df = pd.concat([read_plt(f) for f in plt_files])
    # Find directory of the labels file
    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
        # Drop all rows that don't have taxi as a label
        # f = df.loc[df['label'] == 'taxi']
        # create_taxi_route(f)
        f = df.loc[df['label'] == 'walk']
        create_walk_route(f)
        # f = df.loc[df['label'] == 'bus']
        # create_bus_route(f)
        # f = df.loc[df['label'] == 'bike']
        # create_bike_route(f)
        f = df.loc[df['label'] == 'train']
        create_train_route(f)
    else:
        df['label'] = 0

    return df


# def read_user(user_folder):
#     labels = None
#     # Find all .plt file in the trajectory file of the user
#     plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
#     # Read all the files in plt_files and concatenate to the one dataframe
#     df = pd.concat([read_plt(f) for f in plt_files])
#     create_bus_route(df)
#     return df


def read_all_users(folder):
    # Find the adress of all the user folders
    subfolders = os.listdir(folder)
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder, sf))


if __name__ == '__main__':
    df = read_all_users(r'C:\Users\darao\Desktop\College\Connect\Geolife Trajectories 1.3\Data')

