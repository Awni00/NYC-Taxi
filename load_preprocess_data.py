import pandas as pd
import numpy as np


def calc_spherical_distance(lat1, lon1, lat2, lon2):
    '''
    Calculate the distance between two points in lattitude-longitude coordinates.
    '''

    from math import sin, cos, sqrt, atan2, radians

    #  radius of earth in km
    R = 6373.0

    # lattitude-longitude are given in degrees. convert to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def calc_manhattan_distance(lat1, lon1, lat2, lon2):
    '''calculate manhattan distance between two points on a spher'''

    pickup = (lat1, lon1)
    dropoff_a = (lat1, lon2)
    dropoff_b = (lat2, lon1)

    distance_a = calc_spherical_distance(*pickup, *dropoff_a)
    distance_b = calc_spherical_distance(*pickup, *dropoff_b)

    manhattan_distance = distance_a + distance_b

    return manhattan_distance

def calc_bearing(lat1, lon1, lat2, lon2):
    '''calculate the direction from pickup to dropoff'''

    from math import (
        degrees, radians,
        sin, cos, atan2
    )

    lon1, lat1, lon2, lat2 = (radians(coord) for coord in (lon1, lat1, lon2, lat2))

    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    numerator = sin(dlon) * cos(lat2)
    denominator = (
        cos(lat1) * sin(lat2) -
        (sin(lat1) * cos(lat2) * cos(dlon))
    )

    theta = atan2(numerator, denominator)
    theta_deg = (degrees(theta) + 360) % 360

    return theta_deg


def load_train_data(path):
    '''loads and preprocesses training data, creating additional features.'''

    # read dataframe
    train_data = pd.read_csv(path, index_col=0)

    # add log of trip_duration as variable (we see why later)
    train_data['log_trip_duration'] = np.log(train_data['trip_duration'])

    # convert pickup_date and pickup_time to pickup_datetime
    train_data.insert(0, 'pickup_datetime', train_data.pickup_date + ' ' + train_data.pickup_time)
    train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])

    train_data.drop(columns=['pickup_date', 'pickup_time'], inplace=True)

    # calculate distances using longitude-latitude coords of pickup, dropoff
    distances = [calc_spherical_distance(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                    for x in train_data.itertuples()]
    train_data.insert(2, 'distance_km', distances)

    # calculate l1 (manhattan) distance using longitude-latitude coords of pickup, dropoff
    manhattan_distances = [calc_manhattan_distance(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                for x in train_data.itertuples()]
    train_data.insert(3, 'l1_distance_km', manhattan_distances)

    bearing = [calc_bearing(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                for x in train_data.itertuples()]
    train_data.insert(4, 'bearing', bearing)


    # add dayofweek feature
    train_data.insert(1, 'dayofweek', train_data['pickup_datetime'].apply(lambda x: x.dayofweek))

    # add hour(ofday) feature
    train_data.insert(2, 'hour', train_data['pickup_datetime'].apply(lambda x: x.hour))

    return train_data

def load_test_data(path):
    '''loads and preprocesses testing data, creating additional features.'''

    # read dataframe
    test_data = pd.read_csv(path, index_col=0)

    # convert pickup_date and pickup_time to pickup_datetime
    test_data.insert(0, 'pickup_datetime', test_data.pickup_date + ' ' + test_data.pickup_time)
    test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])

    test_data.drop(columns=['pickup_date', 'pickup_time'], inplace=True)

    # calculate distances using longitude-lattitude coords of pickup, dropoff
    distances = [calc_spherical_distance(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                    for x in test_data.itertuples()]
    test_data.insert(2, 'distance_km', distances)

    # calculate l1 (manhattan) distance using longitude-latitude coords of pickup, dropoff
    manhattan_distances = [calc_manhattan_distance(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                for x in test_data.itertuples()]
    test_data.insert(3, 'l1_distance_km', manhattan_distances)

    bearing = [calc_bearing(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                for x in test_data.itertuples()]
    test_data.insert(4, 'bearing', bearing)

    # add dayofweek feature
    test_data.insert(1, 'dayofweek', test_data['pickup_datetime'].apply(lambda x: x.dayofweek))

    # add hour(ofday) feature
    test_data.insert(2, 'hour', test_data['pickup_datetime'].apply(lambda x: x.hour))

    return test_data