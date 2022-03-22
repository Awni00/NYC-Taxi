import pandas as pd
import numpy as np


def calc_haversine_distance(lat1, lon1, lat2, lon2):
    '''
    Calculate the distance between two points in lattitude-longitude coordinates.

    Uses Haversine distance.
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

def manhattan_dist(lat1, lon1, lat2, lon2):
    # code based on https://medium.com/@simplyjk/why-manhattan-distance-formula-doesnt-apply-to-manhattan-7db0ebb1c5f6
    # by Jayakrishnan (JK) Vijayaraghavan

    # Pickup coordinates
    p = np.expand_dims(np.array([lat1, lon1]), axis=0)

    # Dropoff coordinates
    d = np.expand_dims(np.array([lat2, lon2]), axis=0)

    # inclination of manhattan w.r.t. geographic north
    theta1 = np.radians(-28.904)
    theta2 = np.radians(28.904)

    ## Rotation matrix
    R1 = np.array([[np.cos(theta1), np.sin(theta1)],
                    [-np.sin(theta1), np.cos(theta1)]]
                    )
    R2 = np.array([[np.cos(theta2), np.sin(theta2)],
                    [-np.sin(theta2), np.cos(theta2)]]
                    )

    # Rotate Pickup and Dropoff coordinates by -29 degress in World2
    pT = R1 @ p.T
    dT = R1 @ d.T

    # Coordinates of Hinge point in the rotated world
    vT = np.stack((pT[0,:], dT[1,:]))

    # Coordinates of Hinge point in the real world
    v = R2 @ vT

    ax1_dist = calc_haversine_distance(p.T[0], p.T[1], v[0], v[1])
    ax2_dist = calc_haversine_distance(v[0], v[1], d.T[0], d.T[1])

    manhattan_dist = ax1_dist + ax2_dist

    return manhattan_dist

def load_train_data(path):

    # read dataframe
    train_data = pd.read_csv(path, index_col=0)

    # add log of trip_duration as variable (we see why later)
    train_data['log_trip_duration'] = np.log(train_data['trip_duration'])

    # convert pickup_date and pickup_time to pickup_datetime
    train_data.insert(0, 'pickup_datetime', train_data.pickup_date + ' ' + train_data.pickup_time)
    train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])

    train_data.drop(columns=['pickup_date', 'pickup_time'], inplace=True)

    # calculate distances using longitude-latitude coords of pickup, dropoff
    distances = [calc_haversine_distance(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                    for x in train_data.itertuples()]
    train_data.insert(2, 'distance_km', distances)

    # calculate l1 (manhattan) distance using longitude-latitude coords of pickup, dropoff
    manhattan_distances = [manhattan_dist(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                for x in train_data.itertuples()]
    train_data.insert(3, 'l1_distance_km', manhattan_distances)

    # add dayofweek feature
    train_data.insert(1, 'dayofweek', train_data['pickup_datetime'].apply(lambda x: x.dayofweek))

    # add hour(ofday) feature
    train_data.insert(2, 'hour', train_data['pickup_datetime'].apply(lambda x: x.hour))

    return train_data

def load_test_data(path):

    # read dataframe
    test_data = pd.read_csv(path, index_col=0)

    # convert pickup_date and pickup_time to pickup_datetime
    test_data.insert(0, 'pickup_datetime', test_data.pickup_date + ' ' + test_data.pickup_time)
    test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])

    test_data.drop(columns=['pickup_date', 'pickup_time'], inplace=True)

    # calculate distances using longitude-lattitude coords of pickup, dropoff
    distances = [calc_haversine_distance(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                    for x in test_data.itertuples()]
    test_data.insert(2, 'distance_km', distances)

    # calculate l1 (manhattan) distance using longitude-latitude coords of pickup, dropoff
    manhattan_distances = [manhattan_dist(x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude)
                for x in test_data.itertuples()]
    test_data.insert(3, 'l1_distance_km', manhattan_distances)

    # add dayofweek feature
    test_data.insert(1, 'dayofweek', test_data['pickup_datetime'].apply(lambda x: x.dayofweek))

    # add hour(ofday) feature
    test_data.insert(2, 'hour', test_data['pickup_datetime'].apply(lambda x: x.hour))

    return test_data