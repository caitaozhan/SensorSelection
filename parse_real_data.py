import pandas as pd
import numpy as np
import geopy.distance
import pandas as pd
from subprocess import Popen, PIPE
import matplotlib as plt
import matplotlib.pyplot as plt
import mpu

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)
print (geopy.distance.distance(coords_1, coords_2).km)

def read_gps_file(origin_file):
    print(origin_file)
    df = pd.read_csv(origin_file, sep=',', header=None)
    df[4] = df[4] * -1
    # df[2] = df[2] / 100
    # df[4] = df[4] / 100
    lat_values_in_gps_format = df[2].values
    lon_values_in_gps_format = df[4].values

    DD = (lat_values_in_gps_format.astype(float) / 100.0).astype(int)
    MM = lat_values_in_gps_format.astype(float) - DD * 100
    latDec = DD + MM / 60

    DD = (lon_values_in_gps_format.astype(float) / 100.0).astype(int)
    MM = lon_values_in_gps_format.astype(float) - DD * 100
    lonDec = DD + MM / 60

    lat = np.mean(latDec)
    lon = np.mean(lonDec)
    return np.array([lat, lon])

def read_tran_gps_file(origin_file):
    df = pd.read_csv(origin_file, sep=',', header=None)
    df[4] = df[4] * -1

    lat_values_in_gps_format = df[2].values
    lon_values_in_gps_format = df[4].values

    DD = (lat_values_in_gps_format.astype(float) / 100.0).astype(int)
    MM = lat_values_in_gps_format.astype(float) - DD * 100
    latDec = DD + MM / 60

    DD = (lon_values_in_gps_format.astype(float) / 100.0).astype(int)
    MM = lon_values_in_gps_format.astype(float) - DD * 100
    lonDec = DD + MM / 60

    df[2] = latDec
    df[4] = lonDec
    return df

def construct_lat_long_grid():
    dir_name = '/home/arani2/misc-work/testbed-southP/GPS_locations/'
    grid = np.zeros((10, 10, 2))

    for i in range(10):
        for j in range(10):
            filename = dir_name + str(i) + str(j) +'.gpgga'
            grid[i, j] = read_gps_file(filename)
    return grid

def find_grid(tran_gps_df, grid_of_lat_lon):
    distances = np.zeros((10, 10))
    list_of_min_distances = np.zeros(len(tran_gps_df))
    valid_data = np.ones(len(tran_gps_df), dtype=bool)

    tran_gps_df['distances'] = np.zeros(len(tran_gps_df))
    tran_gps_df['gain'] = np.zeros(len(tran_gps_df))
    gain = 80

    for index, row in tran_gps_df.iterrows():
        for i in range(10):
            for j in range(10):
                coords_1 = (row[2], row[4])
                coords_2 = (grid_of_lat_lon[i][j][0], grid_of_lat_lon[i][j][1])
                distances[i, j] = mpu.haversine_distance(coords_1, coords_2) * 1000

        min_index = np.argmin(distances)
        try:
            list_of_min_distances[index] = np.min(distances)
            if list_of_min_distances[index] > 0.8:
                valid_data[index] = False
                tran_gps_df.at[index, 'distances'] = np.nan
                #row['distances'] = np.nan
            else:
                min_x = min_index // 10
                min_y = min_index % 10
                tran_gps_df.at[index, 'x_coords'] = min_x
                tran_gps_df.at[index, 'y_coords'] = min_y
                tran_gps_df.at[index, 'gain'] = gain
                row['x_coords'] = min_x
                row['y_coords'] = min_y

                if (min_x == 9.0 and min_y == 9.0):
                    gain = 60
                    print(index, min_x, min_y)

                if index % 100 == 0:
                    print(index)
        except:
            pass
    tran_gps_df = tran_gps_df.dropna(subset=['distances'])
    #tran_gps_df = tran_gps_df[tran_gps_df['gain'] == 80]
    tran_gps_df.to_csv('filtered_tran.csv')

    return tran_gps_df
    #bins = np.arange(np.min(list_of_min_distances), np.max(list_of_min_distances), 0.1)
    #data = plt.hist(list_of_min_distances, bins=bins)
    #print(data)
    #plt.show()

grid_of_lat_lon = construct_lat_long_grid()
tran_gps_df = read_tran_gps_file('/home/arani2/misc-work/testbed-southP/laptop/GPGGA')
#print(tran_gps_df)
tran_gps_df = tran_gps_df.dropna(subset=[2, 4])
tran_gps_df = find_grid(tran_gps_df, grid_of_lat_lon)
tran_gps_df = tran_gps_df.groupby(['x_coords', 'y_coords']).first()
tran_gps_df.to_csv('filtered')
#print(tran_gps_df)

# def read_gpgga_file(origin_file):
#     df = pd.read_csv(origin_file, sep=',', header=None)
#     df[4] = df[4] * -1
#     #df[2] = df[2] / 100
#     #df[4] = df[4] / 100
#     lat_values_in_gps_format = df[2].values
#     lon_values_in_gps_format = df[4].values
#
#     DD = (lat_values_in_gps_format.astype(float) / 100.0).astype(int)
#     MM = lat_values_in_gps_format.astype(float) - DD * 100
#     latDec = DD + MM / 60
#     print(DD, MM, latDec)
#
#     DD = (lon_values_in_gps_format.astype(float) / 100.0).astype(int)
#     MM = lon_values_in_gps_format.astype(float) - DD * 100
#     lonDec = DD + MM / 60
#
#     lat = np.mean(latDec)
#     lon = np.mean(lonDec)
#     return (lat, lon)

# origin =  (40.896771, -73.126358)
# end_of_x = (40.896797, -73.126216)
# end_of_x_y = read_gpgga_file('/home/arani2/misc-work/testbed-southP/laptop/99-GPGGA')
# lat_slope_of_y_axis = end_of_x[0] - origin[0]
# lon_slope_of_y_axis = end_of_x[1] - origin[1]
# lat_slope_of_x_axis = end_of_x_y[0] - end_of_x[0]
# lon_slope_of_x_axis = end_of_x_y[1] - end_of_x[1]
# end_of_y = (origin[0] + lat_slope_of_x_axis, origin[1] + lon_slope_of_x_axis)
#
# print('0, 0', origin)
# print('9, 0', end_of_x)
# print('9, 9', end_of_x_y)
# print('0, 9', end_of_y)
#
# grid_points_y_axis = [(origin[0] + lat_slope_of_y_axis * i, (origin[1] + lon_slope_of_x_axis * i)) for i in range(10)]
# grid_points_x_axis = [(origin[0] + lat_slope_of_x_axis * i, (origin[1] + lon_slope_of_x_axis * i)) for i in range(10)]
# x_points = [i % 100 for i in range(100)]
# y_points = [i // 100 for i in range(100)]
# grid_points = [(grid_points_y_axis[x_points[i]] + x_points[i] * lat_slope_of_y_axis + grid_points_x_axis[y_points[i]] * y_points[i] * lat_slope_of_y_axis,
#                grid_points_x_axis[y_points[i]] + x_points[i] * lon_slope_of_x_axis + grid_points_y_axis[x_points[i]] * y_points[i] * lon_slope_of_y_axis) for i in range(100)]
# print(grid_points)



