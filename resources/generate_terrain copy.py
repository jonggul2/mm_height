import quat
import bvh
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import struct
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

""" Basic function for mirroring animation data with this particular skeleton structure """



filename = "hmap_007_smooth.txt"
data = []

# 텍스트 파일에서 데이터 읽기
with open(filename, 'r') as file:
    for line in file:
        row = [float(f) for f in line.split()]
        data.append(row)

# 데이터를 numpy 배열로 변환
data_array = np.array(data)

# 배열의 크기를 얻음
x_size, y_size = data_array.shape

# x 및 y의 새로운 간격을 설정
new_x_step = 1
new_y_step = 1

# 새로운 x 및 y 좌표 생성
new_x = np.linspace(0, x_size - 1, int(x_size / new_x_step))
new_y = np.linspace(0, y_size - 1, int(y_size / new_y_step))

# x 좌표를 뒤집음
new_x = new_x[::-1]
new_y = new_y[::-1]

# 새로운 x 및 y 좌표를 2차원 그리드로 변환
new_X, new_Y = np.meshgrid(new_x, new_y)

new_X = new_X.T
new_Y = new_Y.T
# 데이터를 새로운 좌표로 다시 샘플링
new_data = data_array[:, ::-1]


max_h = np.max(new_data)
min_h = np.min(new_data)

new_data_scaled = new_data + (max_h - min_h)
new_data_scaled = new_data_scaled * 10 / (max_h - min_h)

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D 그래프에 표시
ax.plot_surface(new_X, new_Y, new_data_scaled, cmap='viridis')

# 라벨 및 제목 추가
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Heightmap Visualization')

# 보여주기
plt.show()

height_map_normalized = ((new_data_scaled - new_data_scaled.min()) / (new_data_scaled.max() - new_data_scaled.min()) * 255).astype(np.uint8)

height_map_image = Image.fromarray(height_map_normalized)
height_map_image.save("heightmap1.png")
height_map_image.show()

np.savetxt('terrain_data1.txt', np.column_stack([new_X.flatten(), new_Y.flatten(), new_data_scaled.flatten()]), delimiter=',', comments='')





# filename = "hmap_007_smooth.txt"
# data = []

# # 텍스트 파일에서 데이터 읽기
# with open(filename, 'r') as file:
#     for line in file:
#         row = [float(f) for f in line.split()]
#         data.append(row)

# # 데이터를 numpy 배열로 변환
# data_array = np.array(data)

# # 3D 그래프 생성
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 배열의 크기를 얻음
# x_size, y_size = data_array.shape

# # x, y 좌표 생성
# x = np.arange(x_size)
# y = np.arange(y_size)

# # x, y 좌표를 2차원 그리드로 변환
# X, Y = np.meshgrid(x, y)

# # 데이터 배열을 Z 좌표로 사용
# Z = data_array

# # X 및 Y 좌표 그리드 전치
# X = X.T
# Y = Y.T

# # 3D 그래프에 표시
# ax.plot_surface(X, Y, Z, cmap='viridis')

# # 라벨 및 제목 추가
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('3D Heightmap Visualization')

# # 보여주기
# plt.show()

# height_map = Z
# x_grid = X
# z_grid = Y



# height_map_normalized = ((height_map - height_map.min()) / (height_map.max() - height_map.min()) * 255).astype(np.uint8)

# height_map_image = Image.fromarray(height_map_normalized)
# height_map_image.save("heightmap1.png")
# height_map_image.show()

# np.savetxt('terrain_data1.txt', np.column_stack([x_grid.flatten(), z_grid.flatten(), height_map.flatten()]), delimiter=',', comments='')


# print(np.max(X), np.max(Y))


# filename = "hmap_007_smooth.txt"

# data = []


# with open(filename, 'r') as file:
#     for line in file:
#         row = [float(f) for f in line.split()]
#         data.append(row)

# w = len(data)
# h = len(data[0])


# import pdb
# pdb.set_trace()


# offset = 0.0
# for x in range(w):
#     for y in range(h):
#         offset += data[x][y]
# offset /= w * h

# print(f"Loaded Heightmap '{filename}' ({w} {h})")

# posns = np.zeros((w * h, 3), dtype=np.float32)
# norms = np.zeros((w * h, 3), dtype=np.float32)
# aos = np.zeros(w * h, dtype=np.float32)



# def sample(pos):
#     w = len(data)
#     h = len(data[0])

#     pos = np.array(pos)
#     pos /= hscale
#     pos += np.array([w/2, h/2])

#     a0 = pos[0] % 1.0
#     a1 = pos[1] % 1.0

#     x0 = int(np.floor(pos[0]))
#     x1 = int(np.ceil(pos[0]))
#     y0 = int(np.floor(pos[1]))
#     y1 = int(np.ceil(pos[1]))

#     x0 = max(0, min(x0, w - 1))
#     x1 = max(0, min(x1, w - 1))
#     y0 = max(0, min(y0, h - 1))
#     y1 = max(0, min(y1, h - 1))

#     s0 = vscale * (data[x0][y0] - offset)
#     s1 = vscale * (data[x1][y0] - offset)
#     s2 = vscale * (data[x0][y1] - offset)
#     s3 = vscale * (data[x1][y1] - offset)

#     return ((s0 * (1 - a0) + s1 * a0) * (1 - a1)) + ((s2 * (1 - a0) + s3 * a0) * a1)




# hscale = 3.937007874
# vscale = 3.0


# for x in range(w):
#     for y in range(h):
#         cx = hscale * x
#         cy = hscale * y
#         cw = hscale * w
#         ch = hscale * h
#         posns[x + y * w] = (cx - cw/2, sample((cx - cw/2, cy - ch/2)), cy - ch/2)

# for x in range(w):
#     for y in range(h):
#         norms[x + y * w] = np.array([0, 1, 0]) if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1 else \
#             np.normalize(np.mix(np.cross(posns[(x + 0) + (y + 1) * w] - posns[x + y * w],
#                                             posns[(x + 1) + (y + 0) * w] - posns[x + y * w]),
#                                 np.cross(posns[(x + 0) + (y - 1) * w] - posns[x + y * w],
#                                             posns[(x - 1) + (y + 0) * w] - posns[x + y * w]), 0.5))








# def animation_mirror(lrot, lpos, names, parents):

#     joints_mirror = np.array([(
#         names.index('Left'+n[5:]) if n.startswith('Right') else (
#         names.index('Right'+n[4:]) if n.startswith('Left') else 
#         names.index(n))) for n in names])

#     mirror_pos = np.array([-1, 1, 1])
#     mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

#     grot, gpos = quat.fk(lrot, lpos, parents)

#     gpos_mirror = mirror_pos * gpos[:,joints_mirror]
#     grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:,joints_mirror]))
    
#     return quat.ik(grot_mirror, gpos_mirror, parents)

# """ Files to Process """

# # files = ['obstacles1_subject1.bvh', 'obstacles1_subject2.bvh', 'obstacles1_subject5.bvh']
# files = ['obstacles1_subject1.bvh']

# # folder_path = './bvh'
# # files = []
# # for file_name in os.listdir(folder_path):
# #     files.append(file_name)

# """ We will accumulate data in these lists """

# bone_positions = []
# bone_velocities = []
# bone_rotations = []
# bone_angular_velocities = []
# bone_parents = []
# bone_names = []
    
# range_starts = []
# range_stops = []

# contact_states = []
# contact_positions = []

# """ Loop Over Files """

# for filename in files:
    
#     # For each file we process it mirrored and not mirrored
#     mirror = False
#     # for mirror in [False, True]:
    
#     """ Load Data """
    
#     print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))
    
#     filepath = os.path.join("bvh", filename)
#     bvh_data = bvh.load(filepath)
#     bvh_data['positions'] = bvh_data['positions']
#     bvh_data['rotations'] = bvh_data['rotations']
    
#     positions = bvh_data['positions']
#     rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

#     # Convert from cm to m
#     positions *= 0.01
    
#     if mirror:
#         rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
#         rotations = quat.unroll(rotations)
    
#     """ Supersample """
    
#     nframes = positions.shape[0]
#     nbones = positions.shape[1]
    
#     # Supersample data to 60 fps
#     original_times = np.linspace(0, nframes - 1, nframes)
#     sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1))) # Speed up data by 10%
    
#     # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
#     positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
#     rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
    
#     # Need to re-normalize after super-sampling
#     rotations = quat.normalize(rotations)
    
#     """ Extract Simulation Bone """
    
#     # First compute world space positions/rotations
#     global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])
    
#     # Specify joints to use for simulation bone 
#     sim_position_joint = bvh_data['names'].index("Spine2")
#     sim_rotation_joint = bvh_data['names'].index("Hips")
    
#     # Position comes from spine joint
#     sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint:sim_position_joint+1]
#     sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')
    
#     # Direction comes from projected hip forward direction
#     sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1], np.array([0.0, 1.0, 0.0]))

#     # We need to re-normalize the direction after both projection and smoothing
#     sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
#     sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
#     sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])
    
#     # Extract rotation from direction
#     sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

#     # Transform first joints to be local to sim and append sim as root bone
#     positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
#     rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])
    
#     positions = np.concatenate([sim_position, positions], axis=1)
#     rotations = np.concatenate([sim_rotation, rotations], axis=1)
    
#     bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])
    
#     bone_names = ['Simulation'] + bvh_data['names']
    
#     """ Compute Velocities """
    
#     # Compute velocities via central difference
#     velocities = np.empty_like(positions)
#     velocities[1:-1] = (
#         0.5 * (positions[2:  ] - positions[1:-1]) * 60.0 +
#         0.5 * (positions[1:-1] - positions[ :-2]) * 60.0)
#     velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
#     velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
    
#     # Same for angular velocities
#     angular_velocities = np.zeros_like(positions)
#     angular_velocities[1:-1] = (
#         0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * 60.0 +
#         0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * 60.0)
#     angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
#     angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

#     """ Compute Contact Data """ 

#     global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
#         rotations, 
#         positions, 
#         velocities,
#         angular_velocities,
#         bone_parents)
    
#     contact_velocity_threshold = 0.15
    
#     contact_velocity = np.sqrt(np.sum(global_velocities[:,np.array([
#         bone_names.index("LeftToe"), 
#         bone_names.index("RightToe")])]**2, axis=-1))
    

#     # Contacts are given for when contact bones are below velocity threshold
#     contacts = contact_velocity < contact_velocity_threshold
    
#     # Median filter here acts as a kind of "majority vote", and removes
#     # small regions  where contact is either active or inactive

#     for ci in range(contacts.shape[1]):
    
#         contacts[:,ci] = ndimage.median_filter(
#             contacts[:,ci], 
#             size=6, 
#             mode='nearest')


#     left_contact = global_positions[:, bone_names.index("LeftToe"), :]
#     left_contact = left_contact[(contacts[:, 0] == True)]

#     right_contact = global_positions[:, bone_names.index("RightToe"), :]
#     right_contact = right_contact[(contacts[:, 1] == True)]

#     contact_position = np.concatenate((left_contact, right_contact), axis=0)
    
#     """ Append to Database """
    
#     bone_positions.append(positions)
#     bone_velocities.append(velocities)
#     bone_rotations.append(rotations)
#     bone_angular_velocities.append(angular_velocities)
    
#     offset = 0 if len(range_starts) == 0 else range_stops[-1] 

#     range_starts.append(offset)
#     range_stops.append(offset + len(positions))
    
#     contact_states.append(contacts)
#     contact_positions.append(contact_position)




# bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
# bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
# bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
# bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
# bone_parents = bone_parents.astype(np.int32)

# range_starts = np.array(range_starts).astype(np.int32)
# range_stops = np.array(range_stops).astype(np.int32)

# contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)
# contact_positions = np.concatenate(contact_positions, axis=0).astype(np.float32)


# x_train = np.array([contact_positions[:, 0], contact_positions[:, 2]]).T
# y_train = np.array(contact_positions[:, 1])



# model = KNeighborsRegressor(n_neighbors=5)
# model.fit(x_train, y_train)


# # points = []
# # x = -10
# # while x <= 10:
# #     points.append(x)
# #     x += 0.1

# # 포인트 출력
# # print(points)

# length = 10

# min_x = -length
# max_x = length
# min_z = -length
# max_z = length

# # num_points = 201

# points = np.arange(-10, 10.1, 0.1)

# x_values = points
# z_values = points
# # x_values = np.linspace(min_x, max_x, num_points)
# # z_values = np.linspace(min_z, max_z, num_points)
# x_grid, z_grid = np.meshgrid(x_values, z_values)
# xz_coordinates = np.column_stack([x_grid.ravel(), z_grid.ravel()])

# predicted_heights = model.predict(xz_coordinates)

# # print(np.max(predicted_heights))

# predicted_heights_grid = predicted_heights.reshape(x_grid.shape)
 
# height_map = predicted_heights_grid

# height_map_normalized = ((height_map - height_map.min()) / (height_map.max() - height_map.min()) * 255).astype(np.uint8)

# height_map_image = Image.fromarray(height_map_normalized)
# height_map_image.save("heightmap1.png")
# height_map_image.show()

# np.savetxt('terrain_data1.txt', np.column_stack([x_grid.flatten(), z_grid.flatten(), predicted_heights.flatten()]), delimiter=',', comments='')

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.set_xlim([min_x, max_x])
# ax.set_ylim([min_x, max_x])
# ax.set_zlim([min_x, max_x])

# surf = ax.plot_surface(x_grid, z_grid, predicted_heights_grid, cmap='terrain', edgecolor='none')


# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Height')
# ax.set_title('3D Terrain')

# plt.show()


# # min_x = -6
# # max_x = 6


# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # ax.set_xlim([min_x, max_x])
# # ax.set_ylim([min_x, max_x])
# # ax.set_zlim([0, max_x])

# # ax.scatter(contact_positions[:, 0], contact_positions[:, 2], contact_positions[:, 1], c='b', marker='o', s=20)

# # ax.set_xlabel('X')
# # ax.set_ylabel('Z')
# # ax.set_zlabel('Y')

# # plt.show()






    
# # """ Write Database """

# # print("Writing Database...")

# # with open('database.bin', 'wb') as f:
    
# #     nframes = bone_positions.shape[0]
# #     nbones = bone_positions.shape[1]
# #     nranges = range_starts.shape[0]
# #     ncontacts = contact_states.shape[1]

    
# #     f.write(struct.pack('II', nframes, nbones) + bone_positions.ravel().tobytes())
# #     f.write(struct.pack('II', nframes, nbones) + bone_velocities.ravel().tobytes())
# #     f.write(struct.pack('II', nframes, nbones) + bone_rotations.ravel().tobytes())
# #     f.write(struct.pack('II', nframes, nbones) + bone_angular_velocities.ravel().tobytes())
# #     f.write(struct.pack('I', nbones) + bone_parents.ravel().tobytes())
    
# #     f.write(struct.pack('I', nranges) + range_starts.ravel().tobytes())
# #     f.write(struct.pack('I', nranges) + range_stops.ravel().tobytes())
    
# #     f.write(struct.pack('II', nframes, ncontacts) + contact_states.ravel().tobytes())

    
    