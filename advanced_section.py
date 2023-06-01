
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans, DBSCAN
from stl import mesh
from Space_Optimization import create_cloud
import os                                    

def rename_files():
    folder = "Issues/Actual_Problems"
    for count,filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{filename[:-4]}__issue.txt"
        os.rename(src,dst)

def maxcoords(triangles):
    return np.array(np.amax(triangles,axis=1))

def mincoords(triangles):
    return np.array(np.amin(triangles, axis=1))

def normal(triangles):
    return np.cross(triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0], axis=1)

def area(triangles):
    return np.linalg.norm(normal(triangles), axis=1) / 2

def plane(triangles):
    n = normal(triangles)
    u = n / np.linalg.norm(n, axis=1, keepdims=True)
    d = -np.einsum('ij,ij->i', triangles[:, 0], u)
    return np.hstack((u, d[:, None]))

def validate(Vertex_combined, Points):
    if len(Points) > 0:
        # create an array of indices to access the points array
        i = np.repeat(np.arange(len(Points)), [len(row) for row in Points])
        #j = np.concatenate([np.arange(len(row)) for row in Points])

        # extract the z-coordinates from the points array
        z = np.concatenate(Points)

        # create arrays for the three vertices
        A = np.stack((Vertex_combined[i, 0], Vertex_combined[i, 1], z), axis=1)
        B = np.stack((Vertex_combined[i, 1], Vertex_combined[i, 2], z), axis=1)
        C = np.stack((Vertex_combined[i, 0], Vertex_combined[i, 2], z), axis=1)

        # compute the areas
        areas_temp = area(np.stack((A, B, C), axis=1).reshape(-1,3,3)).reshape(-1,3)
        areas_sum = areas_temp.sum(axis=1)
        areas = area(Vertex_combined)
        areas = np.stack((areas[i]))

        # find the valid points
        mask = np.isclose(areas_sum, areas, atol=0.01)
        valid_points = np.concatenate(Points)[mask]
        return valid_points

def check_triangle_bbox_intersection(triangles, bbox):
    # compute the bounding box for each triangle
    tri_min = np.min(triangles, axis=1)
    tri_max = np.max(triangles, axis=1)

    # check if each vertex is inside the bounding box
    vert_mask = np.all((triangles >= bbox[0]) & (triangles <= bbox[1]), axis=-1)
    vert_intersect = np.any(vert_mask, axis=1)

    # check if any edge intersects the bounding box
    edge_intersect = np.zeros(triangles.shape[0], dtype=bool)
    for dim in range(3):
        for sign in [-1, 1]:
            # create a mask indicating which edges cross the boundary
            edge_mask = np.any((triangles[..., dim] * sign >= bbox[0][dim]) &
                               (triangles[..., dim] * sign <= bbox[1][dim]), axis=1)

            # if an edge crosses each dimension of the bbox, it intersects with it
            edge_intersect |= np.all(edge_mask, axis=-1)

    # check if any face intersects the bounding box
    tri_center = np.mean(triangles, axis=1)
    tri_normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    tri_d = -np.sum(tri_normal * tri_center, axis=-1)
    bbox_min, bbox_max = bbox
    bbox_corners = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]],
                             [bbox_max[0], bbox_min[1], bbox_min[2]],
                             [bbox_max[0], bbox_max[1], bbox_min[2]],
                             [bbox_min[0], bbox_max[1], bbox_min[2]],
                             [bbox_min[0], bbox_min[1], bbox_max[2]],
                             [bbox_max[0], bbox_min[1], bbox_max[2]],
                             [bbox_max[0], bbox_max[1], bbox_max[2]],
                             [bbox_min[0], bbox_max[1], bbox_max[2]]])
    bbox_center = np.mean(bbox_corners, axis=0)
    bbox_normal = np.cross(bbox_corners[1] - bbox_corners[0], bbox_corners[3] - bbox_corners[0])
    bbox_d = -np.sum(bbox_normal * bbox_center, axis=-1)
    face_intersect = np.zeros(triangles.shape[0], dtype=bool)
    for i in range(triangles.shape[0]):
        dist = (tri_normal[i] * bbox_center + tri_d[i]) / np.linalg.norm(tri_normal[i])
        if np.any(np.abs(dist) > np.sqrt(3) / 2):
            continue # triangle is too far away to intersect

        for j in range(3):
            edge_start = triangles[i, j]
            edge_end = triangles[i, (j + 1) % 3]
            edge_dir = edge_end - edge_start
            edge_len = np.linalg.norm(edge_dir)
            edge_dir /= edge_len
            t = - (tri_normal[i] * edge_start + tri_d[i]) / (tri_normal[i] @ edge_dir)
            if t < 0 or t > edge_len:
                continue # edge does not intersect triangle plane

            intersection = edge_start + t * edge_dir
            if np.all(bbox[0] <= intersection) and np.all(intersection <= bbox[1]):
                face_intersect[i] = True
    return vert_intersect | edge_intersect

def make_section(stl1,plane_normal,max_bbox,min_bbox,file_name):
    bbox = np.array((min_bbox,max_bbox))
    mesh1=stl1
    # Example plane normal vector
    res = 0.5
    if plane_normal[0] ==1:
        section_direction = 0
    elif plane_normal[1] ==1:
        section_direction = 1
    else:
        section_direction = 2 
    triangles = mesh1.vectors

    maxs = maxcoords(triangles)
    mins = mincoords(triangles)
    include_index = np.where((maxs[:,section_direction]>=-plane_normal[3]) & (mins[:,section_direction]<=-plane_normal[3]))
    triangles = triangles[include_index]




    planes = plane(triangles)

    mins = mincoords(triangles)
    maxs = maxcoords(triangles)

    points_x = [np.mgrid[(mins[i, 0]):(maxs[i, 0]):res] for i in range(len(mins))]
    points_y = [np.mgrid[(mins[i, 1]):(maxs[i, 1]):res] for i in range(len(mins))]
    points_z = [np.mgrid[(mins[i, 2]):(maxs[i, 2]):res] for i in range(len(mins))]

    points_x = [x.reshape(1,-1).T for x in points_x] 
    points_y = [x.reshape(1,-1).T for x in points_y] 
    points_z = [x.reshape(1,-1).T for x in points_z] 

    if section_direction == 0:
        points_xz = [np.insert(x,[0],-plane_normal[3],axis=1) for x in points_z]
        points_xy = [np.insert(x,[0],-plane_normal[3],axis=1) for x in points_y]
    elif section_direction == 1:
        points_yz = [np.insert(x,[0],-plane_normal[3],axis=1) for x in points_z]
        points_xy = [np.insert(x,[1],-plane_normal[3],axis=1) for x in points_x]
    else:
        points_xz = [np.insert(x,[1],-plane_normal[3],axis=1) for x in points_x]
        points_yz = [np.insert(x,[1],-plane_normal[3],axis=1) for x in points_y]


    if section_direction == 0 or section_direction == 2:
        # Calculate All possible XZ_Points of Plane
        temp = [-((planes[i][0] * np.array(points_xz[i])[:, 0] + planes[i][2] * np.array(points_xz[i])[:, 1] + planes[i][3]) / planes[i][1]) for i in range(len(planes))]
        XZ_Points = [np.hstack((np.atleast_2d(np.array(points_xz[i])[:, 0]).swapaxes(0, 1),np.atleast_2d(temp[i]).swapaxes(0, 1),np.atleast_2d(np.array(points_xz[i])[:, 1]).swapaxes(0, 1))) for i in range(len(temp))]    
        XZ_Points=validate(triangles, XZ_Points)
        
    if section_direction ==1 or section_direction == 2:
        # Calculate All possible YZ_Points of Plane
        temp = [-((planes[i][1] * np.array(points_yz[i])[:, 0] + planes[i][2] * np.array(points_yz[i])[:, 1] + planes[i][3]) / planes[i][0]) for i in range(len(planes))]
        YZ_Points = [np.hstack((np.atleast_2d(temp[i]).swapaxes(0, 1), np.array(points_yz[i]))) for i in range(len(temp))]
        YZ_Points = validate(triangles, YZ_Points)
        
    if section_direction == 0 or section_direction ==1:
    # Calculate All possible XY_Points of Plane
        temp = [-np.round_((planes[i][0] * np.array(points_xy[i])[:, 0] + planes[i][1] * np.array(points_xy[i])[:, 1] + planes[i][3]) / planes[i][2],2) for i in range(len(planes))]
        XY_Points = [np.hstack((np.array(points_xy[i]), np.atleast_2d(temp[i]).swapaxes(0, 1))) for i in range(len(temp))]
        XY_Points = validate(triangles, XY_Points)
        

    if 'XY_Points' in locals() and XY_Points is not None:
        XY_Points = XY_Points[np.where(np.all(bbox[1]>=XY_Points,axis=1) & np.all(bbox[0]<=XY_Points,axis=1)==True)]
        XY_Points = np.array(XY_Points)
        XY_Points = XY_Points.reshape(-1,3)

    if 'XZ_Points' in locals() and XZ_Points is not None :
        XZ_Points = XZ_Points[np.where(np.all(bbox[1]>=XZ_Points,axis=1) & np.all(bbox[0]<=XZ_Points,axis=1)==True)]
        XZ_Points = np.array(XZ_Points)
        XZ_Points = XZ_Points.reshape(-1,3)

    if 'YZ_Points' in locals() and YZ_Points is not None:
        YZ_Points = YZ_Points[np.where(np.all(bbox[1]>=YZ_Points,axis=1) & np.all(bbox[0]<=YZ_Points,axis=1)==True)]
        YZ_Points = np.array(YZ_Points)
        YZ_Points = YZ_Points.reshape(-1,3)

    if section_direction ==0 :
        if XZ_Points is not None and XY_Points is not None:
            vertices1 = np.concatenate((XZ_Points,XY_Points))
        elif XZ_Points is not None:
            vertices1 = XZ_Points
        else:
            vertices1 = XY_Points
        section_file = 'Issues\\'+ file_name + f'__section_x_{str(-plane_normal[3])}.txt'
    elif section_direction ==1:
        if XY_Points is not None and YZ_Points is not None:
            vertices1 = np.concatenate((XY_Points,YZ_Points))
        elif XY_Points is not None:
            vertices1 = XY_Points
        else:
            vertices1 = YZ_Points
        section_file = 'Issues\\'+ file_name + f'__section_y_{str(-plane_normal[3])}.txt'
    else:
        if XZ_Points is not None and YZ_Points is not None:
            vertices1 = np.concatenate((XZ_Points,YZ_Points))
        elif XZ_Points is not None:
            vertices1 = XZ_Points
        else:
            vertices1 = YZ_Points
        section_file = 'Issues\\'+ file_name + f'__section_z_{str(-plane_normal[3])}.txt'

    with open (section_file,'a') as section_file: 
        if 'XY_Points' in locals() and XY_Points is not None:
            np.savetxt(section_file, XY_Points, delimiter=' ')
        if 'YZ_Points' in locals() and YZ_Points is not None:
            np.savetxt(section_file, YZ_Points, delimiter=' ')
        if 'XZ_Points' in locals() and XZ_Points is not None:
            np.savetxt(section_file, XZ_Points, delimiter=' ')

def filereader(path):
    """
    File reading for Point Cloud generated using STL
    """
    coordinates1 = []
    Vertex = []
    facet_normals = []
    with open(path, encoding="utf-8") as file1:
        for lines in file1:
            Vertex = lines.split()
            k = []
            norm =[]
            Vertex = Vertex[0:3]
            for i in Vertex:
                j = i.split('e')
                if len(j) == 2:
                    k.append(round(float(j[0]) * 10 ** float(j[1]), 2))
                else:
                    k.append(round(float(j[0]), 2))
            coordinates1.append(np.array(k))
    file1.close()
    return np.array(coordinates1)

file1 = "77271GW000________________L4_REINF_RR_DR_HINGE_FACE_LH__________AW__BB770A___________________R411090"
file2 = "77211GW000________________XA_PNL_RR_DR_INR_LH___________________AW__BB770A___________________R411090"

create_cloud(file1)
create_cloud(file2)

mesh1 = mesh.Mesh.from_file(file1+'.stl') 
mesh2 = mesh.Mesh.from_file(file2+'.stl') 

part1 = filereader(file1 + "__refined.txt")
part2 = filereader(file2 + "__refined.txt")

tree1= KDTree(part1.reshape(-1,3))
tree2= KDTree(part2.reshape(-1,3))

treshold = 2.0
distances, indices =tree1.query(part2.reshape(-1,3),distance_upper_bound=treshold)

valid_indices = np.where(distances < treshold)[0]

valid_indices_critical_high = np.where((0 < distances) & (distances < treshold/4))[0]
valid_indices_critical_mid = np.where((treshold/4 < distances) & (distances < treshold/2))[0]
valid_indices_critical_mid_low = np.where((treshold/2 < distances) & (distances < 3*treshold/4))[0]
valid_indices_critical_low = np.where((3*treshold/4 < distances ) & (distances< treshold))[0]

vertices1 = part1[indices[valid_indices]]
vertices2 = part2[valid_indices]

lines = np.concatenate((vertices1,vertices2))

dbscan = DBSCAN(eps=2,min_samples=10)
labels_points = dbscan.fit_predict(vertices1)


np.savetxt('points1.txt', np.round_(vertices1,3),delimiter=' ')
np.savetxt('points2.txt', vertices2, delimiter=' ')


np.savetxt('valid_indices_critical_high_1.txt', np.insert(part1[indices[valid_indices_critical_high]],[3],(250,0,0),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")
np.savetxt('valid_indices_critical_high_2.txt', np.insert(part2[valid_indices_critical_high],[3],(250,0,0),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")

np.savetxt('valid_indices_critical_mid_1.txt', np.insert(part1[indices[valid_indices_critical_mid]],[3],(255,140,0),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")
np.savetxt('valid_indices_critical_mid_2.txt', np.insert(part2[valid_indices_critical_mid],[3],(255,140,0),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")

np.savetxt('valid_indices_critical_mid_low_1.txt', np.insert(part1[indices[valid_indices_critical_mid_low]],[3],(255,255,51),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")
np.savetxt('valid_indices_critical_mid_low_2.txt', np.insert(part2[valid_indices_critical_mid_low],[3],(255,255,51),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")

np.savetxt('valid_indices_critical_low_1.txt', np.insert(part1[indices[valid_indices_critical_low]],[3],(0,255,0),axis=1), fmt="%1.5f %1.5f %1.5f %d %d %d")
np.savetxt('valid_indices_critical_low_2.txt', np.insert(part2[valid_indices_critical_low],[3],(0,255,0),axis=1) ,fmt="%1.5f %1.5f %1.5f %d %d %d")


for i in range(0,labels_points.max(),1):
    np.savetxt('points1__{:02d}.txt'.format(i), np.round_(vertices1[np.where(labels_points==i)[0]],3),delimiter=' ')
    problem_region = vertices1[np.where(labels_points==i)[0]]
    max_bbox,min_bbox =np.amax(problem_region,axis=0)+10.0,np.amin(problem_region,axis=0)-10.0
    xlist = np.mgrid[(min_bbox[0]):(max_bbox[0]):1]
    for j in xlist:
        make_section(mesh1,[1,0,0,-int(j)],max_bbox,min_bbox,file1)
        make_section(mesh2,[1,0,0,-int(j)],max_bbox,min_bbox,file2)
    ylist = np.mgrid[(min_bbox[1]):(max_bbox[1]):1]
    for j in ylist:
        make_section(mesh1,[0,1,0,-int(j)],max_bbox,min_bbox,file1)
        make_section(mesh2,[0,1,0,-int(j)],max_bbox,min_bbox,file2)
    zlist = np.mgrid[(min_bbox[2]):(max_bbox[2]):1]
    for j in zlist:
        make_section(mesh1,[0,0,1,-int(j)],max_bbox,min_bbox,file1)
        make_section(mesh2,[0,0,1,-int(j)],max_bbox,min_bbox,file2)
print("Process Finished")



# Step 1: Load the section data and labels
section_data = np.load("section_data.npy")
labels = np.load("labels.npy")

# Step 2: Determine the desired target size (bounding box)
target_size = [10.0, 10.0] # Set the desired size for all sections (e.g., 10 units by 10 units)

# Step 3: Resize the sections by rescaling the coordinates
rescaled_sections = []
for section in section_data:
    # Get the minimum and maximum coordinates in each dimension
    min_coords = np.min(section, axis=0)
    max_coords = np.max(section, axis=0)

    # Calculate the scaling factors for each dimension
    scaling_factors = target_size / (max_coords - min_coords)

    # Rescale the coordinates by multiplying with the scaling factors
    rescaled_section = (section - min_coords) * scaling_factors

    rescaled_sections.append(rescaled_section)

rescaled_sections = np.array(rescaled_sections)

# Step 4: Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    rescaled_sections, labels, test_size=0.2, random_state=42
)

# Rest of the code (model definition, training loop, etc.) remains the same


# from torch.nn import Module
# from torch.nn import Conv2d
# from torch.nn import Linear
# from torch.nn import MaxPool2d
# from torch.nn import ReLU
# from torch.nn import LogSoftmax
# from torch import flatten

# class ClashDetection(Module):
#     def __init__(self,numChannels,classes):
#         super(ClashDetection,self).__init__()

#         self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,kernel_size=(5,5))
#         self.relu1 = ReLU()
#         self.maxpool1 = MaxPool2d(kernel_size=(2,2),stride=(2,2))


#         self.conv1 = Conv2d(in_channels=20, out_channels=50,kernel_size=(5,5))
#         self.relu2 = ReLU()
#         self.maxpool2 = MaxPool2d(kernel_size=(2,2),stride=(2,2))

#         self.fc1 = Linear(in_features=800,out_features=500)
#         self.relu3 = ReLU()
        
#         self.fc2 = Linear(in_features=500,out_features=classes)
#         self.logSoftmax =LogSoftmax(dim=1)

#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)

#         x=self.conv2(x)
#         x=self.relu2(x)
#         x=self.maxpool2(x)

#         x= flatten(x,1)
#         x= self.fc1(x)
#         x = self.relu3(x)

#         x = self.fc2(x)
#         output = self.logSoftmax(x)

#         return output
