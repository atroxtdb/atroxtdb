import numpy as np

def mindistance(triangle1,triangle2):

    # Initialize a variable to store the minimum distance
    min_distance = float('inf')

    # Loop through all vertices of triangle 1
    for i in range(3):
        # Calculate the equation of the plane defined by triangle 2
        normal = np.cross(triangle2[1]-triangle2[0], triangle2[2]-triangle2[0])
        d = -np.dot(normal, triangle2[0])
        denom = np.dot(normal, normal)

        # Find the coordinates of the closest point on the plane of triangle 2 to the current vertex
        num = np.dot(normal, triangle1[i]) + d
        closest_point = triangle1[i] - (num/denom)*normal
        
        # Calculate the distance between the current vertex and the closest point
        distance = np.linalg.norm(triangle1[i]-closest_point)
        
        # Update the minimum distance if necessary
        min_distance = min(min_distance, distance)

    print("Minimum distance between the two triangles:", min_distance)
    
    
    
def gjk(shape1, shape2):
    # Initialize the search direction to be the negative of the difference between the centroids of the two shapes
    d = shape1.mean(axis=0) - shape2.mean(axis=0)
    # Initialize the simplex with the first point from the minkowski difference of the two shapes
    simplex = np.array([shape1[0] - shape2[0]])
    # Iterate until the origin is found or the maximum number of iterations is reached
    for _ in range(shape1.shape[0] + shape2.shape[0]):
        # Find the closest point on the minkowski difference to the current search direction
        closest_point = np.array([np.dot(d, shape1[i] - shape2[j]) for i in range(shape1.shape[0]) for j in range(shape2.shape[0])]).argmax()
        closest_point = shape1[np.argmax(np.dot(shape1, d))] - shape2[np.argmin(np.dot(shape2, d))]
        # Add the closest point to the simplex
        simplex = np.vstack((simplex, closest_point))
        # Check if the origin is inside the simplex
        if np.dot(np.dot(simplex, np.roll(simplex, 1, axis=0)), np.roll(simplex, 2, axis=0)).all() <= 0:
            return 0.0
        # Update the search direction to be the closest point to the origin on the simplex
        normal = np.cross(simplex[1] - simplex[0], simplex[2] - simplex[0])
        d = -normal / np.linalg.norm(normal)
        #d = np.dot(np.linalg.inv(simplex), np.array([1, 0, 0]))
    # If the origin is not found, return the distance from the origin to the simplex
    return np.linalg.norm(d)


def mpr(shape1, shape2):
    # Find the initial portal using the GJK algorithm
    d = shape1.mean(axis=0) - shape2.mean(axis=0)
    simplex = np.array([shape1[0] - shape2[0]])
    for _ in range(shape1.shape[0] + shape2.shape[0]):
        closest_point = shape1[np.argmax(np.dot(shape1, d))] - shape2[np.argmin(np.dot(shape2, d))]
        if np.dot(closest_point, d) <= 0:
            return (np.linalg.norm(d), d)
        simplex = np.vstack((simplex, closest_point))
        d = np.dot(np.linalg.inv(simplex), np.array([1, 0, 0]))
    # Refine the portal using a recursive algorithm
    while True:
        # Find the closest point on the portal to the origin
        closest_point = shape1[np.argmax(np.dot(shape1, d))] - shape2[np.argmin(np.dot(shape2, d))]
        # Check if the portal is small enough
        if np.dot(closest_point, d) <= 0:
            return (np.linalg.norm(d), d)
        # Split the portal into two smaller portals
        portal1 = np.array([simplex[1], closest_point])
        portal2 = np.array([simplex[2], closest_point])
        # Recursively refine each portal
        d1 = mpr(shape1, shape2, portal1)
        d2 = mpr(shape1, shape2, portal2)
        # Return the smallest portal
        if d1[0] < d2[0]:
            return d1
        else:
            return d2

# Define the first shape as a numpy array of vertices
shape1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

# Define the second shape as a numpy array of vertices
shape2 = np.array([[20, 30, 40], [50, 60, 70], [80, 90, 100], [110, 120, 130], [140, 150, 160]])

# Find the minimum distance and collision normal between the two shapes
min_distance, normal = mpr(shape1, shape2)
print("Minimum distance between the two shapes: ", min_distance)
print("Collision normal: ", normal)
