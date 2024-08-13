import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random
from datetime import datetime
plt.ion()
start_time = time.time()

print("Hi! Welcome to the simulation for Informed RRT* algorithm")
print("______________________________________________________________________")

def map():

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 300])
    ax.set_ylim([-100, 100])    

    rectangle_1 = patches.Rectangle((70, 47.5), width=15, height=15, color='red')
    rectangle_2 = patches.Rectangle((70, -07.5), width=15, height=15, color='red')
    rectangle_3 = patches.Rectangle((70, -62.5), width=15, height=15, color='red')

    rectangle_4 = patches.Rectangle((145, 85), width=15, height=15, color='red')
    rectangle_5 = patches.Rectangle((145, 25), width=15, height=15, color='red')
    rectangle_6 = patches.Rectangle((145, -40), width=15, height=15, color='red')
    rectangle_7 = patches.Rectangle((145, -100), width=15, height=15, color='red')
    
    rectangle_8 = patches.Rectangle((220, 47.5), width=15, height=15, color='red')
    rectangle_9 = patches.Rectangle((220, -7.5), width=15, height=15, color='red')
    rectangle_10 = patches.Rectangle((220, -62.5), width=15, height=15, color='red')

    ax.add_patch(rectangle_1)
    ax.add_patch(rectangle_2)
    ax.add_patch(rectangle_3)

    ax.add_patch(rectangle_4)
    ax.add_patch(rectangle_5)
    ax.add_patch(rectangle_6)
    ax.add_patch(rectangle_7)

    ax.add_patch(rectangle_8)
    ax.add_patch(rectangle_9)
    ax.add_patch(rectangle_10)
    
    ax.set_aspect('equal')
    plt.show()


def rectangle_1(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(70-total_bloat) >= 0 and x-(85+total_bloat) <= 0 and y-(47.5-total_bloat) >= 0 and y-(62.5+total_bloat) <= 0:
        return True
    else:
        return False
    
def rectangle_2(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(70-total_bloat) >= 0 and x-(85+total_bloat) <= 0 and y-(-07.5-total_bloat) >= 0 and y-(07.5+total_bloat) <= 0:
        return True
    else:
        return False

def rectangle_3(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(70-total_bloat) >= 0 and x-(85+total_bloat) <= 0 and y-(-62.5-total_bloat) >= 0 and y-(-47.5+total_bloat) <= 0:
        return True
    else:
        return False
    
def rectangle_4(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(145-total_bloat) >= 0 and x-(160+total_bloat) <= 0 and y-(25-total_bloat) >= 0 and y-(40+total_bloat) <= 0:
        return True
    else:
        return False

def rectangle_5(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(145-total_bloat) >= 0 and x-(160+total_bloat) <= 0 and y-(85-total_bloat) >= 0 and y-(100+total_bloat) <= 0:
        return True
    else:
        return False
    
def rectangle_6(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(145-total_bloat) >= 0 and x-(160+total_bloat) <= 0 and y-(-40-total_bloat) >= 0 and y-(-25+total_bloat) <= 0:
        return True
    else:
        return False
    
def rectangle_7(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(145-total_bloat) >= 0 and x-(160+total_bloat) <= 0 and y-(-100-total_bloat) >= 0 and y-(-85+total_bloat) <= 0:
        return True
        return True
    else:
        return False
    
def rectangle_8(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(220-total_bloat) >= 0 and x-(235+total_bloat) <= 0 and y-(47.5-total_bloat) >= 0 and y-(62.5+total_bloat) <= 0:
        return True
    else:
        return False
    
def rectangle_9(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(220-total_bloat) >= 0 and x-(235+total_bloat) <= 0 and y-(-07.5-total_bloat) >= 0 and y-(7.5+total_bloat) <= 0:
        return True
    else:
        return False

def rectangle_10(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if x-(220-total_bloat) >= 0 and x-(235+total_bloat) <= 0 and y-(-62.5-total_bloat) >= 0 and y-(-47.5+total_bloat) <= 0:
        return True
    else:
        return False

def wall(input):
    x = input[0]
    y = input[1]
    total_bloat = input[2]
    if (x < 0) or (x > 300) or (y - total_bloat <= -100) or (y + total_bloat >= 100):
        return True
    else:
        return False
    
def if_obstacle(input):
    if (wall(input) or rectangle_1(input) or rectangle_2(input) or rectangle_3(input) or rectangle_4(input) or rectangle_5(input) or rectangle_6(input) or rectangle_7(input) or rectangle_8(input) or rectangle_9(input) or rectangle_10(input))  == True:
        return True
    else:
        return False
    
#Heuristic eucledian distance
def eucledian_distance(coordinate_1, coordinate_2):
    distance = np.sqrt((((coordinate_2[0] - coordinate_1[0]) ** 2) + ((coordinate_2[1] - coordinate_1[1]) ** 2)))
    return distance
# Finding a random point in the map and the ellipsoidal region
def generate_random_point(cost_max, cost_min, ellipse_center, cost_matrix, bloat):
    if cost_max == float('inf'):
        random_point_x = round(random.uniform(0, 300), 2)
        random_point_y = round(random.uniform(-100 + bloat, 100 - bloat), 2)
    else:
        radii_vector = [cost_max / 2.0, np.sqrt(cost_max ** 2 - cost_min ** 2) / 2.0, np.sqrt(cost_max ** 2 - cost_min ** 2) / 2.0]
        diagonal_radii_matrix = np.diag(radii_vector)

        minor_axis, major_axis = sorted([random.random(), random.random()])
        direct_sample = (major_axis * np.cos(2 * np.pi * minor_axis / major_axis), major_axis * np.sin(2 * np.pi * minor_axis / major_axis), 0)

        spheroid = np.array(direct_sample).reshape(-1, 1)
        random_point = np.dot(np.dot(cost_matrix, diagonal_radii_matrix), spheroid)
        random_point = random_point + ellipse_center
        random_point_x, random_point_y = round(random_point[0, 0], 2), round(random_point[1, 0], 2)

    return random_point_x, random_point_y

 # Finding the nearest neighbour in the graph for the random point
def generate_best_neighbour(current_x, current_Y, explored_nodes):
    minimum_distance = float('inf')
    nearest_node = -1
    
    for node in explored_nodes:
        dist = eucledian_distance(node, (current_x, current_Y))
        if(dist < minimum_distance):
            minimum_distance = dist
            nearest_node = node
    return nearest_node

# Finding a new child between random and nearest point
def generate_child(random_point, nearest_point, bloat):
    stepSize = 4
    # slope of line joining random point and nearest point
    slope_of_line = (random_point[1] - nearest_point[1]) / (random_point[0] - nearest_point[0])
    factor = stepSize * np.sqrt(1 / (1 + (slope_of_line ** 2)))    
    # creating two possible points
    point_1 = (round(nearest_point[0] + factor, 2), round(nearest_point[1] + (slope_of_line * factor), 2))
    point_2 = (round(nearest_point[0] - factor, 2), round(nearest_point[1] - (slope_of_line * factor), 2))
    bool_value1 = False
    bool_value2 = False
    if(obstacle_check(nearest_point, point_1, bloat)):
        bool_value1 = True
    if(obstacle_check(nearest_point, point_2, bloat)):
        bool_value2 = True
    
    # return point with the minimum distance from the random node
    distance_pt1 = eucledian_distance(random_point, point_1)
    distance_pt2 = eucledian_distance(random_point, point_2)
    if(distance_pt1 < distance_pt2):
        return (bool_value1, point_1)
    else:
        return (bool_value2, point_2)

# check for obstacle spaces obstacle between the two points
def obstacle_check(coord_1, coord_2, total_bloat):    
    # get difference of the given two points along x and y direction
    x_difference = coord_2[0] - coord_1[0]
    y_difference = coord_2[1] - coord_1[1]    
    # points to check for obstacle along the line connecting the two points
    possible_points = []
    possible_points.append(coord_1)    
    # get value of maximum difference and then get the intermediate points along the line connecting them
    maximum_diff = max(abs(x_difference), abs(y_difference))    
    for index in range(1, int(np.abs(maximum_diff))):
        intermediate_point = (coord_1[0] + (index * x_difference / np.abs(maximum_diff)), coord_1[1] + (index * y_difference / np.abs(maximum_diff)))
        possible_points.append(intermediate_point)
    
    # check for obstacle
    for point in possible_points:
        if(if_obstacle((point[0], point[1], total_bloat)) == True):
            return True
    return False

def search_for_parent(new_point, all_explored_nodes, cost_to_come):
    #The stepfactor is a threshold for creating the neighbourhood
    stepFactor = 13
    # iterate through the explored points and get nodes within a certain radius 
    # to create the neighbourhood
    neighbourhood = []
    for index in range(0, len(all_explored_nodes)):
        dist = eucledian_distance(new_point, all_explored_nodes[index])
        if(dist < stepFactor):
            neighbourhood.append(all_explored_nodes[index])
    #Using the neighbourhood we look for possible points that can be parent nodes
    dist = cost_to_come[neighbourhood[0]]
    possible_parent = neighbourhood[0]
    for index in range(1, len(neighbourhood)):
        current_cost_distance = cost_to_come[neighbourhood[index]]
        if(current_cost_distance < dist):
            dist = current_cost_distance
            possible_parent = neighbourhood[index]
    return possible_parent, neighbourhood


# Informed RRT*algorithm
def informed_RRT_star(start_position, goal_position, clearance):
    explored_points = []
    c2c = {}
    back_path = {}
    threshold_for_goal = 8
    c2c[start_position] = 0
    explored_points.append(start_position)
    backtrack_nodes = []
    best_cost = float('inf')
    min_cost = eucledian_distance(start_position, goal_position)
    center_point = np.matrix([[(start_position[0] + goal_position[0]) / 2.0], [(start_position[1] + goal_position[1]) / 2.0], [0]])
    direction_matrix = np.dot(np.matrix([[(goal_position[0] - start_position[0]) / min_cost], [(goal_position[1] - start_position[1]) / min_cost], [0]]) , np.matrix([1.0, 0.0, 0.0]))
    U, S, VT = np.linalg.svd(direction_matrix, 1, 1)
    #getting the linear transformation of the direction_matrix M after its decomposition
    cost_matrix = np.dot(np.dot(U, np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(VT))])), VT)
    
    # running 15000 iterations to get the best optimized path
    for step in range(0, 15000):
        # creating a random node in the map from the ellipsoidal region
        (random_x, random_y) = generate_random_point(best_cost, min_cost, center_point, cost_matrix, clearance)
        random_point = (random_x, random_y)
        # Search the tree to get nearest node to random node
        (nearest_x, nearest_y) = generate_best_neighbour(random_x, random_y, explored_points)
        nearest_point = (nearest_x, nearest_y)
        if((nearest_point[0] == random_point[0]) or (nearest_point[1] == random_point[1])):
            continue

        # get new node between the nearest node in the tree and the randomly created node
        (flag, new_point) = generate_child(random_point, nearest_point, clearance)
        if(flag == True):
            continue 

        # get neighbourhood region for the newly created point and then search for a parent in 
        # that neighbourhood for the new point
        generated_parent, neighbourhood = search_for_parent(new_point, explored_points, c2c)
        #Updating the nearest point as the newly generated parent
        nearest_point = generated_parent
        
        # obstacle check for newly created points
        if(obstacle_check(nearest_point, new_point, clearance)):
            continue

        # Updating the graph
        explored_points.append(new_point)        
        back_path[new_point] = nearest_point
        c2c[new_point] = c2c[nearest_point] + eucledian_distance(nearest_point, new_point)
        
        # rewiring the graph with the closest and best parent node for the child node
        for index in range(0, len(neighbourhood)):
            distance_from_start = c2c[new_point] + eucledian_distance(new_point, neighbourhood[index])
            if(distance_from_start < c2c[neighbourhood[index]]):
                c2c[neighbourhood[index]] = distance_from_start
                back_path[neighbourhood[index]] = new_point
        
        # check the eucledian distance between the goal node and the newly created node and compare with the goal threshold
        dist_from_goal = eucledian_distance(new_point, goal_position)
        if(dist_from_goal <= threshold_for_goal):
            new_node_in_backtrack = new_point
            
            # Creating the backtrack path
            temp_path = []
            temp_len = c2c[new_node_in_backtrack]
            while(new_node_in_backtrack != start_position):
                temp_path.append(new_node_in_backtrack)
                new_node_in_backtrack = back_path[new_node_in_backtrack]
            temp_path.append(start_position)
            temp_path = list(reversed(temp_path))
            
            if(best_cost > temp_len):
                best_cost = temp_len
                backtrack_nodes = temp_path

    print(" Best Cost \n", best_cost)
    print("______________________________________________________________________")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Time taken to run the algorithm \n", execution_time)
    print("______________________________________________________________________")
    # return explored and backtrack states and the the entire path
    return (explored_points, backtrack_nodes, back_path)

#Main function

#Clearances to be maintained around the obstacles
clearance = 5
robot_radius = 10.5
total_bloat = (clearance + robot_radius)
#Taking inputs
# print("Taking inputs for the start point. The size of the arena is 300X200 cm.\n")
# start_point_x = input("Enter the x-coordinate of the start point. Enter values such that 0 <= X <= 300. \n")
# start_point_y = input("Enter the y-coordinate of the start point. Enter values such that 0 < Y < 200. \n")
start_point_x = 0
start_point_y = 75
start = (float(start_point_x), float(start_point_y))
while if_obstacle((start[0], start[1], total_bloat)):
    print("These coordinates lie inside the obstacle space. Please enter new values such that 0.2 <= X <= 0.5 and 0.2 <= Y <= 1.8 \n")
    start_point_x = input("Enter the x-coordinate of the start point \n")
    start_point_y = input("Enter the y-coordinate of the start point \n")
    start = (float(start_point_x), float(start_point_y))

# print("Taking inputs for the goal point. The size of the arena is 300X200 cm.\n")
# goal_point_x = input("Enter the x-coordinate of the goal point. Enter values such that 0 <= X <= 300. \n")
# goal_point_y = input("Enter the y-coordinate of the goal point. Enter values such that 0 < Y < 200 \n")
goal_point_x = 300
goal_point_y = -75
goal = (float(goal_point_x), float(goal_point_y))
while if_obstacle((goal[0], goal[1], total_bloat)):
    print("These coordinates lie inside the obstacle space. Please enter new values such that 4.7 <= X <= 5.8 and 0.2 <= Y <= 1.8 \n")
    goal_point_x = input("Enter the x-coordinate of the goal point \n")
    goal_point_y = input("Enter the y-coordinate of the goal point \n")
    goal = (float(goal_point_x), float(goal_point_y))

(explored_states, backtrack_states, path) = informed_RRT_star(start, goal, total_bloat)
print("Printing explored states")
print((explored_states))
print("______________________________________________________________________")
print("Number of explored states \n")
print(len(explored_states))
print("______________________________________________________________________")
print("Printing backtrack states")
print(backtrack_states)
print("______________________________________________________________________")

def visualize(backtrack_path, full_path, explored_path):
    tail_x = []
    tail_y = []
    head_x = []
    head_y = []
    x_coord = []
    y_coord = []

    start_node = backtrack_path[0]
    goal_node = backtrack_path[-1]
    plt.plot(start_node[0], start_node[1], marker="o", markersize=10, color="red")
    plt.plot(goal_node[0], goal_node[1], marker="o", markersize=10, color="red")
    plt.ion()

    # Plotting the explored states
    for index in range(1, len(explored_path)):
        parentNode = full_path[explored_path[index]]
        tail_x.append(parentNode[0])
        tail_y.append(parentNode[1])
        head_x.append((explored_path[index][0] - parentNode[0]))
        head_y.append((explored_path[index][1] - parentNode[1]))

        # tailX = parentNode[0]
        # tailY = parentNode[1]
        # headX = (explored_path[index][0] - parentNode[0])
        # headY = (explored_path[index][1] - parentNode[1])
        # plt.quiver(tailX, tailY, headX, headY, units='xy', scale=1, color='g', label='Explored region')
        # plt.draw()
        # plt.pause(0.01) 

    plt.quiver(np.array(tail_x), np.array(tail_y), np.array(head_x), np.array(head_y), units='xy', scale=1, color='g', label='Explored region')
   
    plt.title("Path Explored using Informed RRT*")
    plt.xlabel("x-coordinate (cm)")
    plt.ylabel("y-coordinate (cm)")
    
    # Plotting the shortest path
    for i in range(len(backtrack_path)):
        x_coord.append(backtrack_path[i][0])
        y_coord.append(backtrack_path[i][1])
        # plt.plot(np.array(x_coord), np.array(y_coord), color="blue", linewidth=5, label='Backtrack path')
        plt.plot(np.array(x_coord), np.array(y_coord), color="blue", linewidth=5)
        plt.draw()
        plt.pause(0.05)

    # for i, (x, y) in enumerate(backtrack_path[:-1]):
    #     n_x, n_y = path[i+1]
    #     plt.plot([x, n_x], [y, n_y], color="blue", linewidth=3)
        
    plt.legend()
    plt.ioff()
    plt.show()

map()
visualize(backtrack_states, path, explored_states)
path_array = np.array(backtrack_states)
np.savetxt('project_path.txt', path_array, delimiter='\t')



