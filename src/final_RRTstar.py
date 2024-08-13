import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random
from datetime import datetime
plt.ion()
start_time = time.time()
print("Hi! Welcome to the simulation for RRT* algorithm")
print("______________________________________________________________________")
def map():

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim([0, 300])
    ax.set_ylim([-100, 100])   

    plt.title("Path explored using RRT*")
    plt.xlabel("x-coordinate(in cm)")
    plt.ylabel("y-coordinate(in cm)") 

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

#Heuristic eucledian distance
def eucledian_distance(coordinate_1, coordinate_2):
    distance = np.sqrt(((coordinate_2[0] - coordinate_1[0]) ** 2) + ((coordinate_2[1] - coordinate_1[1]) ** 2))
    return distance
# check for obstacle spaces obstacle between the two points
def obstacle_check(coord_1, coord_2, total_bloat):
    # get differences along x and y direction
    x_difference = coord_2[0] - coord_1[0]
    y_difference = coord_2[1] - coord_1[1]
    # points to check for obstacle
    possible_points = []
    possible_points.append(coord_1)
    # get value of maximum difference
    maximum_diff = max(abs(x_difference), abs(y_difference))    
    for index in range(1, int(np.abs(maximum_diff))):
        intermediate_point = (coord_1[0] + (index * x_difference / np.abs(maximum_diff)), coord_1[1] + (index * y_difference / np.abs(maximum_diff)))
        possible_points.append(intermediate_point)
    
    for point in possible_points:
        if(if_obstacle((point[0], point[1], total_bloat)) == True):
            return True
    return False

# Finding a random point in the map
def generate_random_point(width, height, min_height, total_bloat):
    random_point_x = round(random.uniform((0),(width)),2)
    random_point_y = round(random.uniform((min_height+total_bloat),(height-total_bloat)),2)

    return (random_point_x, random_point_y)
 # Finding the nearest neighbour in the graph for the random point
def generate_best_neighbour(current_x, current_y, explored_nodes):
    minimum_distance = float('inf')
    nearest_node = -1

    for node in explored_nodes:
        dist = eucledian_distance(node, (current_x,current_y))
        if(dist < minimum_distance):
            minimum_distance = dist
            nearest_node = node
    return nearest_node
# Finding a new child between random and nearest point
def generate_child(step, total_bloat, random_point, nearest_point):
    slope_of_line = (random_point[1] - nearest_point[1]) / (random_point[0] - nearest_point[0])
    factor = step * np.sqrt(1.0 / (1.0 + (slope_of_line ** 2)))
    
    point_1 = (round(nearest_point[0] + factor, 2), round(nearest_point[1] + (slope_of_line * factor), 2))
    point_2 = (round(nearest_point[0] - factor, 2), round(nearest_point[1] - (slope_of_line * factor), 2))
    bool_value1 = False
    bool_value2 = False
        
    if(obstacle_check(nearest_point, point_1,total_bloat)):
        bool_value1 = True
    if(obstacle_check(nearest_point, point_2, total_bloat)):
        bool_value2 = True
    
    # return point with minimum distance to random node
    distance_pt1 = eucledian_distance(random_point, point_1)
    distance_pt2 = eucledian_distance(random_point, point_2)
    if(distance_pt1 < distance_pt2):
        return (bool_value1, point_1)
    else:
        return (bool_value2, point_2)
    
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
    #Using the neighbourhood we look for possible points that can be parents
    dist = cost_to_come[neighbourhood[0]]
    possible_parent = neighbourhood[0]
    for index in range(1, len(neighbourhood)):
        current_cost_distance = cost_to_come[neighbourhood[index]]
        if(current_cost_distance < dist):
            dist = current_cost_distance
            possible_parent = neighbourhood[index]
    return possible_parent, neighbourhood

#RRTstar algorithm
def rrt_star(start, goal, back_path, width, max_height, min_height, t_total_bloat):
    c2c={}
    explored_points = []
    threshold_for_goal = 8
    stepsize = 4
    c2c[start] = 0
    explored_points.append(start)
    b_node = None
     # running 15000 iterations to get the best optimized path
    for step in range(0,15000):
        st = datetime.now()
        # creating a random node in the map
        (random_x,random_y) = generate_random_point(width, max_height, min_height, t_total_bloat) #Generating random node
        random_point = (random_x, random_y)
        # Search the tree to get nearest node to random node
        (nearest_x, nearest_y) = generate_best_neighbour(random_x,random_y,explored_points)
        nearest_point = (nearest_x,nearest_y)

        if((nearest_point[0] == random_point[0]) or (nearest_point[1] == random_point[1])):
            continue
        # get new node between the nearest node in the tree and the randomly created node
        (flag, new_point) = generate_child(stepsize, t_total_bloat, random_point, nearest_point)
        if(flag == True):
            #print("Detected")
            continue

        # get neighbourhood region for the newly created point and then search for a parent in 
        # that neighbourhood for the new point
        generated_parent, neighbour = search_for_parent(new_point, explored_points, c2c)
        #Updating the nearest point as the newly generated parent
        nearest_point = generated_parent
        # obstacle check for newly created points
        if(obstacle_check(nearest_point, new_point, t_total_bloat)):
            print("Generating...")
            continue
        # Updating the graph
        explored_points.append(new_point)
        back_path[new_point] = nearest_point
        c2c[new_point] = c2c[nearest_point] + eucledian_distance(nearest_point, new_point)
        # rewiring the graph with the closest and best parent node for the child node
        for item in range(0, len(neighbour)):
            dist_start = c2c[new_point] + eucledian_distance(new_point, neighbour[item])
            if(dist_start < c2c[neighbour[item]]):
                c2c[neighbour[item]] = dist_start
                back_path[neighbour[item]] = new_point
        # check the eucledian distance between the goal node and the newly created node and compare with the goal threshold
        dist_from_goal = eucledian_distance(new_point, goal)
        if(dist_from_goal <= threshold_for_goal):
            b_node = new_point
            # break

    if(b_node == None):
        return(explored_points, [])
    # Creating the backtrack path
    backstates = []
    length = c2c[b_node]
    while(b_node != start):
        backstates.append(b_node)
        b_node = back_path[b_node]
    backstates.append(start)
    backstates = list(reversed(backstates))

    print(" Best Cost \n", length)
    print("______________________________________________________________________")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Time taken to run the algorithm \n", execution_time)
    print("______________________________________________________________________")
    # return explored and backtrack states
    return(explored_points,backstates)

def visualize(explored, backtracked, nodes):
    tail_x=[]
    tail_y=[]
    head_x=[]
    head_y=[]

    # call map function to generate the map
    map()
    # get the current axis to add the quiver plot on top of the map
    ax = plt.gca()
    # loop over the explored states
    for index in range(1, len(explored)):
        parentNode = nodes[explored[index]]
        tail_x.append(parentNode[0])
        tail_y.append(parentNode[1])
        head_x.append((explored[index][0] - parentNode[0]))
        head_y.append((explored[index][1] - parentNode[1]))

    # plot the explored states
    ax.quiver(np.array(tail_x), np.array(tail_y), np.array(head_x), np.array(head_y), units='xy', scale=1, color='g', label='Explored region')
    plt.legend()
    plt.show()
    
#Plotting the backtrack path
def shortest_path(path):
    start_node = path[0]
    goal_node = path[-1]
    plt.plot(start_node[0], start_node[1], marker="o", markersize=10, color="red")
    plt.plot(goal_node[0], goal_node[1], marker="o", markersize=10, color="red")

    shortest_path_added = False
    for i, (x, y) in enumerate(path[:-1]):
        n_x, n_y = path[i+1]
        if not shortest_path_added:
            plt.plot([x, n_x], [y, n_y], color="blue", linewidth=5, label="backtracked path")
            shortest_path_added = True
        else:
            plt.plot([x, n_x], [y, n_y], color="blue", linewidth=5)

    plt.legend()
    plt.show(block=False)
    plt.pause(5)
    # plt.close()
       
clear = 5
radius = 10.5
t_total_bloatance = clear + radius
max_height = 100
min_height = -100
width = 300
back_path = {}

check = True
while check:
    # start_x = int(input("Enter the x coordinate of the start point "))
    start_x = 0
    # start_y = int(input("Enter the y coordinate of the start point "))
    start_y = 75
    # goal_x = int(input("Enter the x coordinate of the goal point "))
    goal_x = 300
    # goal_y = int(input("Enter the y coordinate of the goal point "))
    goal_y = -75
    obs_state = if_obstacle((start_x,start_y,t_total_bloatance))
    obs_state_g = if_obstacle((goal_x, goal_y, t_total_bloatance))
    if start_x < 0 or start_x > width or start_y > max_height or start_y < min_height or goal_x < 0 or goal_x > width or goal_y > max_height or goal_y < min_height:
        print("Entered points are outside the arena; Try again")
    elif obs_state == True:
        print("Entered points are in obstacle space")
    elif obs_state_g == True:
        print("Entered points are in obstacle space")
    else:
        check = False

start = (start_x, start_y)
goal = (goal_x, goal_y)

check = True
while check:
    obs_state = if_obstacle((start_x,start_y,t_total_bloatance))
    obs_state_g = if_obstacle((goal_x, goal_y, t_total_bloatance))
    if start_x < 0 or start_x > width or start_y > max_height or start_y < min_height or goal_x < 0 or goal_x > width or goal_y > max_height or goal_y < min_height:
        print("Entered points are outside the arena; Try again")
    elif obs_state == True:
        print("Entered points are in obstacle space")
    elif obs_state_g == True:
        print("Entered points are in obstacle space")
    else:
        check = False

explored, backtracked = rrt_star(start, goal, back_path, width, max_height, min_height, t_total_bloatance)
print("Printing no explored states \n", len(explored))
print("______________________________________________________________________")
print("Printing no backtrack states \n", len(backtracked))
print("______________________________________________________________________")
visualize(explored, backtracked, back_path)
shortest_path(backtracked)



