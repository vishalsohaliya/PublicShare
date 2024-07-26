#Depth First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}

dfs(graph, 'A')

####################

#Min-Max Algorithm
import math

def minimax (curDepth, nodeIndex,
			maxTurn, scores, 
			targetDepth):

	# base case : targetDepth reached
	if (curDepth == targetDepth): 
		return scores[nodeIndex]
	
	if (maxTurn):
		return max(minimax(curDepth + 1, nodeIndex * 2, 
					False, scores, targetDepth), 
				minimax(curDepth + 1, nodeIndex * 2 + 1, 
					False, scores, targetDepth))
	
	else:
		return min(minimax(curDepth + 1, nodeIndex * 2, 
					True, scores, targetDepth), 
				minimax(curDepth + 1, nodeIndex * 2 + 1, 
					True, scores, targetDepth))
	
# Driver code
scores = [3, 5, 6, 9, 1, 2, 0, -1]

treeDepth = math.log(len(scores), 2)

print("The optimal value is : ", end = "")
print(minimax(0, 0, True, scores, treeDepth))


####################

#Backtracking approach to solve N Queen’s problem

def print_solution(board):
    for row in board:
        print(" ".join(row))
    print()

def is_safe(board, row, col):
    for i in range(col):
        if board[row][i] == 'Q':
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False
    for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False
    return True

def solve_nqueens_util(board, col):
    if col >= len(board):
        print_solution(board)
        return True

    res = False
    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 'Q'
            res = solve_nqueens_util(board, col + 1) or res
            board[i][col] = '.'
    return res

def solve_nqueens(n):
    board = [['.' for _ in range(n)] for _ in range(n)]
    if not solve_nqueens_util(board, 0):
        print("No solution exists")
    return

# Example usage
n = 4
solve_nqueens(n)


####################

#Breadth First Search Algorithm

from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

bfs(graph, 'A')

####################
#K-nearest Neighbor algorithm

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the KNN function
def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            distance = np.sqrt(np.sum((X_train[i] - test_point) ** 2))
            distances.append((distance, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(k)]
        most_common = Counter(neighbors).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

# Set the number of neighbors
k = 3

# Get predictions
y_pred = knn(X_train, y_train, X_test, k)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


for i in range(len(X_test)):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
    
    
####################
#A* Algorithm.

from heapq import heappop, heappush

# Define the heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define the A* algorithm
def a_star(graph, start, goal):
    open_list = []
    heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_list:
        current = heappop(open_list)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for neighbor in graph.get(current, []):
            tentative_g_score = g_score[current] + 1  # Assumes each move has a cost of 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Define the graph as a dictionary where each key is a node and the value is a list of neighbors
graph = {
    (0, 0): [(0, 1), (1, 0)],
    (0, 1): [(0, 0), (1, 1)],
    (1, 0): [(0, 0), (1, 1)],
    (1, 1): [(0, 1), (1, 0), (1, 2)],
    (1, 2): [(1, 1), (2, 2)],
    (2, 2): [(1, 2)]
}

# Define the start and goal nodes
start = (0, 0)
goal = (2, 2)

# Find the shortest path using A* algorithm
path = a_star(graph, start, goal)

# Print the result
if path:
    print("Path found:", path)
else:
    print("No path found")



####################
#logistic regression

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# For simplicity, we'll use only the first two features and two classes
X = X[y != 2]
y = y[y != 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


############################
#Naive Bayes

from sklearn.datasets import load_iris
iris = load_iris()
  
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target
  
# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
  
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred = gnb.predict(X_test)
  
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

############################
#KNN- classification

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the KNN model
k = 5  # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=wine.target_names)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


############################
#Support Vector Machines

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear')  # Using a linear kernel
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=cancer.target_names)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


############################
# K-Means clustering algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Define the number of clusters to test
k_values = [2, 3, 4, 5]

# Plot results for different values of k
fig, axes = plt.subplots(1, len(k_values), figsize=(20, 5), sharex=True, sharey=True)

for ax, k in zip(axes, k_values):
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    if k > 1:  # Silhouette score requires at least 2 clusters
        silhouette_avg = silhouette_score(X, y_kmeans)
        print(f"Silhouette Score for k={k}: {silhouette_avg:.2f}")
    
    # Plot clusters
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o')
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
    ax.set_title(f'K={k}')

plt.suptitle('K-Means Clustering on Iris Dataset')
plt.show()


############################
#Water Jug problem

from collections import deque

def water_jug_problem(capacity1, capacity2, target):
    # BFS queue
    queue = deque()
    # Set to track visited states
    visited = set()
    
    # Initial state: both jugs are empty
    queue.append((0, 0, []))
    visited.add((0, 0))
    
    while queue:
        jug1, jug2, path = queue.popleft()
        
        # Check if we have reached the target
        if jug1 == target or jug2 == target:
            return path + [(jug1, jug2)]
        
        # Possible states
        possible_states = []
        
        # Fill jug1
        possible_states.append((capacity1, jug2))
        
        # Fill jug2
        possible_states.append((jug1, capacity2))
        
        # Empty jug1
        possible_states.append((0, jug2))
        
        # Empty jug2
        possible_states.append((jug1, 0))
        
        # Pour jug1 into jug2
        transfer = min(jug1, capacity2 - jug2)
        possible_states.append((jug1 - transfer, jug2 + transfer))
        
        # Pour jug2 into jug1
        transfer = min(jug2, capacity1 - jug1)
        possible_states.append((jug1 + transfer, jug2 - transfer))
        
        # Explore all possible states
        for state in possible_states:
            if state not in visited:
                visited.add(state)
                queue.append((state[0], state[1], path + [(jug1, jug2)]))
    
    return None  # No solution found

# Example usage
capacity1 = 4  # Capacity of the first jug
capacity2 = 3  # Capacity of the second jug
target = 2     # Target amount of water to measure

solution = water_jug_problem(capacity1, capacity2, target)

if solution:
    print("Solution steps:")
    for step in solution:
        print(step)
else:
    print("No solution exists.")


############################
#Apriori Algorithm

from itertools import combinations
from collections import defaultdict

def apriori(transactions, min_support):
    def get_frequent_itemsets(itemsets, min_support):
        """Generate frequent itemsets from the given itemsets with a support threshold."""
        itemset_count = defaultdict(int)
        for transaction in transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    itemset_count[itemset] += 1
        return {itemset: count for itemset, count in itemset_count.items() if count >= min_support}
    
    def generate_candidates(itemsets, length):
        """Generate candidate itemsets of a given length from the frequent itemsets."""
        return set(frozenset(combo) for itemset in itemsets for combo in combinations(itemset, length))
    
    # Convert transactions to a list of sets for easier processing
    transactions = [set(transaction) for transaction in transactions]
    
    # Initialize
    all_frequent_itemsets = []
    k = 1
    itemsets = set(frozenset([item]) for transaction in transactions for item in transaction)
    
    # Generate frequent itemsets
    while itemsets:
        frequent_itemsets = get_frequent_itemsets(itemsets, min_support)
        if not frequent_itemsets:
            break
        all_frequent_itemsets.append(frequent_itemsets)
        itemsets = generate_candidates(frequent_itemsets.keys(), k + 1)
        k += 1
    
    return all_frequent_itemsets

# Example usage
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter', 'milk', 'egg'],
    ['bread', 'butter', 'egg'],
]

min_support = 2
frequent_itemsets = apriori(transactions, min_support)

# Print the frequent itemsets
print("Frequent Itemsets:")
for level, itemsets in enumerate(frequent_itemsets, start=1):
    print(f"Level {level}:")
    for itemset, count in itemsets.items():
        print(f"  {set(itemset)}: {count}")
