import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def makeAdjMatrix(PATH: str = "boardgames_100.json"):
    """
    This function takes in a path to a JSON file containing the top 100 board games
    and returns an adjacency matrix representing the graph of the board games.
    The adjacency matrix is a nxn matrix where the entry at row i and column j
    is 1 if there is a directed edge from game i to game j, and 0 otherwise.
    The games are ordered by their ID number, which is the same as the index of the
    game in the JSON file.
    """
    num_nodes = int(PATH[11:-5])  # Get the number of nodes from the file name
    df = pd.read_json(PATH)  # Read the JSON file into a DataFrame
    id_to_idx = {}  # Dictionary mapping ID to index
    inx_to_id = {}
    for i, id in enumerate(df.loc[:,'id']):  # Iterate over the IDs in the DataFrame
        id_to_idx[id] = i
        inx_to_id[i] = id
    df['recommendations'] = df['recommendations'].apply(lambda x: x['fans_liked'])  # Get the list of recommendations
    adj_matrix = np.zeros((num_nodes,num_nodes))  # Initialize the adjacency matrix
    for index in range(num_nodes):  # Iterate over the rows of the DataFrame
        #print(f"{index} : {df.iloc[index]['recommendations']}")
        # Construct the graph
        v = id_to_idx[df.iloc[index]['id']] # Source node
        for endpoint in df.iloc[index]['recommendations']:  # Iterate over the endpoints
            try:
                w = id_to_idx[endpoint] # Destination node
                # Construct a directed edge (v, w)
                adj_matrix[v][w] = 1
            except:
                pass  #print(f"ID: {endpoint} not in top 100, discarded")
    return adj_matrix, id_to_idx, inx_to_id

def svd(A: np.ndarray):
    """
    This function takes in an adjacency matrix A and returns the `si`ngular value
    decomposition of A.
    """
    u,s,v_t = np.linalg.svd(A)  # Compute the SVD
    s = np.diag(s)  # Convert s to a diagonal matrix
    return u,s,v_t


def reconstructSVD(u: np.ndarray, s: np.ndarray, v_t: np.ndarray, k: int = -1):
    """
    This function takes in the singular value decomposition of a matrix A and
    returns the matrix A, truncated to the first k singular values.
    If k is not specified will fully reconstruct
    """
    if k == -1:
        return np.dot(np.dot(u,s),v_t)  # Reconstruct the matrix
    return np.dot(np.dot(u[:,:k],s[:k,:k]),v_t[:k,:])  # Reconstruct the matrix with k singular values

def reconstructPartial(A: np.ndarray, k: int = -1):
    """
    This function takes in an adjacency matrix A and returns the matrix A,
    truncated to the first k singular values.
    """
    u,s,v_t = svd(A) # Compute the SVD
    if k == -1: # Reconstruct the matrix
        return np.dot(np.dot(u,s),v_t)
    return np.dot(np.dot(u[:,:k],s[:k,:k]),v_t[:k,:])  # Reconstruct the matrix with k singular values

def reconstructError(A: np.ndarray, k: int):
    """
    This function takes in an adjacency matrix A and returns the error between
    the original matrix A and the matrix A, truncated to the first k singular
    values.
    """
    return np.linalg.norm(A - reconstructPartial(A, k))  # Compute the error

def projectionOnVector(v: np.ndarray, w: np.ndarray):
    """
    This function takes in two vectors v and w and returns the projection of v
    onto w.
    """
    return np.dot(v,w)/np.dot(w,w)*w  # Compute the projection

def projectMatrixOnVectors(A: np.ndarray, w: np.ndarray):
    """
    This function takes in a matrix A and a eigen-vector matrix w and returns the projection
    of A onto w.
    """
    B = np.zeros(A.shape)  # Initialize the projection matrix
    for i in range(len(A[0])):
        B[:,i] = projectionOnVector(A[:,i],w[:,i])
    return B  # Return the projection matrix

# Test functions
def test_projectionOnVector():
    """
    This function tests the projectionOnVector function.
    """
    v = np.array([1,2,3])
    w = np.array([1,1,1])
    print(projectionOnVector(v,w))

def test_reconstructError():
    """
    This function tests the reconstructError function.
    """
    PATH = "boardgames_40.json"
    adj_matrix,_,_= makeAdjMatrix(PATH)
    print(reconstructError(adj_matrix, 3))

def test_reconstructSVD():
    """
    This function tests the reconstructSVD function.
    """
    PATH = "boardgames_40.json"
    adj_matrix,_,_ = makeAdjMatrix(PATH)
    u,s,v_t = svd(adj_matrix)
    print(reconstructSVD(u,s,v_t,3))

def test_reconstructPartial():
    """
    This function tests the reconstructPartial function.
    """
    PATH = "boardgames_40.json"
    adj_matrix,_,_ = makeAdjMatrix(PATH)
    print(reconstructPartial(adj_matrix,3))

def test_svd():
    """
    This function tests the svd function.
    """
    PATH = "boardgames_40.json"
    adj_matrix,_,_ = makeAdjMatrix(PATH)
    print(svd(adj_matrix))

def test_makeAdjMatrix():
    """
    This function tests the makeAdjMatrix function.
    """
    PATH = "boardgames_40.json"
    print(makeAdjMatrix(PATH))

def getColor(i: int, PATH: str):
  df = pd.read_json(PATH)
  colors = [
    "pink", "orangered", "darksalmon", "saddlebrown", "darkorange",
    "greenyellow", "green", "turqoise", "teal", "cyan", "skyblue", "grey",
    "blue", "violet"
  ]

  def getTypeCount():
    type_counts = {}
    for _types in df.types:
      _types = _types["categories"]
      for _type in _types:
        _type = _type["name"]
        if _type not in list(type_counts.keys()):
          type_counts[_type] = 1
        else:
          type_counts[_type] += 1

    type_counts = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    temp = {}

    for t in type_counts:
      temp[t[0]] = t[1]
    type_counts = temp
    return type_counts

  type_counts = getTypeCount()

  def get_most_common_cat(variables):
    _type = variables[0]
    for var in variables:
      if type_counts[_type] < type_counts[var]:
        _type = var
    return _type

  def get_all_cat(i: int):
    var = []
    for j in range(len(df.types[i]["categories"])):
      var.append(df.types[i]["categories"][j]["name"])
    return var

  cats = list(type_counts.keys())
  var = get_most_common_cat(get_all_cat(i))
  try:
    mark_color = colors[cats.index(var)]
  except:
    mark_color = "black"
  return mark_color, var


def draw_plot(PATH):
    colors = [
        "pink", "orangered", "darksalmon", "saddlebrown", "darkorange",
        "greenyellow", "green", "turqoise", "teal", "cyan", "skyblue", "grey",
        "blue", "violet"
    ]
    
    adj_matrix, id_to_idx, inx_to_id = makeAdjMatrix(PATH)
    u,_,_ = svd(adj_matrix)
    fig = plt.figure(1)
    a = plt.axes(projection='3d')
    #az = plt.axes(projection='2d') 
    x = u[:,0]
    y = u[:,1]
    z = u[:,2]
    _category_color = {}
   
    for i, point in enumerate(adj_matrix):
        # Represent the datapoint in our new basis consisting of x,y,z
        # AKA, the 3 most significant singular vectors of the data matrix
        xs = np.dot(x, point)
        ys = np.dot(y, point)
        zs = np.dot(z, point)
        _color, _category = getColor(i,PATH)
        _category_color[_color] = _category
        a.scatter(xs, ys, zs, color=_color)
    plt.show()
    print(_category_color)
    


def main():
    PATH = "boardgames_100.json"
    draw_plot(PATH)

# Run the main function
if __name__ == "__main__":
    main()