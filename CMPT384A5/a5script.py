import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

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
    for i, id in enumerate(df.loc[:,'id']):  # Iterate over the IDs in the DataFrame
        id_to_idx[id] = i
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
                pass #print(f"ID: {endpoint} not in top 100, discarded")
    return adj_matrix

def svd(A: np.ndarray):
    """
    This function takes in an adjacency matrix A and returns the singular value
    decomposition of A.
    """
    u,s,v_t = np.linalg.svd(A)
    s = np.diag(s)
    return u,s,v_t

def reconstructSVD(u: np.ndarray, s: np.ndarray, v_t: np.ndarray):
    """
    This function takes in the singular value decomposition of a matrix A and
    returns the matrix A.
    """
    return np.dot(np.dot(u,s),v_t)

def reconstructSVDTruncated(u: np.ndarray, s: np.ndarray, v_t: np.ndarray, k: int):
    """
    This function takes in the singular value decomposition of a matrix A and
    returns the matrix A, truncated to the first k singular values.
    """
    return np.dot(np.dot(u[:,:k],s[:k,:k]),v_t[:k,:])

def reconstruct(A: np.ndarray, k: int):
    """
    This function takes in an adjacency matrix A and returns the matrix A,
    truncated to the first k singular values.
    """
    u,s,v_t = np.linalg.svd(A)
    s = np.diag(s)
    return np.dot(np.dot(u[:,:k],s[:k,:k]),v_t[:k,:])

def reconstructError(A: np.ndarray, k: int):
    """
    This function takes in an adjacency matrix A and returns the error between
    the original matrix A and the matrix A, truncated to the first k singular
    values.
    """
    return np.linalg.norm(A - reconstruct(A, k))

def main():
    PATH = "boardgames_40.json"
    adj_matrix = makeAdjMatrix(PATH)
    print(adj_matrix)

    u,s,v_t = svd(adj_matrix)
    print("\n", np.round(reconstructSVD(u,s,v_t), 5))
    

if __name__ == "__main__":
    main()