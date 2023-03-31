import json
import cython
import numpy

def readInJson(JSON_FLLE: str):
    with open(JSON_FLLE) as json_file:
        return json.load(json_file)
    
def nxnMatrixInit(n: int):
    return numpy.zeros((n,n))

def adjacingMatrix(jsonData: list):
    matrix = nxnMatrixInit(len(jsonData))
    for i in range(len(jsonData)):
        for j in range(len(jsonData)):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = jsonData[i].get("recommendations").get("fans_liked")
    return matrix

def test_adjacingMatrix(JSON_FLLE: str):
    matrix = adjacingMatrix(readInJson(JSON_FLLE))
    print(matrix)


def main():
    JSON_FLLE = "boardgames_40.json"
    print(readInJson(JSON_FLLE)[0])


if __name__ == "__main__":
    main()