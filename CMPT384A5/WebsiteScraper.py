import requests
from bs4 import BeautifulSoup


def read_in_website(url_pass):

    # Grab webwage
    rows = []
    games = []
    page = requests.get(url_pass)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="maincontent")

    # Grab each row for top 100 games
    for i in range(0,100):
        rows.append(results.findAll("tr", id = "row_"))

    #print(str(rows[0]).split("<"))    
    i = 0
    elems = []
    for row in rows:
        row_list = str(row).split('<')
        game = dict()
        
        # Add all elements assosiated with the game
        for elem in row_list:
            if ("collection_bggrating" in elem):
                elems += [float(elem.split("t")[-1][9:].split("\t")[0])]
            

        for elem in row_list:
            if ("img alt" in elem):
                i += 1
                game[i-9900] = [elem.split("\"")[1][12:],elems[3*(i-1):3*i]]  # Had a bug that added 9900 to i but now that is removed giving proper ratings
        
    return game




def toString(arg):
    for key in arg.keys():
        print(key, arg[key])

def main():
    URL = "https://boardgamegeek.com/browse/boardgame"
    toString(read_in_website(URL))

if __name__ == "__main__":
    main()