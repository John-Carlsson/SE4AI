import requests
from pathlib import Path

filename = 'fer2013.csv'

my_file = Path(filename)

#Check if file exists and if not download it 
if my_file.is_file() == False:
    url = 'https://seafile.rlp.net/f/81cdc2fc291c44d38275/?dl=1'
    r = requests.get(url)

    filename = 'fer2013.csv'

    with open(filename, 'wb') as f:
            f.write(r.content)