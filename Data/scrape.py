from bs4 import BeautifulSoup as bs
import requests
import pymongo
from pymongo import MongoClient
client = pymongo.MongoClient("localhost", 27017)
db = client.test
db.mood_collection

moods = ['happy', 'sad', 'anger']
url = "https://www.brainyquote.com/quotes/topics/topic_"

quotes = []

for mood in moods:
    tempURL = url + mood + "1.html?vm=l"
    page = requests.get(tempURL).text.encode('utf-8')
    soup = bs(page, 'html.parser')

    pgNumTags = soup.find('div', {'class' : 'pull-left bq-vcenter-parent'})
    lines = pgNumTags.findAll('li')
    numOfPages = int(lines[-2].text)

    fileName = mood + ".txt"

    with open(fileName, 'wb') as f:
        for i in range(1, numOfPages):
            finURL = "https://www.brainyquote.com/quotes/topics/topic_" + mood + str(i) + ".html?vm=l"
            page = requests.get(finURL).text.encode('utf-8')
            soup = bs(page, 'html.parser')
            impDiv = soup.find('div', {'class' : 'reflow_container'})
            reqDivs = impDiv.findAll('a', {'title' : 'view quote'})
            for reqDiv in reqDivs:
                a = reqDiv.text.encode('ASCII', 'ignore')
                f.write(a + "\t\t\t".encode('ascii'))
                db.mood_collection.insert(
                    {
                       "title" : mood,
                       "sentence" : reqDiv.text
                    })
