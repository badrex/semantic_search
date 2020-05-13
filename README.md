# Semantic Search Engine with Word Embeddings

The repository was developed and test in ```Python 3.6.9```. Therefore, it important to create a Python virtual environment with ```Python 3.6.9``` to avoid any inconsistencies. For the virtual environment, ```pipenv``` is preferred.   

Once the virtual environment has been set up, one should install the libraries provided in the  ```requirements.txt``` using python package management system  ```pip```, as follows

 ```>> pip install -r requirements.txt```

 Then, install the German language support for Spacy library as follows

 ```>> python -m spacy download de_core_news_md```

 The content of the compressed folder ```data_structures.zip``` should be compressed into a folder ```data_structures/```. Three ```.npy``` files  and one ```.index``` file in this folder.

 Now, the search engine should ready to run (in the virtual environment), as follows

```>> python query_search.py```

First, the program will load the objects from desk to memory. The program shall print the following print messages

```
Read page2text dictionary from desk ...
Read document frequency dictionary from desk ...
Compute IDF for each term in the collection ...
Read page2text dictionary from desk ...
Initialize Fliar embeddings ...
Initialize query processor object with Spacy and Flair ...
Read FAISS search index from desk ...
```


Then, the program should prompt an input field to receive an input from the user


``` Please enter query here: Fingerfarbe Handabdruck Fußabdruck```

Finally, the search engine should retrieve similar entries in the index, make a JSON object of the result, and print the retrieve pages with text snippets as follows


```
You entered: Fingerfarbe Handabdruck Fußabdruck
{
    "1": {
        "book": "61-1403",
        "page_num": "5",
        "snippet": "m1 Alles über mich BAUSTEINE KINDERGARTEN 3 | 2014So groß sind meine Hände schon! FINGERSPIEL:Meine Faust Sieh dir meine Faust mal an,(Faust zeigen)da wachsen ja fünf Blätter dran. (bis fünf zählen un"
    },
    "2": {
        "book": "64-0711",
        "page_num": "5",
        "snippet": "Bastelanleitung:Mit der Hand kann man nicht nur Handabdrücke machen! Bei der Herstellung der Babyfüße dienen die Handkanten als Stempel. Die Zehen werden mitdem kleinen Finger hinzugefügt. Material:■ "
    },
    "3": {
        "book": "20674",
        "page_num": "87",
        "snippet": "88 Spiele mit Alltagsmaterial Bunter Handabdruck Dieses bewusste Finger Farb Erleben ist einesinnvolle Ergänzung zu den Angeboten „Fühlen,matschen, kneten … (S. 81) und „Alle meine Fingerlein“ (S. 69)"
    },
    "4": {
        "book": "20647",
        "page_num": "88",
        "snippet": "Fingerspitze Fingerknöchel Handgelenk Fingernagel Daumen Handteller Zeigefinger Mittelfinger Ringfinger Kleiner Finger Fingerund Zehenbeweglichkeit Diese zehn kleinen, zappeligen Finger sindunsere Sup"
    },
    "5": {
        "book": "20894",
        "page_num": "42",
        "snippet": "Sich darstellen und aus drücken – Rollenspiele, Theater, Musik, Tanz und Kunst 43Handund Fußabdrücke Dieses Gemeinschaftswerk fördert das Zusammengehörigkeitsgefühl und die Kinder freuensich über das "
    },
    "6": {
        "book": "64-0951",
        "page_num": "9",
        "snippet": "ENGELBausteine Kindergarten Ideen für die Kleinsten 1Material: eine leere Toilettenpapierrolle, gelbe Fingerfarbe, goldenes Tonpapierrechteck, Tonpapier: Hautfarben,Weiß, roter Tonpapierkreis (Mund, D"
    },
    "7": {
        "book": "61-0904",
        "page_num": "6",
        "snippet": "Abdrücke aus Fingerfarbe Material:■ eine lange weiße Papierrolle■ Fingerfarbe■ Pinsel■ Filzstifte So wird’s gemacht:Die Kinder suchen sich ihre Lieblingsfarbe aus und bemalen sich alleineoder gegensei"
    },
    "8": {
        "book": "20775",
        "page_num": "87",
        "snippet": "Der Koch Alter : ab 6 Jahren Beim Sprechen folgende Bewegungen machen :● Bei „Wenn ich furchtbar hungrig bin“ einbekümmertes Gesicht machen und die Händeauf den Bauch legen.● Noten, deren Notenköpfe a"
    },
    "9": {
        "book": "20892",
        "page_num": "6",
        "snippet": "Der Herbst Die Hand des Kindes wird mit brauner Fingerfarbe bestrichen. Die bemalte Hand mit gespreizten Fingern in die Blattmitte drücken. Die Finger sind die Ästedes Baumes. Mit dem Pinsel eine Verl"
    },
    "10": {
        "book": "20853",
        "page_num": "39",
        "snippet": "Sommer Hoppla, jetzt komm’ ich!Kopiervorlagen Fußund Handabdrücke"
    }
}
```
