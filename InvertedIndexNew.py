# encoding : utf-8
import nltk
import os
import json
import csv
# import collections

from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer


class InvIndex(object):
    def __init__(self):
        self.indexClass = defaultdict(dict)
        self.infoClass = defaultdict(dict)
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords += [",", "(", ")", "[", "]", "{", "}", "#", "@", "!", ":", ";", ".", "?"]
        self.stemmer = EnglishStemmer()
        self.w_frecuency = 0
        self.classID = 0
        self.docID = 0
        # diccionario donde guardo los ids de las clases
        self.classes_ids = {"rec.motorcycles" : 0, "comp.sys.mac.hardware" : 1, "talk.politics.misc" : 2, "soc.religion.christian" : 3, "comp.graphics" : 4, "sci.med" : 5, "talk.religion.misc" : 6, "comp.windows.x" : 7, "comp.sys.ibm.pc.hardware" : 8, "talk.politics.guns" : 9, "alt.atheism" : 10, "comp.os.ms-windows.misc" : 11, "sci.crypt" : 12, "sci.space" : 13, "misc.forsale" : 14, "rec.sport.hockey" : 15, "rec.sport.baseball" : 16, "sci.electronics" : 17, "rec.autos" : 18, "talk.politics.mideast" : 19, }


    def save_to_json(self):
        with open('inv_index.json', 'w') as outfile:
            json.dump(self.indexClass, outfile)
        with open('docs_len.json', 'w') as outfile:
            json.dump(self.infoClass, outfile)

    def add_document(self, class_name, doc_content, doc_id):
        """
        Agrega un documento al index
        """
        doc_class_id = self.classes_ids[class_name]
        # Si no he guardado la la clase en mi JSON de info de la clase
        if not doc_class_id in self.infoClass:
            self.infoClass[doc_class_id]['docsinClass'] = 0
            self.infoClass[doc_class_id]['wordsinClass'] = 0
            self.infoClass[doc_class_id]['c_name'] = class_name
            self.infoClass[doc_class_id]['docsinClass'] = 0
        # Por cada palabra lower en el documento
        for token in [t.lower() for t in nltk.word_tokenize(doc_content)]:  # Convierte a minusculas
            self.infoClass[doc_class_id]['wordsinClass'] += 1

            # Si no he guardado la palabra en el index inicializo su frecuencia a 0
            if not token in self.indexClass:
                self.indexClass[token]['w_frecuency'] = 0

            # Si es la primera vez que aparece la palbrra en la clase
            # Agrego la llave del id de la clase y su valor es un diccionario
            # Inicializo la frecuencia en la clase en 0
            # Inicializo el # de docs de la clase que tienen la palabra
            if not doc_class_id in self.indexClass[token]:
                self.indexClass[token][doc_class_id] = {}
                self.indexClass[token][doc_class_id]['class_name'] = class_name
                self.indexClass[token][doc_class_id]['c_frecuency'] = 0                
                self.indexClass[token][doc_class_id]['docs_having_it'] = 0

            # Si es la primera vez que aparece el aparece la palabra
            # en el documento
            # inicializo la un diccionario con informacion del documento

            if not doc_id in self.indexClass[token][doc_class_id]:
                self.indexClass[token][doc_class_id][doc_id] = {}
                self.indexClass[token][doc_class_id][doc_id]['doc_frequency'] = 0
                self.indexClass[token][doc_class_id]['docs_having_it'] += 1

            # aumento la frecuencia de la palabra en el documento
            self.indexClass[token][doc_class_id]['c_frecuency'] += 1
            # aumento la frecuencia de la palabra por clase
            self.indexClass[token][doc_class_id][doc_id]['doc_frequency'] += 1
            # aumento la frecuencia global de la palabra
            self.indexClass[token]['w_frecuency'] += 1

        self.infoClass[doc_class_id]['docsinClass'] += 1

def main():
    myInvIndex = InvIndex()
    csvfile_path = '20ng-train-stemmed.txt'
    number_of_docs = 2000
    with open(csvfile_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_reader = csv.reader(csvfile, delimiter=';')
        doc_id = 0
        # Por cada fila en el archivo txt
        for row in csv_reader:
            if doc_id == number_of_docs:
                break
            class_name = row[0]
            doc_content = row[1]
            # necesito el contador del doc_id
            myInvIndex.add_document(class_name, doc_content, doc_id)
            doc_id += 1
    myInvIndex.save_to_json()


if __name__ == '__main__':
    main()
