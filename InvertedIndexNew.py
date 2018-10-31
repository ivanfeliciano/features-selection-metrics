# encoding : utf-8
import nltk
import os
import json
import csv

from math import log
# import collections

from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer

TOTAL_NUMBER_OF_DOCUMENTS = 0

class InvIndex(object):
    def __init__(self):
        self.indexClass = defaultdict(dict)
        self.infoClass = defaultdict(dict)
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords += [",", "(", ")", "[", "]", "{", "}", "#", "@", "!", ":", ";", ".", "?"]
        self.stemmer = EnglishStemmer()
        self.w_frecuency = 0
        # self.classID = 0
        # self.docID = 0
        # diccionario donde guardo los ids de las clases
        self.classes_ids = {"alt.atheism": 0, "comp.graphics": 1, "comp.os.ms-windows.misc": 2, "comp.sys.ibm.pc.hardware": 3, "comp.sys.mac.hardware": 4, "comp.windows.x": 5, "misc.forsale": 6, "rec.autos": 7, "rec.motorcycles": 8,
                            "rec.sport.baseball": 9, "rec.sport.hockey": 10, "sci.crypt": 11, "sci.electronics": 12, "sci.med": 13, "sci.space": 14, "soc.religion.christian": 15, "talk.politics.guns": 16, "talk.politics.mideast": 17, "talk.politics.misc": 18, "talk.religion.misc": 19, }


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
                self.indexClass[token]['docs_containing_it'] = 0

            # Si es la primera vez que aparece la palbrra en la clase
            # Agrego la llave del id de la clase y su valor es un diccionario
            # Inicializo la frecuencia en la clase en 0
            # Inicializo el # de docs de la clase que tienen la palabra
            if not doc_class_id in self.indexClass[token]:
                self.indexClass[token][doc_class_id] = {}
                self.indexClass[token][doc_class_id]['class_name'] = class_name
                self.indexClass[token][doc_class_id]['c_frecuency'] = 0                
                self.indexClass[token][doc_class_id]['class_docs_having_it'] = 0

            # Si es la primera vez que aparece el aparece la palabra
            # en el documento
            # inicializo la un diccionario con informacion del documento

            if not doc_id in self.indexClass[token][doc_class_id]:
                self.indexClass[token][doc_class_id][doc_id] = {}
                self.indexClass[token][doc_class_id][doc_id]['doc_frequency'] = 0
                self.indexClass[token][doc_class_id]['class_docs_having_it'] += 1
                self.indexClass[token]['docs_containing_it'] += 1

            # aumento la frecuencia de la palabra en el documento
            self.indexClass[token][doc_class_id]['c_frecuency'] += 1
            # aumento la frecuencia de la palabra por clase
            self.indexClass[token][doc_class_id][doc_id]['doc_frequency'] += 1
            # aumento la frecuencia global de la palabra
            self.indexClass[token]['w_frecuency'] += 1

        self.infoClass[doc_class_id]['docsinClass'] += 1
    def compute_w_ij(self):
        """
        Para cada palabra en el diccionario, en cada documento en el que aparece calcula

        w_ij = tf_ij * log(N / dfi)
        tf_ij =  # de ocurrecias de la palabra i en el documento j
        df_i =  # de documentos que contienen i
        N = # total de documentos
        """

        for word in self.indexClass:
            df_i = self.indexClass[word].get('docs_containing_it')
            idf = log(float(TOTAL_NUMBER_OF_DOCUMENTS) / float(df_i), 2)
            for class_id in self.indexClass[word]:
                # si no es el id de una clase se salta a la siguiente iteracion
                if not isinstance(class_id, int):
                    continue
                for doc_id in self.indexClass[word][class_id]:
                    if not isinstance(doc_id, int):
                        continue
                    tf_ij = self.indexClass[word][class_id][doc_id].get(
                        "doc_frequency")
                    w_ij = tf_ij * idf
                    self.indexClass[word][class_id][doc_id]['wij'] = w_ij

                

def main():
    global TOTAL_NUMBER_OF_DOCUMENTS
    myInvIndex = InvIndex()
    csvfile_path = '20ng-train-stemmed.txt'
    with open(csvfile_path, 'r', encoding='utf-8') as csvfile:
        TOTAL_NUMBER_OF_DOCUMENTS =  len(csvfile.readlines())
    with open(csvfile_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_reader = csv.reader(csvfile, delimiter=';')
        doc_id = 0
        # Por cada fila en el archivo txt
        for row in csv_reader:
            class_name = row[0]
            doc_content = row[1]
            # necesito el contador del doc_id
            myInvIndex.add_document(class_name, doc_content, doc_id)
            doc_id += 1
    myInvIndex.compute_w_ij()
    myInvIndex.save_to_json()


if __name__ == '__main__':
    main()
