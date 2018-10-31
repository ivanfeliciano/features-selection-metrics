import json
import sys

class Evaluator(object):

    def __init__(self, index_json_path='inv_index.json', info_class_json_path='docs_len.json'):
        with open(index_json_path) as json_file:
            self.index_reader = json.load(json_file)
        with open(info_class_json_path) as json_file:
            self.class_info_reader = json.load(json_file)

    def get_class_size(self, class_id):
        if class_id in self.class_info_reader:
            return self.class_info_reader[class_id]['docsinClass']

    def true_positives(self, word, positive_class_id_str):
        if word in self.index_reader:
            if positive_class_id_str in self.index_reader[word]:
                return self.index_reader[word][positive_class_id_str].get('class_docs_having_it', 0)
            return 0
    def true_positives_rate(self, word, positive_class_id_str):
        tp = float(self.true_positives(word, positive_class_id_str))
        category_size = float(self.get_class_size(positive_class_id_str))
        return tp / category_size
    def false_positives(self, word, positive_class_id_str):
        fp = 0
        if word in self.index_reader:
            for negative_class_id in self.index_reader[word]:
                if negative_class_id.isdigit():
                    if negative_class_id == positive_class_id_str:
                        continue
                    fp += self.index_reader[word][negative_class_id].get('class_docs_having_it', 0)
        return fp
    def false_positives_rate(self, word, positive_class_id_str):
        fp = float(self.false_positives(word, positive_class_id_str))
        negatives_classes_size = 0.0
        for negative_class_id in self.class_info_reader:
            if negative_class_id.isdigit():
                if negative_class_id == positive_class_id_str:
                    continue
                negatives_classes_size += float(self.get_class_size(negative_class_id))
        return fp / negatives_classes_size

def main():
    test = Evaluator()
    print("TPR = {}".format(test.true_positives_rate('categori', '3')))
    print("FPR = {}".format(test.false_positives_rate('categori', '3')))
if __name__ == '__main__':
    main()
