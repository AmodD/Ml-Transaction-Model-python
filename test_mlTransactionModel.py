import csv
import pandas as pd
import string
import unittest
import mlTransactionModel
read_csv_variable=pd.read_csv
data_csv_path = "Sample Transactions with Target Variable.csv"
data_csv_path1 = "Sample Transactions-Test Data-Assignment-2.csv"

data = read_csv_variable(data_csv_path, dtype=object)
length = len(data)
data2 = pd.read_csv("test.csv")
length1 = len(list(data2))


def test_csv_format():
    try:
        with open(data_csv_path, newline='') as csvfile:
            start = csvfile.read(4096)

            # isprintable does not allow newlines, printable does not allow umlauts...
            if not all([c in string.printable or c.isprintable() for c in start]):
                return False
            dialect = csv.Sniffer().sniff(start)
            return True
    except csv.Error:
        print("Exception")
        # Could not get a csv dialect -> probably not a csv.
        return False


class MyTestCase(unittest.TestCase):
    def test_csv(self):
        self.assertEqual(mlTransactionModel.importFile("Sample Transactions-test Data.csv"),
                         data_csv_path)

    def test_csv_not_found(self):
        with self.assertRaises(FileNotFoundError):
            mlTransactionModel.importFile(data_csv_path1)

    def test_format(self):
        self.assertEqual(data_csv_path.__contains__(".csv"), True)

    # def test_feature_wrong(self):
    #     with self.assertRaises(NameError):
    #         mlTransactionModel.featureExtraction(data, list(data2), length)

    def test_feature_len(self):
        with self.assertRaises(IndexError):
            mlTransactionModel.featureExtraction(data, list(data), 9)

    def test_feature_success(self):
        self.assertEqual(mlTransactionModel.featureExtraction(data, list(data), length), list(data))

    # def test_completeTest(self):
    #     pathCSV = data_csv_path
    #
    #     with open(data_csv_path) as csvfile:
    #         reader = csv.reader(csvfile)
    #         for row in reader:
    #             for item in row:
    #                 try:
    #                     getattr(mlTransactionModel.importFile(data_csv_path), item)()
    #                 except AttributeError:
    #                     print("Unknown attribute", item, "ignored")


if __name__ == '__main__':
    unittest.main()
