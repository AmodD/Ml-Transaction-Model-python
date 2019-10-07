import csv
import string
import unittest
import mlTransactionModel


path="/home/diwakar/Downloads/Sample Transactions-test Data.csv"
path1="/home/diwakar/Downloads/Sample Transactions-Test Data-Assignment-2.csv"
#path3="/home/diwakar/Downloads/Sample Transactions-test Data.txt"

class MyTestCase(unittest.TestCase):
    def test_csv(self):
        self.assertEqual(mlTransactionModel.importFile("/home/diwakar/Downloads/Sample Transactions-test Data.csv"),path )
    def test_csv_not_found(self):
        with self.assertRaises(FileNotFoundError):
            mlTransactionModel.importFile(path1)

    def test_csv_format(self):
        try:
            with open(path, newline='') as csvfile:
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
    def test_format(self):
        self.assertEqual(path.__contains__(".csv"),True)

    # def test_completeTest(self):
    #     pathCSV = path
    #
    #     with open(path) as csvfile:
    #         reader = csv.reader(csvfile)
    #         for row in reader:
    #             for item in row:
    #                 try:
    #                     getattr(mlTransactionModel.importFile(path), item)()
    #                 except AttributeError:
    #                     print("Unknown attribute", item, "ignored")


if __name__ == '__main__':
    unittest.main()
