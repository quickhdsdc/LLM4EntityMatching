import os
import csv
import pandas as pd

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label: int = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, file_path, delimiter=",", encoding='utf-8'):
        """
        Reads a CSV file and returns its content.

        :param file_path: Path to the CSV file.
        :param delimiter: Delimiter used in the CSV file (default is comma).
        :param encoding: Encoding of the file (default is 'utf-8').
        :return: A list of lists, where each sub-list represents a row from the CSV file.
        """
        try:
            data_frame = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            return data_frame
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            return []
        except PermissionError:
            print(f"Permission denied. You do not have access to the file {file_path}.")
            return []
        except Exception as e:
            print(f"An error occurred while reading the file {file_path}: {e}")
            return []


class DeepMatcherProcessor(DataProcessor):
    """Processor for preprocessed DeepMatcher data sets (abt_buy, company, etc.)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_df(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train", self._read_csv(os.path.join(os.path.join(data_dir, "tableA.csv"))), self._read_csv(os.path.join(os.path.join(data_dir, "tableB.csv"))))

    def get_valid_examples(self, data_dir):
        """See base class."""
        return self._create_examples_df(
            self._read_csv(os.path.join(data_dir, "valid.csv")), "valid", self._read_csv(os.path.join(os.path.join(data_dir, "tableA.csv"))), self._read_csv(os.path.join(os.path.join(data_dir, "tableB.csv"))))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_df(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test", self._read_csv(os.path.join(os.path.join(data_dir, "tableA.csv"))), self._read_csv(os.path.join(os.path.join(data_dir, "tableB.csv"))))

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _row_to_entity_string(self, row):
        entity_attributes = []
        columns = [col for col in row]
        columns = columns[1:]
        for column in columns:
            # For each attribute in the row, add "Column_Name Value" or "Column_Name nan" if the value is NaN
            value = 'nan' if pd.isnull(row[column].values) else row[column].values
            entity_attributes.append(f"{column.replace('_', ' ')} {value}")
        # Join all attributes with the [SEP] token and return the string
        return ' [SEP] '.join(entity_attributes) + ' [SEP]'

    def _create_examples(self, input_csv, set_type, tableA_csv, tableB_csv):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, row in input_csv.iterrows():
            id_left = row["ltable_id"]
            id_right = row["rtable_id"]  # Assuming there's a similar ID for the right table
            label = row["label"]
            guid = "%s-%s-%s" % (set_type, id_left, id_right)

            # Find the corresponding row in tableA_csv (text_a)
            row_a = tableA_csv[tableA_csv["id"] == id_left]
            if not row_a.empty:
                text_a = self._row_to_entity_string(row_a)
            else:
                text_a = "Not found"

            # Find the corresponding row in tableB_csv (text_b)
            row_b = tableB_csv[tableB_csv["id"] == id_right]
            if not row_b.empty:
                text_b = self._row_to_entity_string(row_b)
            else:
                text_b = "Not found"

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_df(self, input_csv, set_type, tableA_csv, tableB_csv):
        """Creates examples for the training and dev sets."""
        merged_df = pd.DataFrame()
        for i, row in input_csv.iterrows():
            id_left = row["ltable_id"]
            id_right = row["rtable_id"]  # Assuming there's a similar ID for the right table
            label = row["label"]
            guid = "%s-%s-%s" % (set_type, id_left, id_right)

            # Find the corresponding row in tableA_csv (text_a)
            row_a = tableA_csv[tableA_csv["id"] == id_left].copy()
            # Find the corresponding row in tableB_csv (text_b)
            row_b = tableB_csv[tableB_csv["id"] == id_right].copy()

            row_a.columns = [f'{col}_left' for col in row_a.columns]
            row_b.columns = [f'{col}_right' for col in row_b.columns]
            # Concatenate the rows horizontally
            merged_row = pd.concat([row_a.reset_index(drop=True), row_b.reset_index(drop=True)], axis=1)
            merged_row['guid'] = guid
            if label == 0:
                merged_row['label'] = 'no'
            elif label == 1:
                merged_row['label'] = 'yes'

            # Append to the merged_df DataFrame
            merged_df = pd.concat([merged_df, merged_row], ignore_index=True)

        return merged_df