# Bianca Pop

import math, os, pickle, re
from typing import Tuple, List, Dict


class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        cache of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates `pos_freq/neg_freq` dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
        # get the list of file names from the training data directory
        # os.walk returns a generator (feel free to Google "python generators" if you're
        # curious to learn more, next gets the first value from this generator or the
        # provided default `(None, None, [])` if the generator has no values)
        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")

        # files now holds a list of the filenames
        # self.training_data_directory holds the folder name where these files are

        # stored below is how you would load a file with filename given by `filename`
        # `text` here will be the literal text of the file (i.e. what you would see
        # if you opened the file in a text editor
        # text = self.load_file(os.path.join(self.training_data_directory, files[3]))
        # print(text)

        # *Tip:* training can take a while, to make it more transparent, we can use the
        # enumerate function, which loops over something and has an automatic counter.
        # write something like this to track progress (note the `# type: ignore` comment
        # which tells mypy we know better and it shouldn't complain at us on this line):
        
        # Create a list of stopwords
        file = self.load_file("sorted_stoplist.txt")
        # print(file)
        stopwords = self.tokenize(file)
        print(stopwords)
        
        for index, filename in enumerate(files, 1): # type: ignore
            print(f"Training on file {index} of {len(files)}")
        #     <the rest of your code for updating frequencies here>
            text = self.load_file(os.path.join(self.training_data_directory, filename))
            tokens = self.tokenize(text)
            # print(tokens)

            filtered_tokens = [token for token in tokens if token not in stopwords]
            # print(filtered_tokens)

            if filename.startswith(self.pos_file_prefix):
                self.update_dict(filtered_tokens, self.pos_freqs)
            elif filename.startswith(self.neg_file_prefix):
                self.update_dict(filtered_tokens, self.neg_freqs)


        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)



    def classify(self, text: str) -> str:
        """Classifies given text as positive, negative or neutral from calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification, either positive, negative or neutral
        """
        # TODO: fill me out
        
        tokens = self.tokenize(text)
        print(tokens)

        file = self.load_file("sorted_stoplist.txt")
        stopwords = self.tokenize(file)

        pos_score = 0
        neg_score = 0


        pos_total = sum(self.pos_freqs.values())
        # print(pos_total)
        neg_total = sum(self.neg_freqs.values())
        # print(neg_total)
        
        # Creating the entire vocab and finding the size
        vocab = set(self.pos_freqs.keys()).union(self.neg_freqs.keys())
        vocab_size = len(vocab)

        for token in tokens:
            if token not in stopwords:

                pos_freqs = self.pos_freqs.get(token, 0) + 1
                neg_freqs = self.neg_freqs.get(token, 0) + 1


                pos_score += math.log(pos_freqs / (pos_total + vocab_size))
                neg_score += math.log(neg_freqs / (neg_total + vocab_size))

        print(pos_score, neg_score)


        if pos_score > neg_score:
            return "positive"
        else:
            return "negative"


    def load_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        # TODO: your work here
        # print("update dict")
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1


if __name__ == "__main__":
    b = BayesClassifier()
    a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", \
                       "you", "like", "it", "too"]
    a_dictionary = {}
    b.update_dict(a_list_of_words, a_dictionary)
    assert a_dictionary["I"] == 2, "update_dict test 1"
    assert a_dictionary["like"] == 2, "update_dict test 2"
    assert a_dictionary["really"] == 1, "update_dict test 3"
    assert a_dictionary["too"] == 1, "update_dict test 4"
    print("update_dict tests passed.")

    pos_denominator = sum(b.pos_freqs.values())
    neg_denominator = sum(b.neg_freqs.values())

    print("\nThese are the sums of values in the positive and negative dicitionaries.")
    print(f"sum of positive word counts is: {pos_denominator}")
    print(f"sum of negative word counts is: {neg_denominator}")

    print("\nHere are some sample word counts in the positive and negative dicitionaries.")
    print(f"count for the word 'love' in positive dictionary {b.pos_freqs['love']}")
    print(f"count for the word 'love' in negative dictionary {b.neg_freqs['love']}")
    print(f"count for the word 'terrible' in positive dictionary {b.pos_freqs['terrible']}")
    print(f"count for the word 'terrible' in negative dictionary {b.neg_freqs['terrible']}")
    print(f"count for the word 'computer' in positive dictionary {b.pos_freqs['computer']}")
    print(f"count for the word 'computer' in negative dictionary {b.neg_freqs['computer']}")
    print(f"count for the word 'science' in positive dictionary {b.pos_freqs['science']}")
    print(f"count for the word 'science' in negative dictionary {b.neg_freqs['science']}")


    print("\nHere are some sample probabilities.")
    print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")

    print("\nThe following should all be positive.")
    print(b.classify('I love computer science'))
    print(b.classify('this movie is fantastic'))
    print("\nThe following should all be negative.")
    print(b.classify('rainy days are the worst'))
    print(b.classify('computer science is terrible'))
    print()
    
    print(b.classify("No way should this have beaten Traffic for best movie."))
    pass