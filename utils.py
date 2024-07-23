import textacy
import string, re
from collections import Counter, defaultdict
from math import log

VP_PATTERNS = [
    [{"POS": "ADV"}, {"POS": "VERB"}],
    [{"POS": "NOUN"}, {"POS": "VERB"}],
    [{"POS": "PRON"}, {"POS": "VERB"}],
    [{"POS": "ADJ"}, {"POS": "VERB"}],
    [{"POS": "VERB"}, {"POS": "PART"}],
    [{"POS": "VERB"}, {"POS": "NOUN"}],
    [{"POS": "VERB"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
    [{"POS": "VERB"}, {"POS": "ADV"}],
    [{"POS": "VERB"}, {"POS": "ADJ"}],
    [{"POS": "VERB"}, {"POS": "PRON"}],
    [{"POS": "VERB"}, {"POS": "ADP"}],
    [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "ADV"}],
    [{"POS": "VERB"}, {"POS": "CONJ"}, {"POS": "VERB"}],
    [{"POS": "VERB"}, {"POS": "DET"}, {"POS": "NOUN"}],
    [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "ADJ"}, {"POS": "NOUN"}],
    [{"POS": "VERB"}, {"POS": "ADJ"}, {"POS": "NOUN"}],
    [{"POS": "VERB"}, {"POS": "DET"}, {"POS": "ADJ"}],
    [{"POS": "VERB"}, {"POS": "PART"}, {"POS": "ADP"}]
]

def extract_phrases(text, model, return_value = 'phrase'):
    """
    Extracts phrases and corresponding POS tags from a given text using a provided model.

    Args:
        text (str): The text from which to extract phrases.
        model: The model to use for processing the text.
        return_value (str, optional): The type of value to return. Can be either 'phrase' or 'tag'. Defaults to 'phrase'.

    Returns:
        list: A list of phrases extracted from the text.
        list: A list of corresponding POS tags for each phrase.

    Raises:
        ValueError: If the `return_value` parameter is not 'phrase' or 'tag'.

    """
    doc = model(text)
    verb_phrases = textacy.extract.token_matches(doc, VP_PATTERNS)

    phrase_list = []
    tag_list = []

    # Extract Noun Phrases and corresponding pos tags
    for chunk in doc.noun_chunks:
        phrase_list.append(chunk.text)
        tag_list.append(' '.join([t.pos_ for t in chunk]))
    # Print all Verb Phrase and corresponding pos tags
    for chunk in verb_phrases:
        phrase_list.append(chunk.text)
        tag_list.append(' '.join([t.pos_ for t in chunk]))
    
    if return_value == 'phrase':
        return phrase_list
    elif return_value == 'tag':
        return tag_list
    
    # for cleaning text
def clean_str(sentence ,use=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if not use: return sentence

    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence.strip().lower()

def process_corpus(corpus):
    """
    Process a corpus of text and extract word and pair probabilities.

    Args:
        corpus (list): A list of strings representing the corpus of text.

    Returns:
        tuple: A tuple containing three dictionaries:
            - word_prob (dict): A dictionary mapping each word to its probability.
            - pair_prob (dict): A dictionary mapping each pair of words to its probability.
            - unique_words (set): A set of unique words in the corpus.

    Description:
        This function takes a corpus of text as input and processes it to extract word and pair probabilities.
        It iterates over each line in the corpus, cleans the text using the `clean_str` function, splits the line into words,
        and updates the word count and pair count. It then calculates the word and pair probabilities by dividing the count
        by the total number of words in the corpus. The function returns three dictionaries: `word_prob`, `pair_prob`,
        and `unique_words`.

    Example:
        >>> corpus = ["This is a sample text.", "Another sample text."]
        >>> word_prob, pair_prob, unique_words = process_corpus(corpus)
        >>> word_prob
        {'is': 0.25, 'a': 0.25, 'sample': 0.5, 'text': 0.5, 'this': 0.25, 'another': 0.25}
        >>> pair_prob
        {('is', 'sample'): 0.25, ('sample', 'text'): 0.5, ('this', 'is'): 0.25, ('another', 'sample'): 0.25}
        >>> unique_words
        {'sample', 'text', 'is', 'this', 'another', 'a'}
    """
    unique_words = set()
    word_count = Counter()
    pair_count = defaultdict(int)
    total_words = 0
    
    for line in corpus:
        line = clean_str(line)
        words = line.split()
        total_words += len(words)
        word_count.update(words)
        for i, word in enumerate(words):
            unique_words.add(word)
            for j in range(i + 1, len(words)):
                pair = tuple(sorted([word, words[j]]))
                pair_count[pair] += 1
    
    word_prob = {word: count / total_words for word, count in word_count.items()}
    pair_prob = {pair: count / total_words for pair, count in pair_count.items()}
    
    return word_prob, pair_prob, unique_words

def calculate_pmi(word_prob, pair_prob, word1, word2):
    """
    Calculates the Pointwise Mutual Information (PMI) between two words.

    Parameters:
        word_prob (dict): A dictionary containing the probabilities of each word.
        pair_prob (dict): A dictionary containing the probabilities of each word pair.
        word1 (str): The first word for calculating PMI.
        word2 (str): The second word for calculating PMI.

    Returns:
        float: The calculated Pointwise Mutual Information (PMI) between the two words.
    """
    pair = tuple(sorted([word1, word2]))
    if pair in pair_prob and word1 in word_prob and word2 in word_prob:
        pmi = log(pair_prob[pair] / (word_prob[word1] * word_prob[word2]))
        return pmi
    return 0.0

def process_corpus_tags(corpus: list[list]):
    """
    Process the corpus tags to calculate the probabilities and counts of individual POS tags and pairs of POS tags.

    Parameters:
    corpus (list[list]): A list of lists where each inner list represents a line of text with POS tags.

    Returns:
    pos_tag_prob (dict): A dictionary containing the probability of each unique POS tag.
    pos_tag_pair_prob (dict): A dictionary containing the probability of each unique pair of POS tags.
    unique_pos_tags (set): A set of unique POS tags present in the corpus.
    """
    
    unique_pos_tags = set()
    pos_tag_count = Counter()
    pos_tag_pair_count = defaultdict(int)
    total_pos_tags = 0
    
    for line in corpus:
        total_pos_tags += len(line)
        pos_tag_count.update(line)
        
        for i, pos_tag in enumerate(line):
            unique_pos_tags.add(pos_tag)
            
            for j in range(len(line)):
                if i == j: continue
                pair = tuple(sorted([pos_tag, line[j]]))
                pos_tag_count[pair] += 1
    
    pos_tag_prob = {pos_tag: count / total_pos_tags for pos_tag, count in pos_tag_count.items()}
    pos_tag_pair_prob = {pos_tag_pair: count / total_pos_tags for pos_tag_pair, count in pos_tag_pair_count.items()}
    
    return pos_tag_prob, pos_tag_pair_prob, unique_pos_tags

