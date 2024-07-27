import re
import unicodedata
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import regex
import string

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

lang_to_code = {
    'hindi': 'hi',
    'sanskrit': 'sa',
    'bengali': 'bn',
    'tamil': 'ta',
    'telugu': 'te',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'odia': 'or',
    'punjabi': 'pa',
    'urdu': 'ur',
}

fleurs_lang_to_code = {
    'hi_in': 'hi',
    'bn_in': 'bn',
    'ta_in': 'ta',
    'te_in': 'te',
    'gu_in': 'gu',
    'kn_in': 'kn',
    'ml_in': 'ml',
    'mr_in': 'mr',
}


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


def normalize_sentence(sentence, lang_code):
    '''
    Perform NFC -> NFD normalization for a sentence and a given language
    sentence: string
    lang_code: language code in ISO format
    '''
    lang_code = lang_code.lower()
    if lang_code in lang_to_code.keys():
        lang_code = lang_to_code[lang_code]
    
    if lang_code in fleurs_lang_to_code.keys():
        lang_code = fleurs_lang_to_code[lang_code]

    try:
        factory=IndicNormalizerFactory()
        normalizer=factory.get_normalizer(lang_code)
        normalized_sentence = normalizer.normalize(sentence)
        normalized_sentence = normalized_sentence.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
    except:
        print('Error: Please pass corrrect language code.')
        return sentence
    return normalized_sentence


class IndicTextNormalizer:
    def __init__(self,  use_indic_normalizer:bool = False, lang : str ='hindi'):
        self.lang = lang
        self.use_indic_normalizer  = use_indic_normalizer
        
    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
    
        s = s.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
    
        if self.use_indic_normalizer:
            s = normalize_sentence(sentence=s,lang_code=self.lang)

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s
