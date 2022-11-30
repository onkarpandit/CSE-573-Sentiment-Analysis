import spacy
from keras.callbacks import *
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_md')

special_chars = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '#', '—–']


def refine_text(xyz):
    xyz = re.sub(r"\'s", " \'s", xyz)
    xyz = re.sub(r"\'ve", " \'ve", xyz)
    xyz = re.sub(r"n\'t", " n\'t", xyz)
    xyz = re.sub(r"\'re", " \'re", xyz)
    xyz = re.sub(r"\'d", " \'d", xyz)
    xyz = re.sub(r"\'ll", " \'ll", xyz)
    xyz = re.sub(r",", " , ", xyz)
    xyz = re.sub(r"!", " ! ", xyz)
    xyz = re.sub(r"\(", " \( ", xyz)
    xyz = re.sub(r"\)", " \) ", xyz)
    xyz = re.sub(r"\?", " \? ", xyz)
    xyz = re.sub(r"\s{2,}", " ", xyz)

    x = re.compile('<.*?>')

    xyz = re.sub(x, '', xyz)
    xyz = xyz.replace('_', '')

    return xyz.strip().lower()


def delete_special_chars(content):
    content = str(content)
    for _char in special_chars:
        content = content.replace(_char, f' {_char} ')
    return content


def delete_end_terms(content):
    content = str(content)
    content = content.lower().split()
    breaks = set(stopwords.words("english"))
    content = [term for term in content if not term in breaks and len(term) >= 3]
    content = " ".join(content)
    return content


def scale_text(content):
    content = content.lower().strip()
    data = nlp(content)
    useful_sentences = []
    for line in data.sents:
        line = delete_special_chars(line)
        line = refine_text(line)
        useful_sentences.append(line)
    return useful_sentences
