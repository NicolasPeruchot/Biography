"""Useful functions for cleaning the dataset."""

import pandas as pd
import spacy
from spacy.matcher import DependencyMatcher
from dataset_creation.config import matchers

nlp = spacy.load("en_core_web_sm")


def load_dataset(path: str) -> pd.DataFrame:
    """Load datasets."""
    data = pd.read_csv(path, index_col=0)
    try:
        data = data.drop(columns=["id"])
    except:
        pass
    return data


def load_nlp() -> spacy.language.Language:
    """Load the english model."""
    return spacy.load("en_core_web_sm")


def labels_preparation(data: pd.DataFrame) -> pd.DataFrame:
    """Take care of various symbols or case."""
    text = data["txt"]
    text = text.str.replace("'", "", regex=False)
    text = text.str.replace("]", "", regex=False)
    text = text.str.replace("[", "", regex=False)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("A ", "a ", regex=False)
    text = text.str.replace("And ", "and ", regex=False)
    return text


def get_job(doc) -> str:
    """Get the job for a certain sentence."""
    for word in doc:
        if word.dep_ == "attr":
            return word.text
    return ""


def get_matcher(nlp, matcher_name) -> DependencyMatcher:
    """Returns a matcher."""
    matcher = DependencyMatcher(nlp.vocab)
    matcher.add(matcher_name, matchers[matcher_name])
    return matcher


def apply_matcher(matcher, matcher_name, doc) -> str:
    """Apply a matcher to a sentence."""
    match = matcher(doc)
    if matcher_name == "PERSON":
        if match and matcher_name == "PERSON":
            return doc[match[0][1][1] : match[0][1][0] + 1].text
        else:
            return ""
    else:
        if match:
            Span = sorted(match[0][1])
            Job = doc[Span[0] : Span[1]]
            if Job.ents:
                if Job.ents[0].label_ == "NORP":
                    Job = Job[1:]
            if Job:
                return Job.text
    return ""


def name_duplication(data) -> pd.DataFrame:
    """Get rid of duplication in the dataset."""
    duplicates = set()
    for i in range(len(data)):
        if (
            data.ORG.iloc[i] in data.PERSON.iloc[i]
            or data.GPE.iloc[i] in data.PERSON.iloc[i]
        ):
            duplicates.add(i)
    data = data.drop(index=duplicates)
    return data


def clean_nan(data) -> pd.DataFrame:
    """Get rid of sentences without name, job or nationality."""
    for col in ["PERSON", "JOB", "NORP"]:
        data = data[data[col].notna()]
    return data.reset_index(drop=True)


def remove_word(word, data) -> pd.DataFrame:
    """Remove sentences containing a given word."""
    return data[data.txt.str.contains(word) != 1]
