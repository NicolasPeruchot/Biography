"""Creation of the clean dataset."""

import numpy as np
import pandas as pd
from progressbar import progressbar as pb

from dataset_creation.utils import (
    load_dataset,
    load_nlp,
    labels_preparation,
    apply_matcher,
    get_job,
    get_matcher,
    name_duplication,
    clean_nan,
    remove_word,
)


nlp = load_nlp()

dataset = load_dataset("dataset_creation/old_dataset.csv")

text = labels_preparation(dataset)

wrong = set()

newdata = pd.DataFrame(
    index=text.index, columns=["PERSON", "JOB", "JOB1", "NORP", "GPE", "ORG"]
)

prepared_matchers = {
    "PERSON": get_matcher(nlp, "PERSON"),
    "JOB1": get_matcher(nlp, "JOB1"),
}

for i in pb(range(len(text))):
    doc = nlp(text[i])
    job = get_job(doc)
    if job:
        newdata.loc[i]["JOB"] = job
    else:
        wrong.add(i)
    for matcher in prepared_matchers:
        matching = apply_matcher(prepared_matchers[matcher], matcher, doc)
        if matching:
            newdata.loc[i][matcher] = matching
        else:
            wrong.add(i)

    for col in ["NORP", "GPE", "ORG"]:
        for ent in doc.ents:
            if ent.label_ == col and newdata.iloc[i][col] != newdata.iloc[i][col]:
                newdata.loc[i][col] = ent.text

for df in [newdata, text]:
    df.drop(wrong).reset_index().drop(columns=["index"])

data = pd.concat([text, newdata], axis=1)
data = clean_nan(data)

data.JOB = data.JOB + " " + data.JOB1.replace({np.nan: ""})
data.JOB = data.JOB.str.rstrip()
data = data.replace({np.nan: "XXX"}).drop(columns=["JOB1"])

data = name_duplication(data)

for word in [" as ", " who ", " and ", "nickname", "former"]:
    data = remove_word(word, data)

data = data[data.ORG != "XXX"].drop(columns="GPE")
data.to_csv("model_creation/final_dataset.csv")
