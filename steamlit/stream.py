import streamlit as st
from transformers import T5TokenizerFast, TFT5ForConditionalGeneration


@st.cache(allow_output_mutation=True)
def mod():
    return TFT5ForConditionalGeneration.from_pretrained("NicolasPeruchot/Biography")


model = mod()


@st.cache(allow_output_mutation=True)
def toke():
    return T5TokenizerFast.from_pretrained("t5-base")


tokenizer = toke()


desc = "Bio"

st.title("Biography")
st.write(desc)


name = st.text_input("Name")
job = st.text_input("Job")
country = st.text_input("Nationality")
organisation = st.text_input("Organization")

val = name + "|" + job + "|" + country + "|" + organisation


def generate(text):
    toke = tokenizer(text, return_tensors="tf").input_ids
    outputs = model.generate(toke)
    outputs1 = tokenizer.decode(outputs[0]).replace("<pad> ", "").replace("</s>", "")
    return outputs1


if st.button("Generate Text"):
    st.write(generate(val))
