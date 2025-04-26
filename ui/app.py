import streamlit as st
from chatbot import init_rag

st.set_page_config(page_title="🍽️ Restaurant RAG Chatbot")
st.title("🍽️ Restaurant RAG Chatbot")

# Load model once
@st.cache(allow_output_mutation=True)
def load_model():
    return init_rag()

tokenizer, model = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask about any restaurant:")
if st.button("Send") and query:
    st.session_state.history.append(("AD", query))
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    print(inputs)
    inputs.pop("token_type_ids", None)
    outputs = model.generate(**inputs)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    st.session_state.history.append(("J.A.R.V.I.S", answer))

for speaker, text in st.session_state.history:
    st.markdown(f"**{speaker}:** {text}")