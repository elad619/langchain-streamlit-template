"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from PIL import Image
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from art_config import art_pieces

template = """The following is a friendly conversation between a human and an Art piece of the National Gallery. 
The Art piece is talkative and provides lots of specific details from its context.
If the Art piece does not know the answer to a question, it truthfully says it does not know.
The Art piece is polite and its persona and manners are determined by its context and background.
Information about the Art piece:
{information}

Current conversation:
{{history}}
Human: {{input}}
Art piece:"""

def load_chain(description):
    formatted_template = template.format(information=description)
    llm = OpenAI(temperature=0)
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=formatted_template
    )
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="Art piece")
    )
    return conversation

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.title("ArtBot 🎨")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


art_piece = st.sidebar.selectbox(
    "Select an Art Piece", list(art_pieces.keys()), key="art_piece"
)

image_filepath=art_pieces[art_piece]["image_file_path"]
image = Image.open(image_filepath)
st.image(image, caption=art_piece)
st.header("Description")
st.write(art_pieces[art_piece]["short_description"])
st.header("Chat With the painting!")
user_input = get_text()

# noinspection PyInterpreter
chain = load_chain(art_pieces[art_piece]["short_description"])

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
