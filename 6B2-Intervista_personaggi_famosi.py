from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ESEMPIO CON OPENAI
model = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name="gpt-4.1-nano",
    temperature=0.7,  # livello di creatività/casualità nelle risposte (0: minima variabilità, 1: massima variabilità)
    max_tokens=1024,  # numero massimo di token che il modello può generare in una sola risposta
    request_timeout=30,
)
# ====================== MODELLO ======================
""" model = init_chat_model(
    model="local-model",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
) """

# =========================================================
promptAlan = ChatPromptTemplate.from_template(
    "Act as Alan Watts, esprimi il tuo punto di vista su questo argomento: {topic}, in modo conciso e chiaro",
)
promptPlato = ChatPromptTemplate.from_template(
    "Act as Plato, esprimi il tuo punto di vista su questo argomento: {topic}, in modo conciso e chiaro",
)
promptJung = ChatPromptTemplate.from_template(
    "Act as Carl Jung, esprimi il tuo punto di vista su questo argomento: {topic}, in modo conciso e chiaro",
)


parallel_chain = RunnableParallel(
    alan=(promptAlan | model | StrOutputParser()),
    plato=(promptPlato | model | StrOutputParser()),
    jung=(promptJung | model | StrOutputParser()),
)


def runInterview():

    print("\n=== Opinioni di personaggi famosi ===")

    print("Inserisci un topic su cui far dialogare i personaggi\n")
    print("exit per uscire\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            break

        print("\nAI: ", end="", flush=True)
        result = parallel_chain.invoke({"topic": user_input})

        print("\n--- Alan Watts ---")

        print(result["alan"])
        print("\n--- Platone ---")
        print(result["plato"])
        print("\n--- Carl Jung ---")
        print(result["jung"])


runInterview()
