from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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
)
 """

# ====================== MEMORY STORE ======================

store = {}


def get_history(session_id: str):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# ====================== PROMPTS ======================

alan_prompt = ChatPromptTemplate.from_template("""
    Act as Alan Watts.

    Topic: {topic}

    Conversazione finora:
    {history}

    Sei in un podcast e devi esprimere il tuo punto di vista, rispondi in modo conciso e chiaro. 
    Se stai rispondendo ad un altro personaggio, rivolgiti a lui/lei direttamente. Non fare domande, esprimi solo il tuo punto di vista. 
    Usa tono coinvolgente per rendere la conversazione più vivace e interessante.
    Esprimi il tuo accordo o disaccordo con quello che è stato detto dagli altri personaggi riferendoti al all'altro personaggio e non al pubblico. 
    Se non sei d'accordo con quello che è stato detto dagli altri personaggi, spiega il motivo del tuo disaccordo.
""")

jung_prompt = ChatPromptTemplate.from_template("""
    Act as Carl Jung.

    Topic: {topic}

    Conversazione finora:
    {history}

    Sei in un podcast e devi esprimere il tuo punto di vista, rispondi in modo conciso e chiaro. 
    Se stai rispondendo ad un altro personaggio, rivolgiti a lui/lei direttamente. Non fare domande, esprimi solo il tuo punto di vista. 
    Usa un tono calmo e riflessivo, con un linguaggio più formale e accademico, per rendere la conversazione più equilibrata e interessante.
    Esprimi il tuo accordo o disaccordo con quello che è stato detto dagli altri personaggi riferendoti al all'altro personaggio e non al pubblico. 
    Se non sei d'accordo con quello che è stato detto dagli altri personaggi, spiega il motivo del tuo disaccordo.
""")

# ====================== CHAINS + MEMORY ======================

alan_chain = RunnableWithMessageHistory(
    alan_prompt | model | StrOutputParser(),
    get_history,
    input_messages_key="topic",
    history_messages_key="history",
)

jung_chain = RunnableWithMessageHistory(
    jung_prompt | model | StrOutputParser(),
    get_history,
    input_messages_key="topic",
    history_messages_key="history",
)

# ====================== PODCAST LOOP ======================


def podcast():

    print("\n=== AI Podcast ===")
    print("Scrivi 'exit' per uscire.\n")

    while True:

        topic = input("Scegli argomento: ")

        if topic.lower() == "exit":
            break

        # ================= ROUND 1 =================

        alan_response = alan_chain.invoke(
            {
                "topic": topic,
            },
            config={"session_id": topic},
            # uso il topic come session_id per mantenere la coerenza della conversazione su quello specifico argomento,
            # perchè essendo in un loop si creava un bug in cui partendo da un tema, gli interlocutori continuavano a parlare
            # di quel tema anche quando l'utente ne sceglieva un altro, perchè la session_id era sempre la stessa e
            # quindi la history continuava ad accumularsi senza distinzione di argomento.
            # cosi a ogni cambio topic, si crea una nuova sessione con una history vuota, evitando il problema.
        )

        print("\n--- Alan Watts ---\n")
        print(alan_response)

        jung_response = jung_chain.invoke(
            {
                "topic": topic,
            },
            config={"session_id": topic},
        )

        print("\n--- Carl Jung ---\n")
        print(jung_response)

        # ================= ROUND 2 =================

        alan_reply = alan_chain.invoke(
            {
                "topic": topic,
            },
            config={"session_id": topic},
        )

        print("\n--- Alan Watts ---\n")
        print(alan_reply)

        jung_reply = jung_chain.invoke(
            {
                "topic": topic,
            },
            config={"session_id": topic},
        )

        print("\n--- Carl Jung ---\n")
        print(jung_reply)


podcast()
