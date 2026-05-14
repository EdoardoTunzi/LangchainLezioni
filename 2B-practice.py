from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

local_llm = init_chat_model(
    model="local-model",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "Sei un assistente utile di nome Jarvis. Rispondi sempre in modo chiaro e sintetico. Firmati alla fine di ogni risposta con 'Il tuo amico Jarvis'.",
        ),
        ("human", "{input}"),
    ]
)
chain = prompt | local_llm


def test_prompt():

    print("\n=== TEST ===")

    print("exit per uscire\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            break

        print("\nJarvis: ", end="", flush=True)
        # flush=True per forzare la stampa immediata(effetto chatGPT)
        # flush=True in Python serve a forzare lo svuotamento immediato del buffer di output.
        for chunk in chain.stream({"input": user_input}):

            print(chunk.content, end="", flush=True)

        print()


test_prompt()
