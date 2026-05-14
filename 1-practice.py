import os
from langchain_openai import ChatOpenAI  # installa con pip install langchain-openai
from langchain.chat_models import (
    init_chat_model,
)  # installa con pip install -U langchain
from dotenv import load_dotenv
import asyncio

load_dotenv()  # installa con pip install python-dotenv

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ESEMPIO CON OPENAI
model = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name="gpt-4.1-nano",
    temperature=0.7,  # livello di creatività/casualità nelle risposte (0: minima variabilità, 1: massima variabilità)
    max_tokens=1024,  # numero massimo di token che il modello può generare in una sola risposta
    request_timeout=30,
)

# response = model.invoke("Ciao")
# print(response.content)

# ESEMPIO CON MODELLO LOCALE (LM Studio) - ASSICURARSI CHE LM STUDIO SIA IN ESECUZIONE SULLA PORTA 1234
local_llm = init_chat_model(
    model="local-model",  # ok in locale, ma in produzione va messo il nome del modello che si vuole utilizzare, es: gpt-4o-mini
    # LM Studio è compatibile con l'API di OpenAI, quindi usiamo "openai" come provider anche per i modelli locali
    # quindi si installa con pip install langchain-openai
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# response = local_llm.invoke("Ciao")
# print(response.content)

# ========================================


# scegli qui quale modello vuoi testare

active_model = local_llm

# active_model = model


# =========================================================

# 1. invoke()

# =========================================================


def test_invoke():

    print("\n=== TEST invoke() ===")

    print("Scrivi 'exit' per uscire.\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            print("Chat terminata.")

            break

        response = active_model.invoke(user_input)

        print(f"\nAI: {response.content}")


# =========================================================

# 2. ainvoke()

# =========================================================


async def test_ainvoke():

    print("\n=== TEST ainvoke() ===")

    print("Scrivi 'exit' per uscire.\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            print("Chat terminata.")

            break

        response = await active_model.ainvoke(user_input)

        print(f"\nAI: {response.content}")


# =========================================================

# 3. stream()

# =========================================================


def test_stream():

    print("\n=== TEST stream() ===")

    print("Scrivi 'exit' per uscire.\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            print("Chat terminata.")

            break

        print("\nAI: ", end="", flush=True)

        for chunk in active_model.stream(user_input):

            print(chunk.content, end="", flush=True)

        print()


# =========================================================

# 4. astream()

# =========================================================


async def test_astream():

    print("\n=== TEST astream() ===")

    print("Scrivi 'exit' per uscire.\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            print("Chat terminata.")

            break

        print("\nAI: ", end="", flush=True)

        async for chunk in active_model.astream(user_input):

            print(chunk.content, end="", flush=True)

        print()


# =========================================================

# 5. batch()

# =========================================================


def test_batch():

    print("\n=== TEST batch() ===")

    print("Scrivi più prompt separati da virgola.")

    print("Esempio: ciao, cos'è react, cos'è langchain")

    print("Scrivi 'exit' per uscire.\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            print("Chat terminata.")

            break

        prompts = [p.strip() for p in user_input.split(",")]

        responses = active_model.batch(prompts)

        print()

        for i, response in enumerate(responses, start=1):

            print(f"=== RISPOSTA {i} ===")

            print(response.content)

            print()


# =========================================================

# 6. abatch()

# =========================================================


async def test_abatch():

    print("\n=== TEST abatch() ===")

    print("Scrivi più prompt separati da virgola.")

    print("Esempio: ciao, cos'è react, cos'è langchain")

    print("Scrivi 'exit' per uscire.\n")

    while True:

        user_input = input("\nTu: ")

        if user_input.lower() == "exit":

            print("Chat terminata.")

            break

        prompts = [p.strip() for p in user_input.split(",")]

        responses = await active_model.abatch(prompts)

        print()

        for i, response in enumerate(responses, start=1):

            print(f"=== RISPOSTA {i} ===")

            print(response.content)

            print()


# =========================================================

# ESECUZIONE

# lascia attiva solo quella che vuoi testare

# =========================================================

# test_invoke()

# asyncio.run(test_ainvoke())

# test_stream()

# asyncio.run(test_astream())

# test_batch()

# asyncio.run(test_abatch())
