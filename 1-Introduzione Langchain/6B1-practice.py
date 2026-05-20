from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

# ====================== MODELLO ======================
model = init_chat_model(
    model="local-model",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)
# =========================================================
prompt = ChatPromptTemplate.from_template(
    "descrivi questo topic in modo conciso, usando massimo 15 parole: {topic}",
)

analysis_prompt = ChatPromptTemplate.from_template(
    "è una descrizione corretta e breve?\n{topic}"
)

funny_prompt = ChatPromptTemplate.from_template(
    "crea un proverbio divertente sul tema {topic}"
)

chain = prompt | model | StrOutputParser()

parallel_chain = (
    prompt
    | model
    | StrOutputParser()
    | RunnableParallel(
        description=RunnablePassthrough(),
        check=(analysis_prompt | model | StrOutputParser()),
        funny=(funny_prompt | model | StrOutputParser()),
    )
)
res = parallel_chain.invoke("blockchain")
print(res)
