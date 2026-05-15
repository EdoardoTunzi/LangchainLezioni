from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from IPython.display import display

# Script di esempio: estrazione strutturata di articoli da una fattura in PDF
# Processo generale:
# 1) Caricare il PDF e leggere il testo della pagina
# 2) Definire modelli Pydantic per la struttura di output desiderata
# 3) Inizializzare il modello LLM con output strutturato
# 4) Costruire il prompt e invocare la catena per ottenere dati strutturati
# 5) Convertire l'output in un DataFrame pandas e visualizzarlo

# ====================== MODELLO ======================
model = init_chat_model(
    model="local-model",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# =========================================================
# 1. PDF LOADER
# Caricamento PDF: `PyPDFLoader` estrae il testo dalle pagine del PDF.


doc = PyPDFLoader("data/fattura.pdf").load()

# `doc` è una lista di oggetti pagina; in questo esempio commentato uso la prima pagina.
""" print(
    model.invoke(
        "elenca articoli e quantità presenti nella seguente fattura:\n"
        + doc[0].page_content
    ).content
) """

# 2. MODELLI PYDANTIC
# Definizione dei modelli Pydantic: `Product` rappresenta un singolo articolo, mentre `Products_List` rappresenta la lista complessiva di articoli estratti.


class Product(BaseModel):
    """Details of an item or product present in a document"""

    product_code: str = Field(
        default="--", description="the product code associated with an item"
    )
    description: str = Field(
        default=None, description="the description associated with an item"
    )
    quantity: str = Field(
        default=0,
        description="how many items are in an order, can have different units of measurement",
    )
    price: float = Field(default=0.0, description="the price of the item")


class Products_List(BaseModel):
    """Identifying information about all items in a document"""

    products: List[Product]


# Configurazione dell'output strutturato:
# `with_structured_output` indica al modello di restituire dati che
# possono essere validati/parsati tramite il modello Pydantic `Products_List`.
products_llm = model.with_structured_output(Products_List)

# 3. PROMPT E CATENA
# Creazione del prompt: inseriamo il testo della pagina PDF nella variabile
# `{doc}` per chiedere al modello di elencare gli articoli in formato strutturato.
prompt = PromptTemplate(
    template="elenca codice articoli, descrizione articoli, quantità e prezzo presenti nella seguente fattura:\n{doc}",
    input_variables=["doc"],
)

# Invocazione della catena: eseguiamo il prompt passando il contenuto della
# prima pagina del PDF.
chain = prompt | products_llm

# `articoli_fattura` sarà un'istanza di
# `Products_List` (Pydantic) con il campo `products` popolato.
articoli_fattura = chain.invoke({"doc": doc[0].page_content})

# 4. VISUALIZZAZIONE
# Visualizzazione: creiamo un DataFrame pandas per una comoda ispezione
# tabellare dei risultati estratti e lo mostriamo in output.
df = pd.DataFrame(
    [
        {
            "Codice Prodotto": p.product_code,
            "Descrizione": p.description,
            "Quantità": p.quantity,
            "Prezzo": p.price,
        }
        for p in articoli_fattura.products
    ]
)

display(df)
