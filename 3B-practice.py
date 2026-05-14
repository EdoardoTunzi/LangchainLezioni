from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

# ====================== MODELLO ======================
model = init_chat_model(
    model="local-model",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


# =========================================================
# 1. COMMA SEPARATED LIST - GET_FORMAT_INSTRUCTIONS
# =========================================================


def test_list_parser():
    print("\n=== LIST OUTPUT PARSER ===")

    output_parser = CommaSeparatedListOutputParser()

    prompt = PromptTemplate(
        template="{query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    chain = prompt | model | output_parser

    while True:
        user = input("\nTu: ")
        if user == "exit":
            break

        res = chain.invoke({"query": user})
        print("\nAI:", res)


# =========================================================
# 2. JSON SCHEMA - WITH_STRUCTURED_OUTPUT
# =========================================================


def test_json_schema():
    print("\n=== JSON SCHEMA ===")

    json_schema = {
        "title": "Movie",
        "description": "A movie with details",
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The title of the movie"},
            "year": {
                "type": "integer",
                "description": "The year the movie was released",
            },
            "director": {"type": "string", "description": "The director of the movie"},
            "rating": {"type": "number", "description": "The movie's rating out of 10"},
        },
        "required": ["title", "year", "director", "rating"],
    }

    structured_model = model.with_structured_output(
        json_schema,
        method="json_schema",
    )

    while True:
        user = input("\nTu: ")
        if user == "exit":
            break

        res = structured_model.invoke(user)
        print("\nAI:", res)


# =========================================================
# 3. TYPEDDICT - WITH_STRUCTURED_OUTPUT
# =========================================================


def test_typeddict():
    print("\n=== TYPEDDICT ===")

    """ class MovieDict(TypedDict):
        title: str
        year: int
        director: str
        rating: float """

    class MovieDict(TypedDict):
        """A movie with details."""

        title: Annotated[str, ..., "The title of the movie"]
        year: Annotated[int, ..., "The year the movie was released"]
        director: Annotated[str, ..., "The director of the movie"]
        rating: Annotated[float, ..., "The movie's rating out of 10"]

    structured_model = model.with_structured_output(MovieDict)

    while True:
        user = input("\nTu: ")
        if user == "exit":
            break

        res = structured_model.invoke(user)
        print("\nTYPE:", type(res))
        print("AI:", res)


# =========================================================
# 4. PYDANTIC - WITH_STRUCTURED_OUTPUT
# =========================================================


def test_pydantic():
    print("\n=== PYDANTIC ===")

    """ class Movie(BaseModel):
        title: str = Field(...)
        year: int = Field(...)
        director: str = Field(...)
        rating: float = Field(...) """

    class Movie(BaseModel):
        """A movie with details."""

        title: str = Field(..., description="The title of the movie")
        year: int = Field(..., description="The year the movie was released")
        director: str = Field(..., description="The director of the movie")
        rating: float = Field(..., description="The movie's rating out of 10")

    structured_model = model.with_structured_output(Movie)

    while True:
        user = input("\nTu: ")
        if user == "exit":
            break

        res = structured_model.invoke(user)
        print("\nTYPE:", type(res))
        print("AI:", res)
        print("TITLE:", res.title)


# =========================================================
# MAIN PER TEST RAPIDO
# =========================================================


def main():
    print("""
SCEGLI TEST:
1 - CommaSeparatedListOutputParser
2 - JSON Schema
3 - TypedDict
4 - Pydantic
exit - uscire
""")

    while True:
        choice = input("\nScelta: ")

        if choice == "exit":
            break

        if choice == "1":
            test_list_parser()

        elif choice == "2":
            test_json_schema()

        elif choice == "3":
            test_typeddict()

        elif choice == "4":
            test_pydantic()


if __name__ == "__main__":
    main()
