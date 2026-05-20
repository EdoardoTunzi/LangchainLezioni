import json
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from IPython.display import Image, display

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ESEMPIO CON OPENAI
llm = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name="gpt-4.1-nano",
    temperature=0.5,  # livello di creatività/casualità nelle risposte (0: minima variabilità, 1: massima variabilità)
)

# Definizione del template per l'analisi dell'ambito del progetto
# servirà come prompt per il modello di linguaggio,
# chiedendo di analizzare la descrizione del progetto e fornire
# informazioni specifiche sull'ambito del progetto.
SCOPE_TEMPLATE = """Analizza il seguente progetto e definisci il suo ambito:
Progetto: {project_description}
Fornisci: 1) Funzionalità principali, 2) Tipi di utenti target, 3) Piattaforme di sviluppo
"""


# Definizione del modello Pydantic per strutturare l'output dell'analisi dell'ambito
# specificando i campi richiesti e le loro descrizioni.
class ProjectScope(TypedDict):
    """A project scope definition."""

    main_features: Annotated[List[str], ..., "Funzionalità principali del progetto"]
    target_users: Annotated[List[str], ..., "Tipi di utenti target del progetto"]
    platforms: Annotated[
        List[str], ..., "Piattaforme su cui il progetto sarà sviluppato"
    ]


# Funzione per analizzare l'ambito del progetto
def analyze_scope(project_description: str) -> Dict[str, Any]:
    """ "
    Creazione di un prompt formattato che include:
        # 1. Il template SCOPE_TEMPLATE
        # 2. La descrizione del progetto come variabile di input
        # 3. Istruzioni di formattazione basate sul modello ProjectScope
    """
    prompt = PromptTemplate(
        template=SCOPE_TEMPLATE,
        input_variables=["project_description"],
    )

    result = llm.with_structured_output(
        ProjectScope,
        method="json_schema",
    ).invoke(prompt.format(project_description=project_description))

    print(f"\n\nOutput grezzo dell'LLM per l'analisi: {result}")

    return result


project_description = """
Sviluppare una piattaforma di e-learning che offra corsi online interattivi 
su vari argomenti tecnologici. La piattaforma deve supportare video lezioni, 
quiz interattivi, forum di discussione e un sistema di monitoraggio dei progressi 
degli studenti. Dovrebbe essere accessibile via web e tramite app mobile, 
con funzionalità offline per alcune risorse di apprendimento.
"""

scope = analyze_scope(project_description)

# ====================

TECH_REQ_TEMPLATE = """Basandoti sulla descrizione e l'ambito del progetto, definisci i requisiti tecnici:
Progetto: {project_description}
Ambito: {project_scope}
Fornisci: 1) Tecnologie backend, 2) Tecnologie frontend, 3) Tipo di database, 4) API esterne necessarie 
"""


class TechnicalRequirements(TypedDict):
    backend_tech: Annotated[List[str], ..., "Tecnologie backend richieste"]
    frontend_tech: Annotated[List[str], ..., "Tecnologie frontend richieste"]
    database: Annotated[str, ..., "database da utilizzare"]
    apis: Annotated[List[str], ..., "API esterne necessarie"]


def analyze_tech_requirements(data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(
        template=TECH_REQ_TEMPLATE,
        input_variables=["project_description", "project_scope"],
    )
    result = llm.with_structured_output(
        TechnicalRequirements, method="json_schema"
    ).invoke(prompt.format(**data))

    print(f"\n\nOutput grezzo dell'LLM per i requisiti tecnici: {result}")
    return result


## TEST
techReq = analyze_tech_requirements(
    {"project_description": project_description, "project_scope": scope}
)


# ===================

TASK_TEMPLATE = """Crea una suddivisione dettagliata dei task per il progetto:
Progetto: {project_description}
Ambito: {project_scope}
Requisiti Tecnici: {tech_requirements}
Fornisci: 1) Fasi principali del progetto, 2) Compiti specifici per ogni fase 
"""


class TaskBreakdown(TypedDict):
    phases: Annotated[List[str], ..., "Fasi principali del progetto"]
    tasks: Annotated[Dict[str, List[str]], ..., "Compiti specifici per ogni fase"]


def create_task_breakdown(data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(
        template=TASK_TEMPLATE,
        input_variables=["project_description", "project_scope", "tech_requirements"],
    )

    result = llm.with_structured_output(TaskBreakdown, method="json_schema").invoke(
        prompt.format(**data)
    )

    print(f"\n\nOutput grezzo dell'LLM per la suddivisione dei task: {result}")

    return result


## TEST
breaktask = create_task_breakdown(
    {
        "project_description": project_description,
        "project_scope": scope,
        "tech_requirements": techReq,
    }
)

# ===================

MARKET_TEMPLATE = """Fai una analisi di mercato mercato per il seguente progetto:
Progetto: {project_description}
Ambito: {project_scope}
Fornisci: 1) Dimensione stimata del mercato target, 2) Principali concorrenti, 3) Punti di forza unici 
"""


class MarketAnalysis(TypedDict):
    target_market_size: Annotated[str, ..., "Dimensione stimata del mercato target"]
    main_competitors: Annotated[List[str], ..., "Principali concorrenti"]
    unique_selling_points: Annotated[
        List[str], ..., "Punti di forza unici del progetto"
    ]


def analyze_market(data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(
        template=MARKET_TEMPLATE,
        input_variables=["project_description", "project_scope"],
    )

    result = llm.with_structured_output(MarketAnalysis, method="json_schema").invoke(
        prompt.format(**data)
    )
    print(f"\n\nOutput grezzo dell'LLM per l'analisi di mercato: {result}")
    return result


## TEST
market = analyze_market(
    {"project_description": project_description, "project_scope": scope}
)


# ===================

RESOURCE_TEMPLATE = """Stima le risorse necessarie per il progetto:
Progetto: {project_description}
Suddivisione Task: {task_breakdown}
Fornisci: 1) Dimensione del team, 2) Ruoli necessari, 3) Timeline stimata, 4) Intervallo di budget 
"""


class ResourceEstimate(TypedDict):
    team_size: Annotated[int, ..., "Dimensione consigliata del team"]
    roles: Annotated[List[str], ..., "Ruoli necessari nel team"]
    timeline: Annotated[str, ..., "Stima del tempo totale di sviluppo"]
    budget_range: Annotated[str, ..., "Intervallo di budget stimato"]


def estimate_resources(data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(
        template=RESOURCE_TEMPLATE,
        input_variables=["project_description", "task_breakdown"],
    )

    result = llm.with_structured_output(ResourceEstimate, method="json_schema").invoke(
        prompt.format(**data)
    )

    print(f"\n\nOutput grezzo dell'LLM per la stima delle risorse: {result}")
    return result


## TEST
resource = estimate_resources(
    {"project_description": project_description, "task_breakdown": breaktask}
)
# ===================
RISK_TEMPLATE = """Valuta i rischi potenziali del progetto:
Progetto: {project_description}
Ambito: {project_scope}
Requisiti Tecnici: {tech_requirements}
Fornisci: 1) Rischi tecnici, 2) Rischi di business, 3) Strategie di mitigazione per i rischi principali
"""


class RiskAssessment(TypedDict):
    technical_risks: Annotated[List[str], ..., "Potenziali rischi tecnici"]
    business_risks: Annotated[List[str], ..., "Potenziali rischi di business"]
    mitigation_strategies: Annotated[
        Dict[str, str], ..., "Strategie di mitigazione per i rischi principali"
    ]


def assess_risks(data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(
        template=RISK_TEMPLATE,
        input_variables=["project_description", "project_scope", "tech_requirements"],
    )
    result = llm.with_structured_output(RiskAssessment, method="json_schema").invoke(
        prompt.format(**data)
    )
    print(f"Output grezzo dell'LLM per la valutazione dei rischi: {result}")
    return result


asses = assess_risks(
    {
        "project_description": project_description,
        "project_scope": scope,
        "tech_requirements": techReq,
    }
)

# =================== creazione catena


# Debug - log intermedi
def debug_print(data: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n\nDebug - Dati correnti: {json.dumps(data, indent=2)}")  # indent=2
    return data


# test catena
test = RunnableSequence(
    RunnableLambda(lambda x: {"project_description": x}),
    RunnableParallel(
        {
            "scope": RunnableLambda(analyze_scope),
            "project_description": lambda x: x["project_description"],
        }
    ),
    RunnableLambda(debug_print),
    RunnableLambda(
        lambda x: {
            **x,
            "tech_requirements": analyze_tech_requirements(
                {
                    "project_description": x["project_description"],
                    "project_scope": x["scope"],
                }
            ),
        }
    ),
)

prova = test.invoke(project_description)

# mainchain

# Catena principale
main_chain = RunnableSequence(
    RunnableLambda(
        lambda x: {"project_description": x}
    ),  # recupero l'input e lo associo alla chiave `project_description`
    # {"project_description":"..."}
    RunnableParallel(
        {
            "scope": RunnableLambda(analyze_scope),  # lancio `analyze_scope`
            "project_description": lambda x: x[
                "project_description"
            ],  # mi porto avanti `project_description`
        }
    ),
    RunnableLambda(debug_print),  # chiamo la funzione di log
    # {"scope":"...", "project_description":"..."}
    RunnableLambda(
        lambda x: {
            **x,  # "copia-incolla" le coppie chiave valore del dizionario di input in output
            "tech_requirements": analyze_tech_requirements(
                {  # associa alla chiave `tech_requirements` il risultato della
                    # funzione `analyze_tech_requirements`
                    "project_description": x[
                        "project_description"
                    ],  # recupero di `project_description` per darlo in input
                    # alla funzione `analyze_tech_requirements`
                    "project_scope": x[
                        "scope"
                    ],  # Trasforma un type <dict> in un type <str>
                }
            ),
        }
    ),  # {"scope":"...", "project_description":"...", "tech_requirements":"..."}
    RunnableLambda(debug_print),
    RunnableParallel(
        {
            "task_breakdown": RunnableLambda(
                lambda x: create_task_breakdown(
                    {
                        "project_description": x["project_description"],
                        "project_scope": x["scope"],
                        "tech_requirements": x["tech_requirements"],
                    }
                )
            ),
            "market_analysis": RunnableLambda(
                lambda x: analyze_market(
                    {
                        "project_description": x["project_description"],
                        "project_scope": x["scope"],
                    }
                )
            ),
            "scope": lambda x: x["scope"],
            "tech_requirements": lambda x: x["tech_requirements"],
            "project_description": lambda x: x["project_description"],
        }
    ),  # {"scope":"...", "project_description":"...", "tech_requirements":"...", "task_breakdown":"...", "market_analysis":"..."}
    RunnableLambda(debug_print),
    # {"scope":"...", "project_description":"...", "tech_requirements":"...", "task_breakdown":"...", "market_analysis":"..."}
    RunnableParallel(
        {
            "resources": RunnableLambda(
                lambda x: estimate_resources(
                    {
                        "project_description": x["project_description"],
                        "task_breakdown": x["task_breakdown"],
                    }
                )
            ),
            "risks": RunnableLambda(
                lambda x: assess_risks(
                    {
                        "project_description": x["project_description"],
                        "project_scope": x["scope"],
                        "tech_requirements": x["tech_requirements"],
                    }
                )
            ),
            "tech_requirements": lambda x: x["tech_requirements"],
            "task_breakdown": lambda x: x["task_breakdown"],
            "market_analysis": lambda x: x["market_analysis"],
            "scope": lambda x: x["scope"],
        }  # {"scope":"...", "tech_requirements":"...", "task_breakdown":"...", "market_analysis":"...", "resources":"...", "risks":"..."}
    ),
    RunnableLambda(debug_print),
)


# Funzione per generare il rapporto finale
def generate_final_report(data: Dict[str, Any]) -> str:
    report_template = """
    Rapporto di Pianificazione del Progetto
    
    1. Ambito del Progetto:
    - Funzionalità principali: {scope[main_features]}
    - Utenti target: {scope[target_users]}
    - Piattaforme: {scope[platforms]}
    
    2. Requisiti Tecnici:
    - Backend: {tech_requirements[backend_tech]}
    - Frontend: {tech_requirements[frontend_tech]}
    - Database: {tech_requirements[database]}
    - API esterne: {tech_requirements[apis]}
    
    3. Suddivisione dei Task:
    - Fasi del progetto: {task_breakdown[phases]}
    - Dettaglio dei task per fase:
    {task_details}
    
    4. Stima delle Risorse:
    - Dimensione del team: {resources[team_size]}
    - Ruoli necessari: {resources[roles]}
    - Timeline stimata: {resources[timeline]}
    - Range di budget: {resources[budget_range]}
    
    5. Valutazione dei Rischi:
    - Rischi tecnici: {risks[technical_risks]}
    - Rischi di business: {risks[business_risks]}
    - Strategie di mitigazione:
    {risk_mitigation}
    
    6. Analisi di Mercato:
    - Dimensione del mercato target: {market_analysis[target_market_size]}
    - Principali concorrenti: {market_analysis[main_competitors]}
    - Punti di forza unici: {market_analysis[unique_selling_points]}
    """

    task_details = "\n".join(
        [
            f"      {phase}:\n      - " + "\n      - ".join(tasks)
            for phase, tasks in data["task_breakdown"]["tasks"].items()
        ]
    )
    risk_mitigation = "\n".join(
        [
            f"    - {risk}: {strategy}"
            for risk, strategy in data["risks"]["mitigation_strategies"].items()
        ]
    )

    return report_template.format(
        **data, task_details=task_details, risk_mitigation=risk_mitigation
    )


# Funzione principale per l'analisi del progetto
def analyze_project(project_description: str) -> str:
    result = main_chain.invoke(project_description)
    return generate_final_report(result)


report = analyze_project(project_description)
