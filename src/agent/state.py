from typing import Annotated
from operator import add
from typing_extensions import TypedDict
from pydantic import BaseModel


class AgentState(TypedDict):
    original_query: str
    is_compound: bool
    sub_queries: list[str]
    accumulated_context: Annotated[list[dict], add]
    context_sufficient: bool
    missing_elements: list[str]
    retry_count: int
    final_answer: str
    citations: list[dict]


class QueryPlan(BaseModel):
    is_compound: bool
    sub_queries: list[str]


class InformationCheck(BaseModel):
    has_sufficient_info: bool
    missing_elements: list[str]
