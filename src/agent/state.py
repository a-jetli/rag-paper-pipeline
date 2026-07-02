from typing import Annotated
from operator import add
from typing_extensions import TypedDict
from pydantic import BaseModel


class AgentState(TypedDict):
    original_query: str
    is_compound: bool
    sub_queries: list[str]
    # Full history of every sub-query searched across all retrieval passes.
    # `sub_queries` itself gets overwritten by the reformulator each retry
    # (retriever_node needs only the current pass's queries), so this is the
    # only field that preserves what was actually searched end to end.
    all_sub_queries: Annotated[list[str], add]
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
