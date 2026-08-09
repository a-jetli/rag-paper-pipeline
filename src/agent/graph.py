from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import planner_node, make_retriever_node, grader_node, reformulator_node, synthesizer_node


MAX_RETRIEVAL_PASSES = 2  # initial pass + at most 1 reformulation retry


def route_after_grader(state: AgentState) -> str:
    """Which node runs after the grader.

    Public because the streaming API calls it too: the progress feed reports node
    *completions*, so without asking this directly the client would have to guess
    which arm was taken and correct itself a second later.
    """
    if state["context_sufficient"] or state["retry_count"] >= MAX_RETRIEVAL_PASSES:
        return "synthesizer"
    return "reformulator"


def build_graph(collection, bm25_index):
    retriever_node = make_retriever_node(collection, bm25_index)

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("reformulator", reformulator_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges("grader", route_after_grader, {"synthesizer": "synthesizer", "reformulator": "reformulator"})
    graph.add_edge("reformulator", "retriever")
    graph.add_edge("synthesizer", END)

    return graph.compile()
