from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import planner_node, make_retriever_node, grader_node, reformulator_node, synthesizer_node


MAX_RETRIEVAL_PASSES = 2  # initial pass + at most 1 reformulation retry


def _route_after_grader(state: AgentState) -> str:
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
    graph.add_conditional_edges("grader", _route_after_grader, {"synthesizer": "synthesizer", "reformulator": "reformulator"})
    graph.add_edge("reformulator", "retriever")
    graph.add_edge("synthesizer", END)

    return graph.compile()
