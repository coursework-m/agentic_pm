"""Workflow for the Agentic PM project."""
from langgraph.graph import StateGraph, END
from workflow.state import CustomState
from nodes.nodes import (
    router_node,
    memory_node,
    data_node,
    analyst_node,
    researcher_node,
    tool_node,
    trader_node,
    portfolio_node,
    store_node,
    performance_node
)
def build_workflow(store, checkpointer=None, checkpoint=False):
    """Builds the workflow for the Agentic PM project."""
    workflow = StateGraph(CustomState)
    workflow.add_node("memory_node", memory_node)
    workflow.add_node("data_node", data_node)
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("trader_node", trader_node)
    workflow.add_node("store_node", store_node)
    workflow.add_node("portfolio_node", portfolio_node)
    workflow.add_node("performance_node", performance_node)
    workflow.add_node("router_node", router_node)
    workflow.set_entry_point("memory_node")
    workflow.add_conditional_edges("researcher_node", router_node, {
        "tool_node": "tool_node",
        "trader_node": "trader_node",
    })
    workflow.add_edge("memory_node", "data_node")
    workflow.add_edge("data_node", "analyst_node")
    workflow.add_edge("analyst_node", "researcher_node")
    workflow.add_edge("tool_node", "researcher_node")
    # Old flow
    # workflow.add_edge("trader_node", "store_node")
    # workflow.add_edge("store_node", "portfolio_node")
    # workflow.add_edge("portfolio_node", END)
    # New flow
    # Trader → Portfolio → Store → END
    workflow.add_edge("trader_node", "portfolio_node")
    workflow.add_edge("portfolio_node", "store_node")
    workflow.add_edge("store_node", "performance_node")
    workflow.add_edge("performance_node", END)

    # Checkpointing and store handling
    if checkpoint:
        app = workflow.compile(checkpointer=checkpointer,store=store)
    else:
        app = workflow.compile(store=store)
    return app
