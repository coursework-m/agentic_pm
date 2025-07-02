def generate_graphviz():
    """Generates a Graphviz DOT representation of the workflow."""
    dot = [
        'digraph Workflow {',
        '    rankdir=LR;',
        '    node [shape=box];'
    ]
    # Nodes
    nodes = [
        "memory_node", "data_node", "analyst_node", "researcher_node",
        "tool_node", "trader_node", "store_node", "portfolio_node", "router_node"
    ]
    for node in nodes:
        dot.append(f'    {node};')
    dot.append('    END [shape=doublecircle];')

    # Edges
    dot.append('    memory_node -> data_node;')
    dot.append('    data_node -> analyst_node;')
    dot.append('    analyst_node -> researcher_node;')
    dot.append('    researcher_node -> router_node;')
    dot.append('    router_node -> tool_node [label="tool_node"];')
    dot.append('    router_node -> trader_node [label="trader_node"];')
    dot.append('    tool_node -> researcher_node;')
    dot.append('    trader_node -> store_node;')
    dot.append('    store_node -> portfolio_node;')
    dot.append('    portfolio_node -> END;')

    dot.append('}')
    return '\n'.join(dot)
