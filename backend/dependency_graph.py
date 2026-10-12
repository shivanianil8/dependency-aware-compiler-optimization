"""
backend/dependency_graph.py - Optimization Dependence Graph (ODG)

This module implements:
1. Pass representation as graph nodes with:
   - ID (unique identifier)
   - Label (display name)
   - Description (human-readable explanation of the pass)
   - Status (e.g., 'applicable', 'scheduled')
2. A documented, conservative dependency model for compiler passes:
   - Constant Folding -> Algebraic Simplification
   - Algebraic Simplification -> Dead Code Elimination
3. Graph construction using NetworkX (nodes, directed edges, and reasons).
4. Deterministic topological sorting with stable tie-breaking rules.
5. Cycle detection and clear exception handling.
6. Cytoscape.js-compatible JSON serialization.
7. Structured optimization pipeline planner.
8. Comprehensive built-in test suite.

Note:
The dependency edges represent explicit scheduling assumptions designed for this
educational implementation. They capture how earlier passes create opportunities for
later passes. They are not universal compiler rules, nor does every pair of passes require an edge.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
import networkx as nx


# ============================================================================
# 1. Custom Exceptions
# ============================================================================

class DependencyGraphError(Exception):
    """Base exception for all dependency graph errors."""
    pass


class CyclicDependencyError(DependencyGraphError):
    """Raised when a cycle is detected in the optimization dependence graph."""
    pass


class InvalidPassError(DependencyGraphError):
    """Raised when an unknown optimization pass name is provided."""
    pass


# ============================================================================
# 2. Supported Optimization Passes & Metadata
# ============================================================================

SUPPORTED_PASSES: Dict[str, Dict[str, str]] = {
    "constant_folding": {
        "id": "constant_folding",
        "label": "Constant Folding",
        "description": "Evaluates constant arithmetic expressions at compile time to produce static values.",
        "status": "applicable",
    },
    "algebraic_simplification": {
        "id": "algebraic_simplification",
        "label": "Algebraic Simplification",
        "description": "Applies mathematical identities (e.g., x + 0, x * 1, x * 0) to simplify expressions.",
        "status": "applicable",
    },
    "dead_code_elimination": {
        "id": "dead_code_elimination",
        "label": "Dead Code Elimination",
        "description": "Removes unused variable assignments while preserving print statements and live dependencies.",
        "status": "applicable",
    },
}

# Mapping aliases and display names to canonical pass IDs
PASS_ALIAS_TO_ID: Dict[str, str] = {
    "constant_folding": "constant_folding",
    "constant folding": "constant_folding",
    "Constant Folding": "constant_folding",
    "cf": "constant_folding",
    "algebraic_simplification": "algebraic_simplification",
    "algebraic simplification": "algebraic_simplification",
    "Algebraic Simplification": "algebraic_simplification",
    "as": "algebraic_simplification",
    "dead_code_elimination": "dead_code_elimination",
    "dead code elimination": "dead_code_elimination",
    "Dead Code Elimination": "dead_code_elimination",
    "dce": "dead_code_elimination",
}

# Canonical priority order for stable tie-breaking:
# Constant Folding (0) -> Algebraic Simplification (1) -> Dead Code Elimination (2)
PASS_PRIORITY_ORDER: Dict[str, int] = {
    "constant_folding": 0,
    "algebraic_simplification": 1,
    "dead_code_elimination": 2,
}


# ============================================================================
# 3. Documented Dependency Model
# ============================================================================
# Directed Edge Convention:
# (source -> target) means "source must be scheduled BEFORE target"
# because running 'source' first enables or exposes opportunities for 'target'.

MODELED_DEPENDENCIES: List[Dict[str, str]] = [
    {
        "source": "constant_folding",
        "target": "algebraic_simplification",
        "reason": (
            "Scheduling assumption: Constant folding simplifies constant arithmetic (e.g., 3 - 3 -> 0), "
            "producing identity values (0 or 1) that enable subsequent algebraic simplification rules."
        ),
    },
    {
        "source": "algebraic_simplification",
        "target": "dead_code_elimination",
        "reason": (
            "Scheduling assumption: Algebraic simplification can drop variable references (e.g., x * 0 -> 0), "
            "which eliminates uses of earlier variables and exposes dead assignments for elimination."
        ),
    },
]


def resolve_pass_id(pass_name: str) -> str:
    """
    Normalizes a pass name or alias into its canonical pass ID.
    Raises InvalidPassError if the pass name is not recognized.
    """
    if not isinstance(pass_name, str):
        raise InvalidPassError(f"Pass name must be a string, received {type(pass_name).__name__}.")

    clean_name = pass_name.strip()
    if clean_name in PASS_ALIAS_TO_ID:
        return PASS_ALIAS_TO_ID[clean_name]

    valid_names = sorted(list(set(PASS_ALIAS_TO_ID.values())))
    valid_labels = [SUPPORTED_PASSES[pid]["label"] for pid in valid_names]
    raise InvalidPassError(
        f"Unknown optimization pass '{pass_name}'. "
        f"Supported passes are: {', '.join(valid_labels)}."
    )


def get_pass_metadata(pass_name: str) -> Dict[str, str]:
    """Returns metadata dictionary for a given pass name or alias."""
    canonical_id = resolve_pass_id(pass_name)
    return dict(SUPPORTED_PASSES[canonical_id])


# ============================================================================
# 4. Graph Construction
# ============================================================================

def build_dependency_graph(applicable_passes: List[str]) -> nx.DiGraph:
    """
    Builds a directed Optimization Dependence Graph (ODG) containing only the
    applicable passes and their relevant modeled dependency edges.

    Parameters:
        applicable_passes: A list of pass names (strings). Can be canonical IDs
                           (e.g., 'constant_folding') or labels (e.g., 'Constant Folding').

    Returns:
        A NetworkX DiGraph where:
        - Nodes represent applicable passes with attributes {id, label, description, status}.
        - Directed edges represent ordering dependencies with attributes {id, source, target, reason}.
    """
    graph = nx.DiGraph()

    if not applicable_passes:
        return graph

    # Validate and resolve all pass names first
    resolved_ids: Set[str] = set()
    for name in applicable_passes:
        canonical_id = resolve_pass_id(name)
        resolved_ids.add(canonical_id)

    # Add relevant nodes
    for pid in resolved_ids:
        meta = SUPPORTED_PASSES[pid]
        graph.add_node(
            pid,
            id=meta["id"],
            label=meta["label"],
            description=meta["description"],
            status=meta["status"],
        )

    # Add only the modeled edges whose source and target are both present in the graph
    for dep in MODELED_DEPENDENCIES:
        src = dep["source"]
        dst = dep["target"]
        if src in resolved_ids and dst in resolved_ids:
            edge_id = f"edge_{src}_{dst}"
            graph.add_edge(
                src,
                dst,
                id=edge_id,
                source=src,
                target=dst,
                reason=dep["reason"],
            )

    return graph


# ============================================================================
# 5. Deterministic Topological Sorting & Cycle Detection
# ============================================================================

def get_pass_order(graph: nx.DiGraph, as_ids: bool = False) -> List[str]:
    """
    Computes a deterministic topological order for the given dependency graph.

    Tie-breaking rule:
      When multiple passes have in-degree 0 (all prerequisite dependencies satisfied),
      ties are broken deterministically using the canonical pass priority:
        1. Constant Folding (priority 0)
        2. Algebraic Simplification (priority 1)
        3. Dead Code Elimination (priority 2)
      Followed by alphabetical order on node ID as a stable fallback.

    Parameters:
        graph: NetworkX DiGraph representing the pass dependencies.
        as_ids: If True, returns list of canonical IDs; otherwise returns display labels.

    Returns:
        A list of pass names in valid execution order.

    Raises:
        CyclicDependencyError: If a directed cycle exists in the graph.
    """
    if graph.number_of_nodes() == 0:
        return []

    # Check for cycles
    if not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph, orientation="original")
            cycle_nodes = [u for u, v, _ in cycle]
            cycle_nodes.append(cycle[0][0])
            cycle_names = [graph.nodes[n].get("label", n) for n in cycle_nodes]
            cycle_str = " -> ".join(cycle_names)
        except Exception:
            cycle_str = "Cycle detected in graph"
        raise CyclicDependencyError(
            f"Cyclic dependency detected: {cycle_str}. A valid topological pass order does not exist."
        )

    # Deterministic Kahn's Algorithm
    in_degrees: Dict[str, int] = {node: graph.in_degree(node) for node in graph.nodes()}

    def tie_break_key(node_id: str) -> Tuple[int, str]:
        return (PASS_PRIORITY_ORDER.get(node_id, 999), node_id)

    # Collect nodes with in-degree 0
    ready = [node for node, deg in in_degrees.items() if deg == 0]
    ready.sort(key=tie_break_key)

    ordered_ids: List[str] = []

    while ready:
        curr = ready.pop(0)
        ordered_ids.append(curr)

        for succ in graph.successors(curr):
            in_degrees[succ] -= 1
            if in_degrees[succ] == 0:
                ready.append(succ)
                ready.sort(key=tie_break_key)

    if len(ordered_ids) != graph.number_of_nodes():
        raise CyclicDependencyError("Unable to compute topological sort due to cycle.")

    if as_ids:
        return ordered_ids
    return [graph.nodes[n].get("label", n) for n in ordered_ids]


# ============================================================================
# 6. Cytoscape.js JSON Serialization
# ============================================================================

def serialize_graph(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Serializes a NetworkX DiGraph into JSON-compatible format for Cytoscape.js.

    Output format:
      {
        "nodes": [{"data": {"id": ..., "label": ..., "status": ..., "description": ...}}],
        "edges": [{"data": {"id": ..., "source": ..., "target": ..., "reason": ...}}],
        "elements": {
          "nodes": [...],
          "edges": [...]
        }
      }
    """
    nodes_list: List[Dict[str, Any]] = []
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        nodes_list.append({
            "data": {
                "id": str(node_id),
                "label": node_data.get("label", str(node_id)),
                "status": node_data.get("status", "applicable"),
                "description": node_data.get("description", ""),
            }
        })

    edges_list: List[Dict[str, Any]] = []
    for u, v in graph.edges():
        edge_data = graph.edges[u, v]
        edge_id = edge_data.get("id", f"edge_{u}_{v}")
        edges_list.append({
            "data": {
                "id": edge_id,
                "source": str(u),
                "target": str(v),
                "reason": edge_data.get("reason", ""),
            }
        })

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "elements": {
            "nodes": nodes_list,
            "edges": edges_list,
        },
    }


# ============================================================================
# 7. Pipeline Plan Generation
# ============================================================================

def get_optimization_plan(applicable_passes: List[str]) -> Dict[str, Any]:
    """
    Builds the graph, determines the deterministic pass order, and returns
    a comprehensive dictionary with dependency details and Cytoscape serialization.

    Returns:
        Structured dictionary:
        - applicable_passes: list of applicable pass labels
        - pass_order: calculated deterministic topological order
        - pass_order_ids: canonical IDs for the pass order
        - dependencies: list of explicit dependency relationships with reasons
        - graph: Cytoscape.js serialized graph
        - is_acyclic: boolean indicating whether graph is a valid DAG
        - node_count: number of nodes
        - edge_count: number of edges
    """
    graph = build_dependency_graph(applicable_passes)
    pass_order = get_pass_order(graph, as_ids=False)
    pass_order_ids = get_pass_order(graph, as_ids=True)
    serialized = serialize_graph(graph)

    dependencies_list: List[Dict[str, str]] = []
    for u, v in graph.edges():
        dependencies_list.append({
            "source_id": u,
            "target_id": v,
            "source_label": graph.nodes[u].get("label", u),
            "target_label": graph.nodes[v].get("label", v),
            "reason": graph.edges[u, v].get("reason", ""),
        })

    return {
        "applicable_passes": [get_pass_metadata(p)["label"] for p in applicable_passes],
        "pass_order": pass_order,
        "pass_order_ids": pass_order_ids,
        "dependencies": dependencies_list,
        "graph": serialized,
        "is_acyclic": nx.is_directed_acyclic_graph(graph),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }


# ============================================================================
# 8. Self-Contained Test Suite
# ============================================================================

def run_tests() -> None:
    """Runs tests covering all specified requirements."""
    print("=" * 65)
    print("Dependency-Aware Compiler Optimization - Dependency Graph Tests")
    print("=" * 65)

    # Test 1: Empty input
    print("\n[Test 1] Empty Input Handling")
    g_empty = build_dependency_graph([])
    assert g_empty.number_of_nodes() == 0
    assert g_empty.number_of_edges() == 0
    order_empty = get_pass_order(g_empty)
    assert order_empty == []
    serialized_empty = serialize_graph(g_empty)
    assert serialized_empty["nodes"] == []
    assert serialized_empty["edges"] == []
    print("  -> Passed: Empty list correctly yields empty graph and order.")

    # Test 2: Graph creation with all 3 passes
    print("\n[Test 2] Full Graph Creation & Edge Count")
    all_passes = ["Constant Folding", "Algebraic Simplification", "Dead Code Elimination"]
    g_full = build_dependency_graph(all_passes)
    assert g_full.number_of_nodes() == 3
    assert g_full.number_of_edges() == 2
    assert g_full.has_edge("constant_folding", "algebraic_simplification")
    assert g_full.has_edge("algebraic_simplification", "dead_code_elimination")
    assert not g_full.has_edge("constant_folding", "dead_code_elimination")
    print("  -> Passed: Graph has 3 nodes and exactly 2 modeled dependency edges.")

    # Test 3: Edge reasons verification
    print("\n[Test 3] Edge Reason Documentation")
    edge_cf_as = g_full.edges["constant_folding", "algebraic_simplification"]
    edge_as_dce = g_full.edges["algebraic_simplification", "dead_code_elimination"]
    assert "reason" in edge_cf_as and len(edge_cf_as["reason"]) > 10
    assert "reason" in edge_as_dce and len(edge_as_dce["reason"]) > 10
    print(f"  CF -> AS reason: {edge_cf_as['reason'][:60]}...")
    print(f"  AS -> DCE reason: {edge_as_dce['reason'][:60]}...")
    print("  -> Passed: Every modeled edge contains clear documentation.")

    # Test 4: Deterministic ordering for full graph
    print("\n[Test 4] Deterministic Topological Pass Order")
    order_labels = get_pass_order(g_full, as_ids=False)
    order_ids = get_pass_order(g_full, as_ids=True)
    assert order_labels == ["Constant Folding", "Algebraic Simplification", "Dead Code Elimination"]
    assert order_ids == ["constant_folding", "algebraic_simplification", "dead_code_elimination"]
    print(f"  Order: {order_labels}")
    print("  -> Passed: Order is strictly deterministic.")

    # Test 5: Stable tie-breaking without direct edges
    print("\n[Test 5] Stable Tie-Breaking with Independent Passes")
    # Constant Folding and Dead Code Elimination have no direct edge
    g_sub = build_dependency_graph(["Dead Code Elimination", "Constant Folding"])
    assert g_sub.number_of_nodes() == 2
    assert g_sub.number_of_edges() == 0  # no direct modeled edge between them
    order_sub = get_pass_order(g_sub)
    # Tie-breaking priority puts Constant Folding first
    assert order_sub == ["Constant Folding", "Dead Code Elimination"]
    print(f"  Subset order: {order_sub}")
    print("  -> Passed: Tie-breaking priority reliably orders independent passes.")

    # Test 6: Cycle detection
    print("\n[Test 6] Cycle Detection")
    g_cyclic = nx.DiGraph()
    g_cyclic.add_node("constant_folding", label="Constant Folding")
    g_cyclic.add_node("dead_code_elimination", label="Dead Code Elimination")
    # Introduce an intentional cycle
    g_cyclic.add_edge("constant_folding", "dead_code_elimination", reason="test forward")
    g_cyclic.add_edge("dead_code_elimination", "constant_folding", reason="test reverse cycle")
    try:
        get_pass_order(g_cyclic)
        assert False, "Should have raised CyclicDependencyError!"
    except CyclicDependencyError as e:
        print(f"  -> Caught expected cycle exception: {e}")
        print("  -> Passed: Cycle detected and descriptive exception raised.")

    # Test 7: Unknown pass rejection
    print("\n[Test 7] Unknown Pass Name Rejection")
    try:
        build_dependency_graph(["NonExistentPass"])
        assert False, "Should have raised InvalidPassError!"
    except InvalidPassError as e:
        print(f"  -> Caught expected invalid pass error: {e}")
        print("  -> Passed: Unknown pass name rejected cleanly.")

    # Test 8: Cytoscape.js serialization
    print("\n[Test 8] Cytoscape Serialization Format")
    serialized = serialize_graph(g_full)
    assert "nodes" in serialized and "edges" in serialized and "elements" in serialized
    assert len(serialized["nodes"]) == 3
    assert len(serialized["edges"]) == 2
    # Verify node schema
    first_node = serialized["nodes"][0]["data"]
    assert "id" in first_node and "label" in first_node and "status" in first_node and "description" in first_node
    # Verify edge schema
    first_edge = serialized["edges"][0]["data"]
    assert "id" in first_edge and "source" in first_edge and "target" in first_edge and "reason" in first_edge
    print(f"  Sample node: {first_node}")
    print(f"  Sample edge: {first_edge}")
    print("  -> Passed: Serialized structure strictly satisfies Cytoscape.js format.")

    # Test 9: Complete optimization plan function
    print("\n[Test 9] Optimization Plan Generation")
    plan = get_optimization_plan(["Dead Code Elimination", "Constant Folding", "Algebraic Simplification"])
    assert plan["is_acyclic"] is True
    assert plan["pass_order"] == ["Constant Folding", "Algebraic Simplification", "Dead Code Elimination"]
    assert len(plan["dependencies"]) == 2
    assert plan["node_count"] == 3
    assert plan["edge_count"] == 2
    print("  Plan pass order:", plan["pass_order"])
    print("  Dependencies count:", len(plan["dependencies"]))
    print("  -> Passed: Optimization plan generated successfully.")

    print("\n" + "=" * 65)
    print("All dependency graph tests completed and passed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
