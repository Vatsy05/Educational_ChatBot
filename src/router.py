# src/router.py

def route_query(query: str) -> str:
    """
    Routes a query to a domain.
    Returns: 'dsa', 'os', or 'general'
    """

    q = query.lower()

    dsa_keywords = [
        "sort", "sorting", "merge", "quick", "heap",
        "tree", "bst", "binary search tree",
        "graph", "dfs", "bfs", "dijkstra",
        "algorithm", "search", "complexity"
    ]

    os_keywords = [
        "paging", "memory", "virtual",
        "deadlock", "process", "thread",
        "scheduling", "cpu",
        "semaphore", "mutex",
        "page table", "tlb"
    ]

    if any(k in q for k in dsa_keywords):
        return "dsa"

    if any(k in q for k in os_keywords):
        return "os"

    return "general"
