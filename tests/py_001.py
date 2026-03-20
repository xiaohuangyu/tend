from collections import deque
import unittest


def is_bipartite(graph: list[list[int]]) -> bool:
    n = len(graph)
    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor < 0 or neighbor >= n:
                    raise ValueError("graph contains an invalid vertex index")
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True


class TestIsBipartite(unittest.TestCase):
    def test_connected_bipartite(self) -> None:
        graph = [
            [1, 3],
            [0, 2],
            [1, 3],
            [0, 2],
        ]
        self.assertTrue(is_bipartite(graph))

    def test_non_bipartite_odd_cycle(self) -> None:
        graph = [
            [1, 2],
            [0, 2],
            [0, 1],
        ]
        self.assertFalse(is_bipartite(graph))

    def test_disconnected_graph(self) -> None:
        graph = [
            [1],
            [0],
            [3],
            [2],
            [],
        ]
        self.assertTrue(is_bipartite(graph))

    def test_self_loop(self) -> None:
        graph = [
            [0],
        ]
        self.assertFalse(is_bipartite(graph))

    def test_invalid_vertex_index(self) -> None:
        graph = [
            [1],
            [2],
        ]
        with self.assertRaises(ValueError):
            is_bipartite(graph)


if __name__ == "__main__":
    unittest.main()
