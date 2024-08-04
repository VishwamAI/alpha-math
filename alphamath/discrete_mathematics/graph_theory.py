import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_vertex(self, vertex):
        """Add a vertex to the graph."""
        self.graph.add_node(vertex)

    def add_edge(self, vertex1, vertex2):
        """Add an edge between two vertices in the graph."""
        self.graph.add_edge(vertex1, vertex2)

    def remove_vertex(self, vertex):
        """Remove a vertex and all its incident edges from the graph."""
        self.graph.remove_node(vertex)

    def remove_edge(self, vertex1, vertex2):
        """Remove an edge between two vertices in the graph."""
        self.graph.remove_edge(vertex1, vertex2)

    def get_vertices(self):
        """Return all vertices in the graph."""
        return list(self.graph.nodes())

    def get_edges(self):
        """Return all edges in the graph."""
        return list(self.graph.edges())

    def get_degree(self, vertex):
        """Return the degree of a vertex."""
        return self.graph.degree[vertex]

    def is_connected(self):
        """Check if the graph is connected."""
        return nx.is_connected(self.graph)

    def find_shortest_path(self, start, end):
        """Find the shortest path between two vertices."""
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return None

    def get_connected_components(self):
        """Return all connected components in the graph."""
        return list(nx.connected_components(self.graph))

    def is_cyclic(self):
        """Check if the graph contains a cycle."""
        return len(list(nx.simple_cycles(self.graph))) > 0

    def get_minimum_spanning_tree(self):
        """Return the minimum spanning tree of the graph."""
        return nx.minimum_spanning_tree(self.graph)

    def visualize(self):
        """Visualize the graph using matplotlib."""
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
        plt.title("Graph Visualization")
        plt.show()

# Example usage
if __name__ == "__main__":
    g = Graph()
    g.add_vertex(1)
    g.add_vertex(2)
    g.add_vertex(3)
    g.add_vertex(4)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 1)

    print("Vertices:", g.get_vertices())
    print("Edges:", g.get_edges())
    print("Is connected:", g.is_connected())
    print("Shortest path from 1 to 3:", g.find_shortest_path(1, 3))
    print("Is cyclic:", g.is_cyclic())

    g.visualize()
