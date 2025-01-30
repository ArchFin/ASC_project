import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

# Define a simple neural network class
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Create a network
network = SimpleNeuralNetwork(input_size=5, hidden_size=8, output_size=3)

# Visualise connectivity
def visualise_connectivity(network):
    G = nx.DiGraph()  # Directed graph for the connections
    
    layers = [network.hidden_layer, network.output_layer]
    layer_sizes = [5, 8, 3]  # Input, hidden, output sizes
    
    # Create nodes for each layer
    node_offset = 0
    node_positions = {}
    for layer_idx, size in enumerate(layer_sizes):
        for node in range(size):
            node_id = node + node_offset
            G.add_node(node_id, layer=layer_idx)
            node_positions[node_id] = (layer_idx, -node)  # x = layer_idx, y = -node
        node_offset += size
    
    # Add edges based on weight matrices
    offset_from, offset_to = 0, layer_sizes[0]
    for i, layer in enumerate(layers):
        weights = layer.weight.detach().numpy()
        for from_node in range(weights.shape[1]):  # Iterate over input nodes
            for to_node in range(weights.shape[0]):  # Iterate over output nodes
                weight = weights[to_node, from_node]
                if abs(weight) > 0.01:  # Ignore very small weights
                    G.add_edge(from_node + offset_from, to_node + offset_to, weight=weight)
        offset_from += layer_sizes[i]
        offset_to += layer_sizes[i + 1]

    # Draw the graph
    plt.figure(figsize=(10, 6))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos=node_positions, with_labels=False, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos=node_positions, labels={n: f"{n}" for n in G.nodes})
    nx.draw_networkx_edge_labels(
        G, pos=node_positions, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    )
    plt.title("Neural Network Connectivity")
    plt.axis("off")
    plt.show()

# Visualise the created network
visualise_connectivity(network)
