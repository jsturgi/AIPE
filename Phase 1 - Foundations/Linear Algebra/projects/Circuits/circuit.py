import numpy as np
from components import Node, Resistor, VoltageSource, CurrentSource

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
class Circuit:
    def __init__(self):
        self.components = []
    def add(self, components: list):
        for component in components:
            self.components.append(component)
    
    def solve(self, ground):
        nodes = set()
        voltage_sources = []
        for component in self.components:
            nodes.add(component.positive_node)
            nodes.add(component.negative_node)
            if isinstance(component, VoltageSource):
                voltage_sources.append(component)
        nodes.discard(ground)
        node_to_index = {}
        for i, node in enumerate(nodes):
            node_to_index[node] = i
        n = len(nodes)
        volt_to_index = {}
        for i, vsource in enumerate(voltage_sources, start=n):
            volt_to_index[vsource] = i
            
        A = np.zeros((n+len(voltage_sources), n+len(voltage_sources)))
        b = np.zeros(n+len(voltage_sources))
        for component in self.components:
            if isinstance(component, Resistor):
                # get indices for this resistor's nodes
                i = node_to_index.get(component.positive_node)
                j = node_to_index.get(component.negative_node)
                conductance = 1 / component.resistance
                if i is not None and j is not None:
                    # both nodes in matrix
                    A[i][i] += conductance
                    A[j][j] += conductance
                    A[i][j] -= conductance
                    A[j][i] -= conductance
                elif i is not None:
                    # only positive_node in matrix
                    A[i][i] += conductance
                elif j is not None:
                    # only negative_node in matrix
                    A[j][j] += conductance
            if isinstance(component, CurrentSource):
                i = node_to_index.get(component.positive_node)
                j = node_to_index.get(component.negative_node)
                if i is not None:
                    b[i] += component.current
                if j is not None:
                    b[j] -= component.current
            if isinstance(component, VoltageSource):
                i = node_to_index.get(component.positive_node)
                j = node_to_index.get(component.negative_node)
                vs_idx = volt_to_index.get(component)
                if i is not None and j is not None:
                    A[i][vs_idx] += 1
                    A[vs_idx][i] += 1
                    A[j][vs_idx] -= 1
                    A[vs_idx][j] -= 1
                    b[vs_idx] = component.voltage
                elif i is not None:
                    A[i][vs_idx] += 1
                    A[vs_idx][i] += 1
                    b[vs_idx] = component.voltage
                elif j is not None:
                    A[j][vs_idx] -= 1
                    A[vs_idx][j] -= 1
                    b[vs_idx] = component.voltage
                    
        x = np.linalg.solve(A,b)
        for node, index in node_to_index.items():
            node.voltage = x[index]
        for volt, index in volt_to_index.items():
            volt._current = x[index]
        ground.voltage = 0

    def power_summary(self):
        resistor_power = 0
        volt_source_power = 0
        current_source_power = 0
        for component in self.components:
            if isinstance(component, Resistor):
                p = (component.voltage()**2) / component.resistance
                resistor_power += p
            elif isinstance(component, VoltageSource):
                p = component.voltage * component._current
                volt_source_power += p
            elif isinstance(component, CurrentSource):
                p = component.voltage() * component.current
                current_source_power += p
        # Power conservation: delivered = dissipated
        # In MNA convention, source power is negative when delivering
        # So: -volt_source_power - current_source_power = resistor_power
        # Balance should be ~0 when: delivered - dissipated = 0
        balance = (-volt_source_power - current_source_power) - resistor_power

        return (
            f"Power Summary:\n"
            f"    Resistors:       {resistor_power:.1f} W (dissipated)\n"
            f"    Voltage Sources: {-volt_source_power:.1f} W (delivered)\n"
            f"    Current Sources: {-current_source_power:.1f} W\n"
            f"    Balance:         {balance:.1f} W (should be ~0)"
        )
    
    def kcl_summary(self):
        nodes = set()
        for component in self.components:
            nodes.add(component.positive_node)
            nodes.add(component.negative_node)
        node_sums = {node: 0.0 for node in nodes}
        for component in self.components:
            if isinstance(component, Resistor):
                node_sums[component.positive_node] -= component.current()
                node_sums[component.negative_node] += component.current()
            elif isinstance(component, VoltageSource):
                if component._current is not None:
                    # _current is current INTO + terminal (negative when delivering)
                    # Current INTO node from Vs = -_current at positive terminal
                    node_sums[component.positive_node] -= component._current
                    node_sums[component.negative_node] += component._current
            elif isinstance(component, CurrentSource):
                node_sums[component.positive_node] += component.current
                node_sums[component.negative_node] -= component.current

        lines = ["KCL Check:"]
        for node, current_sum in node_sums.items():
            lines.append(f"    {node.id}:".ljust(14) + f"{current_sum: .3f} A")

        return "\n".join(lines)

    def node_summary(self):
        """Return a formatted string of all node voltages."""
        nodes = set()
        for component in self.components:
            nodes.add(component.positive_node)
            nodes.add(component.negative_node)

        lines = ["Node Voltages:"]
        for node in sorted(nodes, key=lambda n: str(n.id)):
            v = node.voltage if node.voltage is not None else float('nan')
            lines.append(f"    {node.id}:".ljust(14) + f"{v: .3f} V")

        return "\n".join(lines)

    def component_summary(self):
        """Return a formatted string of all component values."""
        lines = ["Component Values:"]
        for i, comp in enumerate(self.components):
            if isinstance(comp, Resistor):
                v = comp.voltage()
                i_val = comp.current()
                p = v * i_val
                lines.append(
                    f"    R ({comp.resistance:.0f} ohm): "
                    f"{v:.3f} V, {i_val*1000:.3f} mA, {p*1000:.3f} mW"
                )
            elif isinstance(comp, VoltageSource):
                i_val = comp._current if comp._current else 0
                p = comp.voltage * i_val
                lines.append(
                    f"    Vs ({comp.voltage:.1f} V): "
                    f"{-i_val*1000:.3f} mA, {-p*1000:.3f} mW delivered"
                )
            elif isinstance(comp, CurrentSource):
                v = comp.voltage()
                p = v * comp.current
                lines.append(
                    f"    Is ({comp.current*1000:.3f} mA): "
                    f"{v:.3f} V, {p*1000:.3f} mW delivered"
                )

        return "\n".join(lines)

    def draw(self, figsize=(10, 8), with_values=True, layout='spring'):
        """
        Draw the circuit topology using NetworkX.

        Parameters:
            figsize: tuple, figure size (width, height)
            with_values: bool, show voltage/current values on edges
            layout: str, 'spring', 'circular', 'shell', or 'kamada_kawai'

        Returns:
            fig, ax: matplotlib figure and axes objects
        """
        if not HAS_VISUALIZATION:
            raise ImportError(
                "Visualization requires networkx and matplotlib. "
                "Install with: pip install networkx matplotlib"
            )

        G = nx.MultiGraph()

        # Collect all nodes
        nodes = set()
        for comp in self.components:
            nodes.add(comp.positive_node)
            nodes.add(comp.negative_node)

        # Add nodes to graph
        for node in nodes:
            v = node.voltage if node.voltage is not None else 0
            G.add_node(node.id, voltage=v)

        # Add edges for each component
        edge_labels = {}
        for comp in self.components:
            n1 = comp.positive_node.id
            n2 = comp.negative_node.id

            if isinstance(comp, Resistor):
                label = f"R={comp.resistance:.0f}Ω"
                if with_values and comp.positive_node.voltage is not None:
                    i = comp.current() * 1000  # mA
                    label += f"\n{i:.2f}mA"
                G.add_edge(n1, n2, component='resistor', label=label)
                edge_labels[(n1, n2)] = label

            elif isinstance(comp, VoltageSource):
                label = f"Vs={comp.voltage:.1f}V"
                if with_values and comp._current is not None:
                    i = -comp._current * 1000  # mA
                    label += f"\n{i:.2f}mA"
                G.add_edge(n1, n2, component='voltage_source', label=label)
                edge_labels[(n1, n2)] = label

            elif isinstance(comp, CurrentSource):
                label = f"Is={comp.current*1000:.2f}mA"
                if with_values and comp.positive_node.voltage is not None:
                    v = comp.voltage()
                    label += f"\n{v:.2f}V"
                G.add_edge(n1, n2, component='current_source', label=label)
                edge_labels[(n1, n2)] = label

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        node_colors = []
        for node_id in G.nodes():
            v = G.nodes[node_id].get('voltage', 0)
            if v == 0:
                node_colors.append('lightgreen')  # Ground
            else:
                node_colors.append('lightblue')

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=700,
            edgecolors='black'
        )

        # Draw node labels with voltage
        if with_values:
            labels = {
                n: f"{n}\n{G.nodes[n]['voltage']:.2f}V"
                for n in G.nodes()
            }
        else:
            labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray',
            width=2,
            connectionstyle='arc3,rad=0.1'
        )

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, ax=ax,
            font_size=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

        ax.set_title("Circuit Topology", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        return fig, ax