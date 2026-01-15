"""
Example Circuits for the Circuit Analyzer Project

This module provides pre-built example circuits demonstrating:
1. Voltage Divider - with formula verification
2. Wheatstone Bridge - balanced and unbalanced configurations
3. Complex Network - 5+ node circuit

Each example includes the circuit setup, solution, and verification.
"""

from components import Node, Resistor, VoltageSource, CurrentSource
from circuit import Circuit


def voltage_divider(vin=10.0, r1=1000, r2=2000, verbose=True):
    """
    Classic voltage divider circuit.

    Circuit:
        Vs(Vin) --> [R1] --> Vout --> [R2] --> GND

    Formula: Vout = Vin * R2 / (R1 + R2)

    Parameters:
        vin: Input voltage (default 10V)
        r1: Upper resistor value in ohms (default 1k)
        r2: Lower resistor value in ohms (default 2k)
        verbose: Print results if True

    Returns:
        dict with circuit, nodes, and verification results
    """
    # Create nodes
    gnd = Node("GND")
    n1 = Node("Vin")
    n2 = Node("Vout")

    # Create components
    vs = VoltageSource(vin, n1, gnd)
    R1 = Resistor(r1, n1, n2)
    R2 = Resistor(r2, n2, gnd)

    # Build and solve circuit
    circuit = Circuit()
    circuit.add([vs, R1, R2])
    circuit.solve(gnd)

    # Calculate theoretical value
    vout_theoretical = vin * r2 / (r1 + r2)
    vout_actual = n2.voltage
    error = abs(vout_actual - vout_theoretical)

    if verbose:
        print("=" * 50)
        print("VOLTAGE DIVIDER")
        print("=" * 50)
        print(f"\nCircuit: Vs({vin}V) --> R1({r1}Ω) --> Vout --> R2({r2}Ω) --> GND")
        print(f"\nFormula: Vout = Vin × R2/(R1+R2)")
        print(f"         Vout = {vin} × {r2}/({r1}+{r2})")
        print(f"         Vout = {vin} × {r2}/{r1+r2}")
        print(f"         Vout = {vout_theoretical:.4f} V")
        print(f"\nSimulated: Vout = {vout_actual:.4f} V")
        print(f"Error: {error:.2e} V")
        print(f"\n{circuit.node_summary()}")
        print(f"\n{circuit.component_summary()}")
        print(f"\n{circuit.power_summary()}")
        print(f"\n{circuit.kcl_summary()}")

    return {
        'circuit': circuit,
        'nodes': {'gnd': gnd, 'vin': n1, 'vout': n2},
        'components': {'vs': vs, 'r1': R1, 'r2': R2},
        'theoretical': vout_theoretical,
        'actual': vout_actual,
        'error': error,
        'verified': error < 1e-10
    }


def wheatstone_bridge(vs_value=10.0, r1=1000, r2=2000, r3=1000, r4=2000, r5=1000, verbose=True):
    """
    Wheatstone bridge circuit.

    Circuit:
                Vin
               /   \\
             R1     R2
             /       \\
           Va --R5-- Vb
             \\       /
             R3     R4
               \\   /
               GND

    Balanced when: R1/R3 = R2/R4
    Bridge voltage: Va - Vb = 0 when balanced

    Parameters:
        vs_value: Source voltage (default 10V)
        r1, r2, r3, r4: Bridge resistors
        r5: Bridge/galvanometer resistor
        verbose: Print results if True

    Returns:
        dict with circuit, nodes, and balance analysis
    """
    # Create nodes
    gnd = Node("GND")
    vin = Node("Vin")
    va = Node("Va")
    vb = Node("Vb")

    # Create components
    vs = VoltageSource(vs_value, vin, gnd)
    R1 = Resistor(r1, vin, va)
    R2 = Resistor(r2, vin, vb)
    R3 = Resistor(r3, va, gnd)
    R4 = Resistor(r4, vb, gnd)
    R5 = Resistor(r5, va, vb)  # Bridge resistor

    # Build and solve circuit
    circuit = Circuit()
    circuit.add([vs, R1, R2, R3, R4, R5])
    circuit.solve(gnd)

    # Calculate balance
    ratio1 = r1 / r3
    ratio2 = r2 / r4
    is_balanced = abs(ratio1 - ratio2) < 1e-10
    bridge_voltage = va.voltage - vb.voltage
    bridge_current = R5.current()

    if verbose:
        print("=" * 50)
        print("WHEATSTONE BRIDGE")
        print("=" * 50)
        print(f"\nR1={r1}Ω, R2={r2}Ω, R3={r3}Ω, R4={r4}Ω, R5={r5}Ω")
        print(f"\nBalance condition: R1/R3 = R2/R4")
        print(f"  R1/R3 = {r1}/{r3} = {ratio1:.4f}")
        print(f"  R2/R4 = {r2}/{r4} = {ratio2:.4f}")
        print(f"  Status: {'BALANCED' if is_balanced else 'UNBALANCED'}")
        print(f"\nBridge voltage (Va - Vb): {bridge_voltage:.6f} V")
        print(f"Bridge current through R5: {bridge_current*1000:.6f} mA")
        print(f"\n{circuit.node_summary()}")
        print(f"\n{circuit.power_summary()}")
        print(f"\n{circuit.kcl_summary()}")

    return {
        'circuit': circuit,
        'nodes': {'gnd': gnd, 'vin': vin, 'va': va, 'vb': vb},
        'components': {'vs': vs, 'r1': R1, 'r2': R2, 'r3': R3, 'r4': R4, 'r5': R5},
        'is_balanced': is_balanced,
        'bridge_voltage': bridge_voltage,
        'bridge_current': bridge_current,
        'ratio1': ratio1,
        'ratio2': ratio2
    }


def wheatstone_balanced(verbose=True):
    """Balanced Wheatstone bridge (R1/R3 = R2/R4 = 1.0)."""
    return wheatstone_bridge(
        vs_value=10.0,
        r1=1000, r2=1000,
        r3=1000, r4=1000,
        r5=1000,
        verbose=verbose
    )


def wheatstone_unbalanced(verbose=True):
    """Unbalanced Wheatstone bridge (R1/R3 ≠ R2/R4)."""
    return wheatstone_bridge(
        vs_value=10.0,
        r1=1000, r2=2000,
        r3=2000, r4=1000,
        r5=1000,
        verbose=verbose
    )


def complex_network(verbose=True):
    """
    Complex network with 5+ nodes demonstrating MNA capabilities.

    Circuit topology:
                    n1
                   /|\\
                  / | \\
               R1  R2  Vs1
                /   |   \\
              n2   n3   n4
              |\\   |   /|
              | R5 R6 R7 |
              |  \\ | /   |
              R3  \\|/   R8
              |    n5    |
              |    |     |
              +----+-----+
                  GND

    Features:
    - 5 non-ground nodes
    - Mix of voltage source and resistors
    - Multiple parallel paths
    - Demonstrates full MNA matrix construction
    """
    # Create nodes
    gnd = Node("GND")
    n1 = Node("N1")
    n2 = Node("N2")
    n3 = Node("N3")
    n4 = Node("N4")
    n5 = Node("N5")

    # Create components
    vs1 = VoltageSource(12.0, n1, gnd)  # Main supply

    # Top tier resistors
    R1 = Resistor(1000, n1, n2)   # N1 to N2
    R2 = Resistor(2000, n1, n3)   # N1 to N3
    R3 = Resistor(1500, n2, gnd)  # N2 to GND

    # Cross connections
    R4 = Resistor(3000, n1, n4)   # N1 to N4
    R5 = Resistor(2200, n2, n5)   # N2 to N5
    R6 = Resistor(1800, n3, n5)   # N3 to N5
    R7 = Resistor(2700, n4, n5)   # N4 to N5

    # Bottom tier
    R8 = Resistor(1000, n4, gnd)  # N4 to GND
    R9 = Resistor(500, n5, gnd)   # N5 to GND

    # Build and solve circuit
    circuit = Circuit()
    circuit.add([vs1, R1, R2, R3, R4, R5, R6, R7, R8, R9])
    circuit.solve(gnd)

    if verbose:
        print("=" * 50)
        print("COMPLEX NETWORK (5+ NODES)")
        print("=" * 50)
        print("\nTopology: Star-mesh network with 5 non-ground nodes")
        print("Components: 1 voltage source, 9 resistors")
        print(f"\n{circuit.node_summary()}")
        print(f"\n{circuit.component_summary()}")
        print(f"\n{circuit.power_summary()}")
        print(f"\n{circuit.kcl_summary()}")

    return {
        'circuit': circuit,
        'nodes': {'gnd': gnd, 'n1': n1, 'n2': n2, 'n3': n3, 'n4': n4, 'n5': n5},
        'components': {
            'vs1': vs1,
            'r1': R1, 'r2': R2, 'r3': R3, 'r4': R4, 'r5': R5,
            'r6': R6, 'r7': R7, 'r8': R8, 'r9': R9
        }
    }


def current_source_example(verbose=True):
    """
    Example circuit with current source.

    Circuit:
        Is(5mA) --> [R1=1k] --> [R2=2k] --> GND

    Demonstrates current source handling in MNA.
    """
    gnd = Node("GND")
    n1 = Node("N1")
    n2 = Node("N2")

    Is = CurrentSource(0.005, n1, gnd)  # 5mA
    R1 = Resistor(1000, n1, n2)
    R2 = Resistor(2000, n2, gnd)

    circuit = Circuit()
    circuit.add([Is, R1, R2])
    circuit.solve(gnd)

    # Theoretical: V_n1 = I × (R1 + R2) = 5mA × 3k = 15V
    # V_n2 = I × R2 = 5mA × 2k = 10V
    v_n1_theoretical = 0.005 * 3000
    v_n2_theoretical = 0.005 * 2000

    if verbose:
        print("=" * 50)
        print("CURRENT SOURCE EXAMPLE")
        print("=" * 50)
        print(f"\nCircuit: Is(5mA) --> R1(1kΩ) --> N2 --> R2(2kΩ) --> GND")
        print(f"\nTheoretical:")
        print(f"  V_N1 = I × (R1+R2) = 5mA × 3kΩ = {v_n1_theoretical:.1f} V")
        print(f"  V_N2 = I × R2 = 5mA × 2kΩ = {v_n2_theoretical:.1f} V")
        print(f"\nSimulated:")
        print(f"  V_N1 = {n1.voltage:.4f} V")
        print(f"  V_N2 = {n2.voltage:.4f} V")
        print(f"\n{circuit.power_summary()}")
        print(f"\n{circuit.kcl_summary()}")

    return {
        'circuit': circuit,
        'nodes': {'gnd': gnd, 'n1': n1, 'n2': n2},
        'components': {'is': Is, 'r1': R1, 'r2': R2},
        'theoretical': {'n1': v_n1_theoretical, 'n2': v_n2_theoretical},
        'verified': (
            abs(n1.voltage - v_n1_theoretical) < 1e-10 and
            abs(n2.voltage - v_n2_theoretical) < 1e-10
        )
    }


def mixed_sources_example(verbose=True):
    """
    Example with both voltage and current sources.

    Demonstrates superposition principle in circuit analysis.
    """
    gnd = Node("GND")
    n1 = Node("N1")
    n2 = Node("N2")

    Vs = VoltageSource(10.0, n1, gnd)
    Is = CurrentSource(0.002, n2, gnd)  # 2mA into N2
    R1 = Resistor(1000, n1, n2)
    R2 = Resistor(2000, n2, gnd)

    circuit = Circuit()
    circuit.add([Vs, Is, R1, R2])
    circuit.solve(gnd)

    if verbose:
        print("=" * 50)
        print("MIXED SOURCES EXAMPLE")
        print("=" * 50)
        print("\nCircuit: Vs(10V) at N1, Is(2mA) at N2, R1(1k) N1-N2, R2(2k) N2-GND")
        print(f"\n{circuit.node_summary()}")
        print(f"\n{circuit.component_summary()}")
        print(f"\n{circuit.power_summary()}")
        print(f"\n{circuit.kcl_summary()}")

    return {
        'circuit': circuit,
        'nodes': {'gnd': gnd, 'n1': n1, 'n2': n2},
        'components': {'vs': Vs, 'is': Is, 'r1': R1, 'r2': R2}
    }


def run_all_examples():
    """Run all example circuits."""
    print("\n" + "=" * 60)
    print(" CIRCUIT ANALYZER - EXAMPLE CIRCUITS")
    print("=" * 60)

    print("\n\n")
    voltage_divider()

    print("\n\n")
    wheatstone_balanced()

    print("\n\n")
    wheatstone_unbalanced()

    print("\n\n")
    complex_network()

    print("\n\n")
    current_source_example()

    print("\n\n")
    mixed_sources_example()

    print("\n\n" + "=" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
