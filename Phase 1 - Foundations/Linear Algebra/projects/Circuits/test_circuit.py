"""
Tests for Circuit Analyzer Project

Tests cover:
- Component classes (Node, Resistor, VoltageSource, CurrentSource)
- Circuit class and MNA solver
- Integration tests with real circuit configurations
"""

import pytest
import numpy as np
from components import Node, Resistor, VoltageSource, CurrentSource
from circuit import Circuit


# =============================================================================
# Node Tests
# =============================================================================

class TestNode:
    """Tests for the Node class."""

    def test_node_initialization(self):
        """Node should initialize with id and None voltage."""
        node = Node("n1")
        assert node.id == "n1"
        assert node.voltage is None

    def test_node_id_can_be_any_type(self):
        """Node id can be string, int, or other hashable type."""
        node_str = Node("input")
        node_int = Node(1)
        node_tuple = Node(("a", 1))

        assert node_str.id == "input"
        assert node_int.id == 1
        assert node_tuple.id == ("a", 1)

    def test_node_voltage_assignment(self):
        """Node voltage can be set after initialization."""
        node = Node("n1")
        node.voltage = 5.0
        assert node.voltage == 5.0

    def test_node_reset(self):
        """Reset should clear the voltage to None."""
        node = Node("n1")
        node.voltage = 10.0
        node.reset()
        assert node.voltage is None

    def test_node_reset_when_already_none(self):
        """Reset on unset node should not raise error."""
        node = Node("n1")
        node.reset()  # Should not raise
        assert node.voltage is None


# =============================================================================
# Resistor Tests
# =============================================================================

class TestResistor:
    """Tests for the Resistor class."""

    def test_resistor_initialization(self):
        """Resistor should store resistance and node references."""
        n1 = Node("n1")
        n2 = Node("n2")
        r = Resistor(1000, n1, n2)

        assert r.resistance == 1000
        assert r.positive_node is n1
        assert r.negative_node is n2

    def test_resistor_voltage_calculation(self):
        """Voltage should be V+ - V-."""
        n1 = Node("n1")
        n2 = Node("n2")
        n1.voltage = 10.0
        n2.voltage = 3.0

        r = Resistor(100, n1, n2)
        assert r.voltage() == 7.0

    def test_resistor_voltage_negative(self):
        """Voltage can be negative if V- > V+."""
        n1 = Node("n1")
        n2 = Node("n2")
        n1.voltage = 2.0
        n2.voltage = 8.0

        r = Resistor(100, n1, n2)
        assert r.voltage() == -6.0

    def test_resistor_current_ohms_law(self):
        """Current should follow Ohm's Law: I = V/R."""
        n1 = Node("n1")
        n2 = Node("n2")
        n1.voltage = 10.0
        n2.voltage = 0.0

        r = Resistor(1000, n1, n2)  # 1k ohm
        assert r.current() == 0.01  # 10V / 1000 ohms = 10mA

    def test_resistor_current_with_different_resistances(self):
        """Verify current calculation with various resistance values."""
        n1 = Node("n1")
        n2 = Node("n2")
        n1.voltage = 5.0
        n2.voltage = 0.0

        # 5V across different resistances
        r1 = Resistor(100, n1, n2)
        r2 = Resistor(500, n1, n2)
        r3 = Resistor(2500, n1, n2)

        assert r1.current() == 0.05    # 50mA
        assert r2.current() == 0.01    # 10mA
        assert r3.current() == 0.002   # 2mA

    def test_resistor_zero_voltage_zero_current(self):
        """Zero voltage difference should give zero current."""
        n1 = Node("n1")
        n2 = Node("n2")
        n1.voltage = 5.0
        n2.voltage = 5.0

        r = Resistor(100, n1, n2)
        assert r.current() == 0.0


# =============================================================================
# VoltageSource Tests
# =============================================================================

class TestVoltageSource:
    """Tests for the VoltageSource class."""

    def test_voltage_source_initialization(self):
        """VoltageSource should store voltage and node references."""
        n1 = Node("n1")
        n2 = Node("n2")
        vs = VoltageSource(12.0, n1, n2)

        assert vs.voltage == 12.0
        assert vs.positive_node is n1
        assert vs.negative_node is n2

    def test_voltage_source_current_placeholder(self):
        """VoltageSource should have _current placeholder."""
        n1 = Node("n1")
        n2 = Node("n2")
        vs = VoltageSource(5.0, n1, n2)

        assert vs._current is None


# =============================================================================
# CurrentSource Tests
# =============================================================================

class TestCurrentSource:
    """Tests for the CurrentSource class."""

    def test_current_source_initialization(self):
        """CurrentSource should store current and node references."""
        n1 = Node("n1")
        n2 = Node("n2")
        cs = CurrentSource(0.001, n1, n2)  # 1mA

        assert cs.current == 0.001
        assert cs.positive_node is n1
        assert cs.negative_node is n2

    def test_current_source_voltage_calculation(self):
        """Voltage should be V+ - V-."""
        n1 = Node("n1")
        n2 = Node("n2")
        n1.voltage = 7.0
        n2.voltage = 2.0

        cs = CurrentSource(0.01, n1, n2)
        assert cs.voltage() == 5.0


# =============================================================================
# Circuit Class Tests
# =============================================================================

class TestCircuit:
    """Tests for the Circuit class."""

    def test_circuit_initialization(self):
        """Circuit should initialize with empty components list."""
        c = Circuit()
        assert c.components == []

    def test_circuit_add_single_component(self):
        """Add should append a single component."""
        c = Circuit()
        n1 = Node("n1")
        n2 = Node("n2")
        r = Resistor(100, n1, n2)

        c.add([r])
        assert len(c.components) == 1
        assert c.components[0] is r

    def test_circuit_add_multiple_components(self):
        """Add should append multiple components."""
        c = Circuit()
        n1 = Node("n1")
        n2 = Node("n2")
        n3 = Node("n3")
        r1 = Resistor(100, n1, n2)
        r2 = Resistor(200, n2, n3)

        c.add([r1, r2])
        assert len(c.components) == 2

    def test_circuit_add_called_multiple_times(self):
        """Multiple add calls should accumulate components."""
        c = Circuit()
        n1 = Node("n1")
        n2 = Node("n2")
        r1 = Resistor(100, n1, n2)
        r2 = Resistor(200, n1, n2)

        c.add([r1])
        c.add([r2])
        assert len(c.components) == 2


# =============================================================================
# Circuit Solver Tests - Simple Circuits
# =============================================================================

class TestCircuitSolver:
    """Tests for the Circuit.solve() method with various circuit configurations."""

    def test_single_resistor_with_current_source(self):
        """
        Simple circuit: current source driving a single resistor.

        Circuit:
            I(1mA) --> [R=1k] --> GND

        Expected: V = IR = 0.001 * 1000 = 1V
        """
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(1000, n1, gnd)
        cs = CurrentSource(0.001, n1, gnd)  # 1mA into n1

        c = Circuit()
        c.add([r, cs])
        c.solve(gnd)

        assert gnd.voltage == 0.0
        assert pytest.approx(n1.voltage, rel=1e-9) == 1.0

    def test_voltage_divider_equal_resistors(self):
        """
        Voltage divider with equal resistors.

        Circuit:
            I --> [R1=1k] --> Vmid --> [R2=1k] --> GND

        With 2mA source:
        - Total R = 2k ohms
        - V_top = 2mA * 2k = 4V
        - V_mid = 2mA * 1k = 2V (half due to equal division)
        """
        gnd = Node("gnd")
        n1 = Node("n1")  # top node
        n2 = Node("n2")  # middle node

        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(1000, n2, gnd)
        cs = CurrentSource(0.002, n1, gnd)  # 2mA

        c = Circuit()
        c.add([r1, r2, cs])
        c.solve(gnd)

        assert gnd.voltage == 0.0
        assert pytest.approx(n1.voltage, rel=1e-9) == 4.0  # 2mA * 2k total
        assert pytest.approx(n2.voltage, rel=1e-9) == 2.0  # 2mA * 1k

    def test_voltage_divider_unequal_resistors(self):
        """
        Voltage divider with 1:3 ratio.

        Circuit:
            I(1mA) --> [R1=1k] --> Vmid --> [R2=3k] --> GND

        - Total R = 4k ohms
        - V_top = 1mA * 4k = 4V
        - V_mid = 1mA * 3k = 3V (voltage across R2)
        """
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(3000, n2, gnd)
        cs = CurrentSource(0.001, n1, gnd)

        c = Circuit()
        c.add([r1, r2, cs])
        c.solve(gnd)

        assert pytest.approx(n1.voltage, rel=1e-9) == 4.0
        assert pytest.approx(n2.voltage, rel=1e-9) == 3.0

    def test_parallel_resistors(self):
        """
        Two resistors in parallel.

        Circuit:
            I(1mA) --> [R1=1k || R2=1k] --> GND

        Equivalent resistance = 500 ohms
        V = 1mA * 500 = 0.5V
        """
        gnd = Node("gnd")
        n1 = Node("n1")

        r1 = Resistor(1000, n1, gnd)
        r2 = Resistor(1000, n1, gnd)
        cs = CurrentSource(0.001, n1, gnd)

        c = Circuit()
        c.add([r1, r2, cs])
        c.solve(gnd)

        assert pytest.approx(n1.voltage, rel=1e-9) == 0.5

    def test_three_node_network(self):
        """
        T-network with three nodes.

        Circuit:
               n1 ---[R1=1k]--- n2 ---[R2=1k]--- n3
                                |
                            [R3=1k]
                                |
                               GND

        Current source: 1mA into n1, 1mA out of n3
        """
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")
        n3 = Node("n3")

        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(1000, n2, n3)
        r3 = Resistor(1000, n2, gnd)
        cs1 = CurrentSource(0.001, n1, gnd)  # 1mA into n1
        cs2 = CurrentSource(0.001, gnd, n3)  # 1mA out of n3

        c = Circuit()
        c.add([r1, r2, r3, cs1, cs2])
        c.solve(gnd)

        # Verify KCL at n2: current in = current out
        i_r1 = (n1.voltage - n2.voltage) / 1000
        i_r2 = (n2.voltage - n3.voltage) / 1000
        i_r3 = (n2.voltage - gnd.voltage) / 1000

        assert pytest.approx(i_r1, abs=1e-12) == i_r2 + i_r3

    def test_current_conservation(self):
        """Verify Kirchhoff's Current Law holds after solving."""
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(2000, n2, gnd)
        cs = CurrentSource(0.003, n1, gnd)  # 3mA

        c = Circuit()
        c.add([r1, r2, cs])
        c.solve(gnd)

        # Current through R1 should equal current through R2
        i_r1 = r1.current()
        i_r2 = r2.current()

        assert pytest.approx(i_r1, rel=1e-9) == i_r2
        assert pytest.approx(i_r1, rel=1e-9) == 0.003

    def test_resistor_voltage_after_solve(self):
        """Resistor voltage method should work after circuit is solved."""
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(1000, n2, gnd)
        cs = CurrentSource(0.002, n1, gnd)

        c = Circuit()
        c.add([r1, r2, cs])
        c.solve(gnd)

        # Each resistor drops half the total voltage
        assert pytest.approx(r1.voltage(), rel=1e-9) == 2.0
        assert pytest.approx(r2.voltage(), rel=1e-9) == 2.0


# =============================================================================
# Voltage Source Tests - MNA Extended Matrix
# =============================================================================

class TestVoltageSources:
    """Tests for circuits with voltage sources using MNA."""

    def test_voltage_source_single_resistor(self):
        """
        Simple circuit: voltage source with single resistor.

        Circuit:
            Vs(10V) --> [R=1k] --> GND

        Expected:
        - V at positive terminal = 10V
        - Current = V/R = 10V / 1k = 10mA
        """
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(1000, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        assert gnd.voltage == 0.0
        assert pytest.approx(n1.voltage, rel=1e-9) == 10.0
        assert pytest.approx(vs._current, rel=1e-9) == -0.01  # 10mA (negative = current leaving +)

    def test_voltage_source_enforces_voltage_difference(self):
        """Voltage source should enforce exact voltage between its terminals."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(5.0, n1, gnd)
        r = Resistor(500, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        # Voltage difference must equal source voltage
        assert pytest.approx(n1.voltage - gnd.voltage, rel=1e-9) == 5.0

    def test_voltage_divider_with_voltage_source(self):
        """
        Classic voltage divider driven by voltage source.

        Circuit:
            Vs(12V) --> [R1=1k] --> Vmid --> [R2=2k] --> GND

        Formula: Vout = Vin * R2 / (R1 + R2) = 12 * 2k / 3k = 8V
        """
        gnd = Node("gnd")
        n1 = Node("n1")  # Voltage source output
        n2 = Node("n2")  # Middle node (Vout)

        vs = VoltageSource(12.0, n1, gnd)
        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(2000, n2, gnd)

        c = Circuit()
        c.add([vs, r1, r2])
        c.solve(gnd)

        assert pytest.approx(n1.voltage, rel=1e-9) == 12.0
        assert pytest.approx(n2.voltage, rel=1e-9) == 8.0  # 12 * 2/3

    def test_voltage_divider_equal_resistors_with_vs(self):
        """Voltage divider with equal resistors should give half voltage."""
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        vs = VoltageSource(10.0, n1, gnd)
        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(1000, n2, gnd)

        c = Circuit()
        c.add([vs, r1, r2])
        c.solve(gnd)

        assert pytest.approx(n2.voltage, rel=1e-9) == 5.0

    def test_voltage_source_current_calculation(self):
        """Verify voltage source current is correctly calculated."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(5.0, n1, gnd)
        r = Resistor(100, n1, gnd)  # 100 ohms

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        # Current through resistor = V/R = 5/100 = 50mA
        resistor_current = r.current()
        assert pytest.approx(resistor_current, rel=1e-9) == 0.05

        # Voltage source current should equal resistor current (opposite sign convention)
        # Current leaving the + terminal is negative in MNA convention
        assert pytest.approx(abs(vs._current), rel=1e-9) == 0.05

    def test_voltage_source_not_connected_to_ground(self):
        """
        Voltage source between two non-ground nodes.

        Circuit:
            I(1mA) --> n1 --> [Vs=5V] --> n2 --> [R=1k] --> GND

        Vs enforces V(n1) - V(n2) = 5V
        """
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        cs = CurrentSource(0.001, n1, gnd)  # 1mA into n1
        vs = VoltageSource(5.0, n1, n2)  # 5V drop from n1 to n2
        r = Resistor(1000, n2, gnd)

        c = Circuit()
        c.add([cs, vs, r])
        c.solve(gnd)

        # n2 = I * R = 1mA * 1k = 1V
        # n1 = n2 + 5V = 6V
        assert pytest.approx(n2.voltage, rel=1e-9) == 1.0
        assert pytest.approx(n1.voltage, rel=1e-9) == 6.0
        assert pytest.approx(n1.voltage - n2.voltage, rel=1e-9) == 5.0

    def test_multiple_voltage_sources_in_series(self):
        """
        Two voltage sources in series.

        Circuit:
            Vs1(5V) --> n1 --> Vs2(3V) --> n2 --> [R=1k] --> GND

        Total voltage = 5V + 3V = 8V
        """
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        vs1 = VoltageSource(5.0, n1, n2)
        vs2 = VoltageSource(3.0, n2, gnd)
        r = Resistor(1000, n1, gnd)

        c = Circuit()
        c.add([vs1, vs2, r])
        c.solve(gnd)

        assert pytest.approx(n2.voltage, rel=1e-9) == 3.0
        assert pytest.approx(n1.voltage, rel=1e-9) == 8.0

    def test_voltage_source_opposing_current_source(self):
        """
        Voltage source with parallel current source.

        Circuit:
            Vs(10V) in parallel with I(5mA) and R(1k)

        Voltage is fixed at 10V by voltage source.
        Resistor draws 10mA, current source supplies 5mA.
        Voltage source must supply remaining 5mA.
        """
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(1000, n1, gnd)
        cs = CurrentSource(0.005, n1, gnd)  # 5mA into node

        c = Circuit()
        c.add([vs, r, cs])
        c.solve(gnd)

        assert pytest.approx(n1.voltage, rel=1e-9) == 10.0

        # Resistor draws 10mA out
        # Current source supplies 5mA in
        # Voltage source supplies remaining 5mA
        resistor_current = r.current()  # 10mA
        assert pytest.approx(resistor_current, rel=1e-9) == 0.01

    def test_voltage_source_power_delivered(self):
        """Verify power delivered by voltage source."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        # Power = V * I = 10V * 100mA = 1W
        # Current is negative (leaving +), so power delivered is V * |I|
        power_delivered = vs.voltage * abs(vs._current)
        power_dissipated = r.voltage() * r.current()

        assert pytest.approx(power_delivered, rel=1e-9) == 1.0  # 1 Watt
        assert pytest.approx(power_delivered, rel=1e-9) == power_dissipated

    def test_wheatstone_bridge_with_voltage_source(self):
        """
        Wheatstone bridge circuit driven by voltage source.

        Circuit:
                    n1 (Vs+)
                   /  \\
                R1=1k  R2=2k
                 /      \\
               n2 --R5-- n3
                 \\      /
                R3=1k  R4=2k
                   \\  /
                   GND

        With R1/R3 = R2/R4 (balanced), n2 and n3 should be at same voltage.
        """
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")
        n3 = Node("n3")

        vs = VoltageSource(10.0, n1, gnd)
        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(2000, n1, n3)
        r3 = Resistor(1000, n2, gnd)
        r4 = Resistor(2000, n3, gnd)
        r5 = Resistor(1000, n2, n3)  # Bridge resistor

        c = Circuit()
        c.add([vs, r1, r2, r3, r4, r5])
        c.solve(gnd)

        # In balanced bridge, V(n2) = V(n3)
        # V(n2) = Vs * R3/(R1+R3) = 10 * 1k/2k = 5V
        # V(n3) = Vs * R4/(R2+R4) = 10 * 2k/4k = 5V
        assert pytest.approx(n2.voltage, rel=1e-9) == 5.0
        assert pytest.approx(n3.voltage, rel=1e-9) == 5.0

        # No current through bridge resistor when balanced
        bridge_current = (n2.voltage - n3.voltage) / r5.resistance
        assert pytest.approx(bridge_current, abs=1e-12) == 0.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_node_circuit(self):
        """Circuit with only ground and one other node."""
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(1000, n1, gnd)
        cs = CurrentSource(0.005, n1, gnd)

        c = Circuit()
        c.add([r, cs])
        c.solve(gnd)

        assert n1.voltage == 5.0

    def test_very_small_resistance(self):
        """Circuit with very small resistance (high conductance)."""
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(0.001, n1, gnd)  # 1 milliohm
        cs = CurrentSource(1000, n1, gnd)  # 1000A

        c = Circuit()
        c.add([r, cs])
        c.solve(gnd)

        assert pytest.approx(n1.voltage, rel=1e-9) == 1.0

    def test_large_resistance(self):
        """Circuit with large resistance values."""
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(1e9, n1, gnd)  # 1 gigaohm
        cs = CurrentSource(1e-9, n1, gnd)  # 1 nanoamp

        c = Circuit()
        c.add([r, cs])
        c.solve(gnd)

        assert pytest.approx(n1.voltage, rel=1e-6) == 1.0

    def test_multiple_current_sources_same_node(self):
        """Multiple current sources feeding same node."""
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(1000, n1, gnd)
        cs1 = CurrentSource(0.001, n1, gnd)
        cs2 = CurrentSource(0.002, n1, gnd)

        c = Circuit()
        c.add([r, cs1, cs2])
        c.solve(gnd)

        # Total current = 3mA, so V = 3mA * 1k = 3V
        assert pytest.approx(n1.voltage, rel=1e-9) == 3.0

    def test_opposing_current_sources(self):
        """Current sources in opposition."""
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(1000, n1, gnd)
        cs1 = CurrentSource(0.005, n1, gnd)  # 5mA in
        cs2 = CurrentSource(0.002, gnd, n1)  # 2mA out (reversed)

        c = Circuit()
        c.add([r, cs1, cs2])
        c.solve(gnd)

        # Net current = 5mA - 2mA = 3mA
        assert pytest.approx(n1.voltage, rel=1e-9) == 3.0


# =============================================================================
# Power Calculations (Manual verification)
# =============================================================================

class TestPowerCalculations:
    """Tests verifying power conservation in circuits."""

    def test_power_in_single_resistor(self):
        """Verify P = I²R = V²/R = VI for single resistor."""
        gnd = Node("gnd")
        n1 = Node("n1")

        r = Resistor(1000, n1, gnd)
        cs = CurrentSource(0.002, n1, gnd)  # 2mA

        c = Circuit()
        c.add([r, cs])
        c.solve(gnd)

        v = r.voltage()
        i = r.current()

        # Power dissipated in resistor
        p_vi = v * i
        p_i2r = i * i * r.resistance
        p_v2r = v * v / r.resistance

        assert pytest.approx(p_vi, rel=1e-9) == p_i2r
        assert pytest.approx(p_vi, rel=1e-9) == p_v2r
        assert pytest.approx(p_vi, rel=1e-9) == 0.004  # 4mW

    def test_power_conservation_series_circuit(self):
        """Power delivered by source equals power dissipated in resistors."""
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        r1 = Resistor(1000, n1, n2)
        r2 = Resistor(2000, n2, gnd)
        i_source = 0.001  # 1mA
        cs = CurrentSource(i_source, n1, gnd)

        c = Circuit()
        c.add([r1, r2, cs])
        c.solve(gnd)

        # Power from source = V_source * I_source
        p_source = n1.voltage * i_source

        # Power in resistors
        p_r1 = r1.voltage() * r1.current()
        p_r2 = r2.voltage() * r2.current()

        assert pytest.approx(p_source, rel=1e-9) == p_r1 + p_r2


# =============================================================================
# Power Summary Output Tests
# =============================================================================

class TestPowerSummary:
    """Tests for the power_summary method output format."""

    def test_power_summary_returns_string(self):
        """power_summary should return a formatted string."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.power_summary()
        assert isinstance(result, str)

    def test_power_summary_format(self):
        """power_summary should have correct format structure."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.power_summary()

        assert "Power Summary:" in result
        assert "Resistors:" in result
        assert "Voltage Sources:" in result
        assert "Current Sources:" in result
        assert "Balance:" in result
        assert "(dissipated)" in result
        assert "(delivered)" in result
        assert "(should be ~0)" in result

    def test_power_summary_values(self):
        """power_summary should show correct power values."""
        gnd = Node("gnd")
        n1 = Node("n1")

        # 10V across 100 ohms = 100mA, P = 1W
        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.power_summary()

        # Should show 1.0 W for both resistor and voltage source
        assert "1.0 W (dissipated)" in result
        assert "1.0 W (delivered)" in result
        assert "0.0 W (should be ~0)" in result

    def test_power_summary_with_current_source(self):
        """power_summary should include current source power."""
        gnd = Node("gnd")
        n1 = Node("n1")

        # 5mA into 1k resistor = 5V, P = 25mW
        cs = CurrentSource(0.005, n1, gnd)
        r = Resistor(1000, n1, gnd)

        c = Circuit()
        c.add([cs, r])
        c.solve(gnd)

        result = c.power_summary()

        assert "Power Summary:" in result
        assert "Current Sources:" in result


# =============================================================================
# KCL Summary Output Tests
# =============================================================================

class TestKCLSummary:
    """Tests for the kcl_summary method output format."""

    def test_kcl_summary_returns_string(self):
        """kcl_summary should return a formatted string."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.kcl_summary()
        assert isinstance(result, str)

    def test_kcl_summary_format(self):
        """kcl_summary should have correct format structure."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.kcl_summary()

        assert "KCL Check:" in result
        assert "n1:" in result
        assert "gnd:" in result
        assert "A" in result

    def test_kcl_summary_values_near_zero(self):
        """KCL sums at each node should be approximately zero."""
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        vs = VoltageSource(10.0, n1, gnd)
        r1 = Resistor(100, n1, n2)
        r2 = Resistor(200, n2, gnd)

        c = Circuit()
        c.add([vs, r1, r2])
        c.solve(gnd)

        result = c.kcl_summary()

        # All values should show 0.000 A
        assert "0.000 A" in result

    def test_kcl_summary_includes_all_nodes(self):
        """kcl_summary should include all nodes in the circuit."""
        gnd = Node("ground")
        n1 = Node("node_a")
        n2 = Node("node_b")

        vs = VoltageSource(5.0, n1, gnd)
        r1 = Resistor(100, n1, n2)
        r2 = Resistor(100, n2, gnd)

        c = Circuit()
        c.add([vs, r1, r2])
        c.solve(gnd)

        result = c.kcl_summary()

        assert "node_a:" in result
        assert "node_b:" in result
        assert "ground:" in result


# =============================================================================
# Node Summary Tests
# =============================================================================

class TestNodeSummary:
    """Tests for the node_summary method."""

    def test_node_summary_returns_string(self):
        """node_summary should return a formatted string."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.node_summary()
        assert isinstance(result, str)
        assert "Node Voltages:" in result

    def test_node_summary_includes_all_nodes(self):
        """node_summary should include all nodes."""
        gnd = Node("gnd")
        n1 = Node("n1")
        n2 = Node("n2")

        vs = VoltageSource(10.0, n1, gnd)
        r1 = Resistor(100, n1, n2)
        r2 = Resistor(100, n2, gnd)

        c = Circuit()
        c.add([vs, r1, r2])
        c.solve(gnd)

        result = c.node_summary()
        assert "gnd:" in result
        assert "n1:" in result
        assert "n2:" in result


# =============================================================================
# Component Summary Tests
# =============================================================================

class TestComponentSummary:
    """Tests for the component_summary method."""

    def test_component_summary_returns_string(self):
        """component_summary should return a formatted string."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.component_summary()
        assert isinstance(result, str)
        assert "Component Values:" in result

    def test_component_summary_includes_resistors(self):
        """component_summary should include resistor details."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.component_summary()
        assert "R (100 ohm)" in result
        assert "mA" in result
        assert "mW" in result

    def test_component_summary_includes_voltage_sources(self):
        """component_summary should include voltage source details."""
        gnd = Node("gnd")
        n1 = Node("n1")

        vs = VoltageSource(10.0, n1, gnd)
        r = Resistor(100, n1, gnd)

        c = Circuit()
        c.add([vs, r])
        c.solve(gnd)

        result = c.component_summary()
        assert "Vs (10.0 V)" in result
        assert "delivered" in result


# =============================================================================
# Examples Module Tests
# =============================================================================

class TestExamples:
    """Tests for the examples module."""

    def test_voltage_divider_formula_verification(self):
        """Voltage divider should match theoretical formula."""
        import examples
        result = examples.voltage_divider(vin=12.0, r1=1000, r2=3000, verbose=False)

        # Vout = Vin * R2 / (R1 + R2) = 12 * 3000 / 4000 = 9V
        assert result['verified']
        assert pytest.approx(result['theoretical'], rel=1e-9) == 9.0
        assert pytest.approx(result['actual'], rel=1e-9) == 9.0

    def test_wheatstone_balanced(self):
        """Balanced Wheatstone bridge should have zero bridge current."""
        import examples
        result = examples.wheatstone_balanced(verbose=False)

        assert result['is_balanced']
        assert pytest.approx(result['bridge_voltage'], abs=1e-10) == 0.0
        assert pytest.approx(result['bridge_current'], abs=1e-10) == 0.0

    def test_wheatstone_unbalanced(self):
        """Unbalanced Wheatstone bridge should have non-zero bridge current."""
        import examples
        result = examples.wheatstone_unbalanced(verbose=False)

        assert not result['is_balanced']
        assert result['bridge_voltage'] != 0.0
        assert result['bridge_current'] != 0.0

    def test_complex_network_nodes(self):
        """Complex network should have 6 nodes (5 + ground)."""
        import examples
        result = examples.complex_network(verbose=False)

        assert len(result['nodes']) == 6
        assert 'gnd' in result['nodes']

    def test_complex_network_kcl(self):
        """Complex network should satisfy KCL at all nodes."""
        import examples
        result = examples.complex_network(verbose=False)

        # KCL summary should show all zeros
        kcl = result['circuit'].kcl_summary()
        assert "0.000 A" in kcl

    def test_current_source_example_verification(self):
        """Current source example should match theoretical values."""
        import examples
        result = examples.current_source_example(verbose=False)

        assert result['verified']

    def test_mixed_sources_power_conservation(self):
        """Mixed sources circuit should conserve power."""
        import examples
        result = examples.mixed_sources_example(verbose=False)

        power = result['circuit'].power_summary()
        assert "0.0 W (should be ~0)" in power


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
