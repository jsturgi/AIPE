"""
Component Classes for Circuit Project
"""

class Node:
    def __init__(self, id):
        self.id = id
        self.voltage = None
        
    def reset(self):
        self.voltage = None
       

class Resistor:
    def __init__(self, resistance, positive_node, negative_node ):
        self.resistance = resistance
        self.positive_node = positive_node
        self.negative_node = negative_node
        
    def voltage(self) -> float:
        return self.positive_node.voltage - self.negative_node.voltage
    def current(self) -> float:
        return self.voltage() / self.resistance

class VoltageSource():
    def __init__(self, voltage, positive_node, negative_node):
      self.voltage = voltage
      self.positive_node = positive_node
      self.negative_node = negative_node
      self._current = None
    
    

class CurrentSource():
    def __init__(self, current, positive_node, negative_node):
        self.current = current
        self.positive_node = positive_node
        self.negative_node = negative_node
    
    def voltage(self) -> float:
        return self.positive_node.voltage - self.negative_node.voltage
        