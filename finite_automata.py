from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
    Iterable,
    Literal,
    Set,
    Dict,
    Any,
    Optional,
    Union,
)
from graphviz import Digraph


class FA(ABC):
    """Abstract base class for Finite Automata.

    This class defines the common interface and attributes for all types of finite automata.
    """

    def __init__(
        self,
        q: Set[str],
        sigma: Set[str],
        delta: Dict[str, Dict[str, Any]],
        initial_state: str,
        f: Set[str],
    ) -> None:
        """Initialize a Finite Automaton.

        Args:
            q: Set of states
            sigma: Set of input symbols (alphabet)
            delta: Transition function
            initial_state: Initial state
            f: Set of final states
        """
        self.q = q
        self.sigma = sigma
        self.delta = delta
        self.initial_state = initial_state
        self.f = f

    # =================================================================
    # VALIDATION METHODS
    # =================================================================

    def is_valid(self) -> bool:
        """Validate the automaton's configuration.

        Returns:
            True if the automaton is valid, raises exception otherwise
        """
        if not self.q:
            raise ValueError("Finite automaton must have at least one state.")
        if self.initial_state is None:
            raise ValueError("Initial state must be defined.")
        if not self._validate_initial_state():
            raise ValueError("Initial state is not in the set of states.")
        if not self._validate_final_states():
            raise ValueError("Some final states are not in the set of states.")
        if not self._validate_transitions():
            raise ValueError("Transition function is invalid.")
        return True

    def _validate_initial_state(self) -> bool:
        """Validate that the initial state is in the set of states.

        Returns:
            True if the initial state is valid, False otherwise
        """
        return self.initial_state in self.q

    def _validate_final_states(self) -> bool:
        """Validate that all final states are in the set of states.

        Returns:
            True if all final states are valid, False otherwise
        """
        return all(f in self.q for f in self.f)

    def _validate_transition(
        self, q: Set[str], next_state: Union[str, Set[str], None]
    ) -> bool:
        """Validate a single transition destination.

        Args:
            q: Set of valid states
            next_state: Destination state(s) to validate

        Returns:
            True if the transition is valid, False otherwise
        """
        if next_state is None:
            return True
        if isinstance(next_state, str):
            return next_state in q
        if isinstance(next_state, set):
            return all(ns in q for ns in next_state)
        return False

    def _validate_transitions(self) -> bool:
        """Validate that all transitions are valid.

        Returns:
            True if all transitions are valid, False otherwise
        """
        for state in self.delta:
            if state not in self.q:
                return False

            for symbol, next_state in self.delta[state].items():
                # Empty string (epsilon) transitions are allowed
                if symbol and symbol not in self.sigma:
                    return False

                if not self._validate_transition(self.q, next_state):
                    return False

        return True

    # =================================================================
    # ACCEPTANCE AND EXECUTION
    # =================================================================

    def accept(self, string: str) -> bool:
        """Check if the automaton accepts a given string.

        Args:
            string: Input string to check

        Returns:
            True if the string is accepted, False otherwise
        """
        self.is_valid()
        if not isinstance(string, str):
            raise TypeError("Input must be a string.")

        current_states = self._epsilon_closure({self.initial_state})
        
        for symbol in string:
            if symbol not in self.sigma:
                return False
            
            next_states: Set[str] = set()
            for state in current_states:
                transitions = self.delta.get(state, {})
                destinations = transitions.get(symbol)
                next_states.update(self._coerce_to_set(destinations))
            
            current_states = self._epsilon_closure(next_states)

        return any(state in self.f for state in current_states)

    # =================================================================
    # BOOLEAN OPERATIONS
    # =================================================================

    def complement(self) -> "FA":
        """Create the complement of this automaton.

        Returns:
            A new automaton that accepts all strings not accepted by this one
        """
        self.is_valid()
        completed = self._make_total_deterministic(self)
        complement_finals = completed.q - completed.f
        return self.__class__(
            completed.q,
            completed.sigma,
            completed.delta,
            completed.initial_state,
            complement_finals,
        )

    def union(self, fa: "FA") -> "FA":
        """Create the union of this automaton with another.

        Args:
            fa: Another finite automaton

        Returns:
            A new automaton that accepts strings accepted by either automaton
        """
        return self._construct_product_automaton(
            fa, lambda left_final, right_final: left_final or right_final
        )

    def intersection(self, fa: "FA") -> "FA":
        """Create the intersection of this automaton with another.

        Args:
            fa: Another finite automaton

        Returns:
            A new automaton that accepts strings accepted by both automata
        """
        return self._construct_product_automaton(
            fa, lambda left_final, right_final: left_final and right_final
        )

    def product(self, fa: "FA") -> "FA":
        """Create the product of this automaton with another.

        Args:
            fa: Another finite automaton

        Returns:
            A new automaton representing the product of both automata
        """
        return self._construct_product_automaton(fa, lambda *_: True)

    # =================================================================
    # VISUALIZATION
    # =================================================================

    def view(
        self,
        file_name: str,
        file_format: Literal["svg", "png"] = "png",
        node_attr: Optional[Dict[str, str]] = None,
        edge_attr: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a visual representation of the automaton.

        Args:
            file_name: Name of the output file
            file_format: Format of the output file (svg or png)
            node_attr: Attributes for nodes in the visualization
            edge_attr: Attributes for edges in the visualization
        """
        dot = self._create_base_graph(file_name, file_format, node_attr, edge_attr)
        
        # Add transitions
        for state, transitions in self.delta.items():
            for symbol, next_state in transitions.items():
                label = "ε" if symbol == "" else symbol
                for destination in self._coerce_to_set(next_state):
                    dot.edge(state, destination, label=label)
        
        dot.render(file_name, cleanup=True)

    def _create_base_graph(
        self,
        file_name: str,
        file_format: Literal["svg", "png"] = "png",
        node_attr: Optional[Dict[str, str]] = None,
        edge_attr: Optional[Dict[str, str]] = None,
    ) -> Digraph:
        """Create a base graph for visualization.

        Args:
            file_name: Name of the output file
            file_format: Format of the output file (svg or png)
            node_attr: Attributes for nodes in the visualization
            edge_attr: Attributes for edges in the visualization

        Returns:
            A configured Digraph object
        """
        dot = Digraph(
            name=file_name,
            format=file_format,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )
        dot.graph_attr["rankdir"] = "LR"
        
        # Invisible start node
        dot.node("", "", shape="plaintext")

        # Final states (double circle)
        for f in self.f:
            dot.node(f, f, shape="doublecircle")

        # Regular states (single circle)
        for q in self.q:
            if q not in self.f:
                dot.node(q, q, shape="circle")

        # Arrow to initial state
        dot.edge("", self.initial_state, label="")
        return dot

    # =================================================================
    # HELPER METHODS
    # =================================================================

    def _coerce_to_set(self, next_state: Union[str, Set[str], None]) -> Set[str]:
        """Convert a transition destination to a set of states.

        Args:
            next_state: State, set of states, or None

        Returns:
            Set of destination states
        """
        if next_state is None:
            return set()
        if isinstance(next_state, str):
            return {next_state}
        if isinstance(next_state, set):
            return set(next_state)
        raise TypeError("Next state must be a string, a set of strings, or None.")

    def _epsilon_closure(self, states: Iterable[str]) -> Set[str]:
        """Compute the epsilon closure of a set of states.

        Args:
            states: Set of states to compute closure for

        Returns:
            Set of all states reachable via epsilon transitions
        """
        closure = set(states)
        stack = list(states)
        
        while stack:
            state = stack.pop()
            epsilon_destinations = self.delta.get(state, {}).get("", None)
            
            for target in self._coerce_to_set(epsilon_destinations):
                if target not in closure:
                    closure.add(target)
                    stack.append(target)
        
        return closure

    def _is_deterministic(self) -> bool:
        """Check if the automaton is deterministic.

        Returns:
            True if deterministic, False otherwise
        """
        for transitions in self.delta.values():
            for symbol, destination in transitions.items():
                # Epsilon transitions make it non-deterministic
                if symbol == "":
                    return False
                
                # Multiple destinations make it non-deterministic
                if isinstance(destination, set):
                    if len(destination) != 1:
                        return False
                elif not isinstance(destination, str):
                    return False
        
        return True

    def _generate_sink_state_name(self, taken: Set[str]) -> str:
        """Generate a unique name for a sink state.

        Args:
            taken: Set of already used state names

        Returns:
            A unique state name
        """
        base = "__sink__"
        if base not in taken:
            return base
        
        index = 1
        while f"{base}_{index}" in taken:
            index += 1
        return f"{base}_{index}"

    def _make_total_deterministic(
        self, automaton: "FA"
    ) -> "_DeterministicRepresentation":
        """Convert to a total deterministic automaton.

        Args:
            automaton: The automaton to convert

        Returns:
            A deterministic representation with complete transition function
        """
        if type(self) is not type(automaton):
            raise TypeError("Automata must be of the same concrete type.")
        if not automaton._is_deterministic():
            raise ValueError("Operation requires a deterministic automaton.")

        sigma = set(automaton.sigma)
        q = set(automaton.q)
        
        # Convert all set destinations to single strings
        delta: Dict[str, Dict[str, str]] = {
            state: {
                symbol: (
                    next(iter(destination))
                    if isinstance(destination, set)
                    else destination
                )
                for symbol, destination in transitions.items()
            }
            for state, transitions in automaton.delta.items()
        }

        # Add missing transitions (make it total)
        sink_state: Optional[str] = None
        for state in q:
            transitions = delta.setdefault(state, {})
            for symbol in sigma:
                if symbol not in transitions:
                    if sink_state is None:
                        sink_state = automaton._generate_sink_state_name(q)
                    transitions[symbol] = sink_state
        
        # Add sink state with self-loops if needed
        if sink_state is not None:
            q.add(sink_state)
            delta[sink_state] = {symbol: sink_state for symbol in sigma}

        return FA._DeterministicRepresentation(
            q=q,
            sigma=sigma,
            delta=delta,
            initial_state=automaton.initial_state,
            f=set(automaton.f),
        )

    def _combine_state_names(self, left: str, right: str) -> str:
        """Create a combined state name for product construction.

        Args:
            left: Left state name
            right: Right state name

        Returns:
            Combined state name
        """
        return f"({left}×{right})"

    def _construct_product_automaton(
        self,
        other: "FA",
        final_condition: Callable[[bool, bool], bool],
    ) -> "FA":
        """Construct a product automaton with custom final state condition.

        Args:
            other: The other automaton
            final_condition: Function determining if product state is final

        Returns:
            Product automaton
        """
        self.is_valid()
        other.is_valid()

        left = self._make_total_deterministic(self)
        right = self._make_total_deterministic(other)

        if left.sigma != right.sigma:
            raise ValueError("Automata must share the same alphabet for this operation.")

        product_states: Set[str] = set()
        product_delta: Dict[str, Dict[str, str]] = {}
        product_finals: Set[str] = set()

        # Construct product states and transitions
        for left_state in left.q:
            for right_state in right.q:
                combined_state = self._combine_state_names(left_state, right_state)
                product_states.add(combined_state)
                product_delta[combined_state] = {}
                
                # Add transitions for each symbol
                for symbol in left.sigma:
                    next_left = left.delta[left_state][symbol]
                    next_right = right.delta[right_state][symbol]
                    product_delta[combined_state][symbol] = self._combine_state_names(
                        next_left, next_right
                    )
                
                # Determine if this product state is final
                if final_condition(
                    left_state in left.f,
                    right_state in right.f,
                ):
                    product_finals.add(combined_state)

        initial_state = self._combine_state_names(
            left.initial_state,
            right.initial_state,
        )

        return self.__class__(
            product_states,
            left.sigma,
            product_delta,
            initial_state,
            product_finals,
        )

    # =================================================================
    # INTERNAL CLASS
    # =================================================================

    class _DeterministicRepresentation:
        """Internal representation of a deterministic automaton."""

        def __init__(
            self,
            q: Set[str],
            sigma: Set[str],
            delta: Dict[str, Dict[str, str]],
            initial_state: str,
            f: Set[str],
        ) -> None:
            self.q = q
            self.sigma = sigma
            self.delta = delta
            self.initial_state = initial_state
            self.f = f
