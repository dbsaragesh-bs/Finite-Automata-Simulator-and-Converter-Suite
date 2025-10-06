# streamlit_app.py
import streamlit as st
from typing import Dict, Set, Tuple, List, Any
from graphviz import Digraph
from chatbot import render_chatbot

# Local modules (ensure these files are in the same directory)
from finite_automata import FA
from nfa import NFA, generate_nfa_visualization, validate_nfa, animate_nfa_validation, accept as dict_accept
from dfa import DFA

# -----------------------------------------------------------------------------
# Helper parsers and converters
# -----------------------------------------------------------------------------
def parse_states(text: str) -> Set[str]:
    return {s.strip() for s in text.split(",") if s.strip()}

def parse_symbols(text: str) -> Set[str]:
    return {s.strip() for s in text.split(",") if s.strip()}

def parse_transitions(text: str, is_nfa: bool) -> Tuple[Dict[str, Dict[str, Set[str]]], List[str]]:
    delta: Dict[str, Dict[str, Set[str]]] = {}
    errors = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for lineno, ln in enumerate(lines, start=1):
        if "->" in ln:
            lhs, rhs = ln.split("->", 1)
        elif ":" in ln:
            lhs, rhs = ln.split(":", 1)
        else:
            errors.append(f"Line {lineno}: missing '->' or ':'")
            continue

        lhs = lhs.strip()
        rhs = rhs.strip()
        if "," in lhs:
            state_part, sym_part = lhs.split(",", 1)
        else:
            errors.append(f"Line {lineno}: left side must be 'state,symbol'")
            continue
        state = state_part.strip()
        symbol = sym_part.strip()
        if symbol in {"ε", "epsilon"}:
            symbol = ""  # empty string for epsilon
        if rhs == "":
            dests = []
        else:
            dests = [d.strip() for d in rhs.split(",") if d.strip()]

        delta.setdefault(state, {})
        delta[state].setdefault(symbol, set())
        for d in dests:
            delta[state][symbol].add(d)

    return delta, errors

def build_dfa_from_inputs(states_txt, symbols_txt, transitions_txt, initial_state, finals_txt):
    q = parse_states(states_txt)
    sigma = parse_symbols(symbols_txt)
    delta_raw, errors = parse_transitions(transitions_txt, is_nfa=False)
    delta: Dict[str, Dict[str, str]] = {}
    for s in delta_raw:
        delta[s] = {}
        for sym, dests in delta_raw[s].items():
            if len(dests) == 0:
                delta[s][sym] = ""
            elif len(dests) > 1:
                delta[s][sym] = ",".join(sorted(dests))
            else:
                delta[s][sym] = next(iter(dests))
    f = parse_states(finals_txt)
    return q, sigma, delta, initial_state, f, errors

def build_nfa_from_inputs(states_txt, symbols_txt, transitions_txt, initial_state, finals_txt):
    q = parse_states(states_txt)
    sigma = parse_symbols(symbols_txt)
    delta, errors = parse_transitions(transitions_txt, is_nfa=True)
    f = parse_states(finals_txt)
    return q, sigma, delta, initial_state, f, errors

def nfa_to_dict_for_visualization(nfa: NFA) -> Dict[str, Any]:
    transitions = {}
    for s, trans in nfa.delta.items():
        for sym, dests in trans.items():
            transitions[(s, "ε" if sym == "" else sym)] = set(dests)
    return {
        "states": set(nfa.q),
        "start_state": nfa.initial_state,
        "end_states": set(nfa.f),
        "transitions": transitions,
    }

# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "define"

if "automaton_obj" not in st.session_state:
    st.session_state.automaton_obj = None
if "automaton_type" not in st.session_state:
    st.session_state.automaton_type = None
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "type": "NFA",
        "states": "q0,q1,q2",
        "symbols": "0,1",
        "start": "q0",
        "finals": "q2",
        "transitions": "q0,0 -> q0\nq0,1 -> q0,q1\nq1,0 -> q2"
    }

def go_define():
    st.session_state.page = "define"

def go_ops():
    st.session_state.page = "ops"

def go_binary():
    st.session_state.page = "binary"

def exit_button():
    cols = st.columns([1, 1, 1, 1, 0.5])
    if cols[-1].button("Exit / Definition"):
        st.session_state.page = "define"

# -----------------------------------------------------------------------------
# Page 1: Automata Definition
# -----------------------------------------------------------------------------
def page_define():
    st.title("Automata Definition")
    exit_button()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Core inputs")
        typ = st.selectbox("Type", ["NFA", "DFA"], index=0 if st.session_state.inputs["type"] == "NFA" else 1)
        states_txt = st.text_input("States (comma-separated)", value=st.session_state.inputs["states"])
        symbols_txt = st.text_input("Symbols (comma-separated)", value=st.session_state.inputs["symbols"])
        start_state = st.text_input("Start state", value=st.session_state.inputs["start"])

    with col_right:
        st.subheader("Finals & Transitions")
        finals_txt = st.text_input("Final state(s) (comma-separated)", value=st.session_state.inputs["finals"])
        st.caption("Transitions format: state,symbol -> dest1,dest2")
        transitions_txt = st.text_area("Transitions", height=180, value=st.session_state.inputs["transitions"])

    st.markdown("---")
    col_action1, col_action2 = st.columns([1, 1])
    with col_action1:
        if st.button("Validate & Continue"):
            st.session_state.inputs.update({
                "type": typ,
                "states": states_txt,
                "symbols": symbols_txt,
                "start": start_state,
                "finals": finals_txt,
                "transitions": transitions_txt
            })

            if typ == "NFA":
                q, sigma, delta, initial_state, f, errors = build_nfa_from_inputs(states_txt, symbols_txt, transitions_txt, start_state, finals_txt)
                if errors:
                    st.error("Transition parsing errors:\n" + "\n".join(errors))
                    return
                try:
                    automaton = NFA(q, sigma, delta, initial_state, f)
                    automaton.is_valid()
                except Exception as e:
                    st.error(f"Invalid NFA: {e}")
                    return
                st.session_state.automaton_obj = automaton
                st.session_state.automaton_type = "NFA"
                st.success("Valid NFA — moving to operations page.")
                st.session_state.page = "ops"
            else:
                q, sigma, delta, initial_state, f, errors = build_dfa_from_inputs(states_txt, symbols_txt, transitions_txt, start_state, finals_txt)
                if errors:
                    st.error("Transition parsing errors:\n" + "\n".join(errors))
                    return
                try:
                    automaton = DFA(q, sigma, delta, initial_state, f)
                    automaton.is_valid()
                except Exception as e:
                    st.error(f"Invalid DFA: {e}")
                    return
                st.session_state.automaton_obj = automaton
                st.session_state.automaton_type = "DFA"
                st.success("Valid DFA — moving to operations page.")
                st.session_state.page = "ops"

    with col_action2:
        if st.button("Reset to Examples"):
            st.session_state.inputs = {
                "type": "NFA",
                "states": "q0,q1,q2",
                "symbols": "0,1",
                "start": "q0",
                "finals": "q2",
                "transitions": "q0,0 -> q0\nq0,1 -> q0,q1\nq1,0 -> q2"
            }
            st.experimental_rerun()

# -----------------------------------------------------------------------------
# Page 2: Operations & Visualization
# -----------------------------------------------------------------------------
def page_ops():
    st.title("Automaton Operations")
    exit_button()

    automaton = st.session_state.automaton_obj
    if automaton is None:
        st.error("No automaton loaded. Returning to definition.")
        st.session_state.page = "define"
        return

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Visualization")
        if st.session_state.automaton_type == "NFA":
            nfa_dict = nfa_to_dict_for_visualization(automaton)
            dot = generate_nfa_visualization(nfa_dict)
            st.graphviz_chart(dot.source, use_container_width=True)
        else:
            dot = automaton.generate_dfa_visualization()
            st.graphviz_chart(dot.source, use_container_width=True)

        st.write("Automaton summary:")
        st.write("Type:", st.session_state.automaton_type)
        st.write("States:", automaton.q)
        st.write("Alphabet:", automaton.sigma)
        st.write("Start:", automaton.initial_state)
        st.write("Finals:", automaton.f)

    with col_right:
        st.subheader("Operations")
        typ = st.session_state.automaton_type
        if typ == "NFA":
            if st.button("Convert to DFA"):
                try:
                    result = automaton.get_dfa()
                    st.session_state.automaton_obj = result
                    st.session_state.automaton_type = "DFA"
                    st.success("Converted to DFA — updated automaton shown.")
                except Exception as e:
                    st.error(f"Error: {e}")
            if st.button("Minimize"):
                try:
                    result = automaton.minimize()
                    st.session_state.automaton_obj = result
                    st.success("Minimized (NFA via DFA method) — updated automaton shown.")
                except Exception as e:
                    st.error(f"Error: {e}")
            if st.button("Complement"):
                try:
                    result = automaton.complement()
                    st.session_state.automaton_obj = result
                    st.success("Complement created — updated automaton shown.")
                except Exception as e:
                    st.error(f"Error: {e}")

            st.markdown("---")
            st.subheader("Dict-based accept & animate")
            test_string = st.text_input("String to test (dict-based accept)", value="")
            if st.button("Dict-based accept (non-animated)"):
                dict_nfa = nfa_to_dict_for_visualization(automaton)
                ok, checks = validate_nfa(dict_nfa, test_string)
                st.write("Accepted?", ok)
                st.write("Step checks:", checks)
            if st.button("Dict-based accept (animated)"):
                dict_nfa = nfa_to_dict_for_visualization(automaton)
                ok, checks = validate_nfa(dict_nfa, test_string)
                if not ok:
                    st.warning("String not accepted — animation will still show traversal.")
                try:
                    animate_nfa_validation(dict_nfa, checks)
                except Exception as e:
                    st.error(f"Animation error (Streamlit required): {e}")

        else:  # DFA ops
            if st.button("Minimize DFA"):
                try:
                    result = automaton.minimize()
                    st.session_state.automaton_obj = result
                    st.success("DFA minimized — updated automaton shown.")
                except Exception as e:
                    st.error(f"Error: {e}")
            if st.button("Complement DFA"):
                try:
                    result = automaton.complement()
                    st.session_state.automaton_obj = result
                    st.success("Complement DFA created — updated automaton shown.")
                except Exception as e:
                    st.error(f"Error: {e}")
            if st.button("To NFA"):
                try:
                    result = automaton.get_nfa()
                    st.session_state.automaton_obj = result
                    st.session_state.automaton_type = "NFA"
                    st.success("Converted to NFA — updated automaton shown.")
                except Exception as e:
                    st.error(f"Error: {e}")

            st.markdown("---")
            st.subheader("DFA accept & animate")
            test_string = st.text_input("String to test (DFA accept)", value="", key="dfa_test")
            if st.button("DFA accept (non-animated)"):
                ok = automaton.accept(test_string, animate=False)
                st.write("Accepted?", ok)
            if st.button("DFA accept (animated)"):
                ok = automaton.accept(test_string, animate=True)
                st.write("Accepted?", ok)

        st.markdown("---")
        if st.button("Binary Operation (use another automaton)"):
            st.session_state.prefill = st.session_state.inputs.copy()
            st.session_state.left_automaton = {
                "type": st.session_state.automaton_type,
                "obj": st.session_state.automaton_obj,
                "inputs": st.session_state.inputs.copy()
            }
            st.session_state.page = "binary"

# -----------------------------------------------------------------------------
# Page 3: Binary operation
# -----------------------------------------------------------------------------
def page_binary():
    st.title("Binary Operation — Select two automata and operation")
    exit_button()

    left_prefill = st.session_state.get("left_automaton", None)
    if left_prefill is None:
        left_prefill = {
            "type": st.session_state.automaton_type,
            "obj": st.session_state.automaton_obj,
            "inputs": st.session_state.inputs.copy()
        }

    left_inputs = left_prefill["inputs"]
    st.subheader("Left automaton (pre-filled)")
    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        left_type = st.selectbox("Type (left)", ["NFA", "DFA"], index=0 if left_inputs["type"] == "NFA" else 1, key="left_type")
        left_states = st.text_input("States (left)", value=left_inputs["states"], key="left_states")
        left_symbols = st.text_input("Symbols (left)", value=left_inputs["symbols"], key="left_symbols")
        left_start = st.text_input("Start (left)", value=left_inputs["start"], key="left_start")
    with col_l2:
        left_finals = st.text_input("Finals (left)", value=left_inputs["finals"], key="left_finals")
        left_trans = st.text_area("Transitions (left)", value=left_inputs["transitions"], key="left_trans")

    st.markdown("---")
    st.subheader("Right automaton (empty — fill details)")
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        right_type = st.selectbox("Type (right)", ["NFA", "DFA"], index=0, key="right_type")
        right_states = st.text_input("States (right)", value="q0,q1", key="right_states")
        right_symbols = st.text_input("Symbols (right)", value="0,1", key="right_symbols")
        right_start = st.text_input("Start (right)", value="q0", key="right_start")
    with col_r2:
        right_finals = st.text_input("Finals (right)", value="q1", key="right_finals")
        right_trans = st.text_area("Transitions (right)", value="q0,0 -> q1", key="right_trans")

    st.markdown("---")
    st.subheader("Choose binary operation (result replaces current automaton)")
    col_ops1, col_ops2 = st.columns([1, 1])

    def parse_and_build_saved(left_or_right):
        if left_or_right == "left":
            typ = left_type
            s_txt = left_states; sy_txt = left_symbols; t_txt = left_trans; start = left_start; finals = left_finals
        else:
            typ = right_type
            s_txt = right_states; sy_txt = right_symbols; t_txt = right_trans; start = right_start; finals = right_finals

        if typ == "NFA":
            q, sigma, delta, initial_state, f, errors = build_nfa_from_inputs(s_txt, sy_txt, t_txt, start, finals)
            if errors:
                return None, typ, "Transition parsing errors: " + "; ".join(errors)
            try:
                obj = NFA(q, sigma, delta, initial_state, f)
                obj.is_valid()
            except Exception as e:
                return None, typ, f"Invalid NFA: {e}"
            return obj, typ, None
        else:
            q, sigma, delta, initial_state, f, errors = build_dfa_from_inputs(s_txt, sy_txt, t_txt, start, finals)
            if errors:
                return None, typ, "Transition parsing errors: " + "; ".join(errors)
            try:
                obj = DFA(q, sigma, delta, initial_state, f)
                obj.is_valid()
            except Exception as e:
                return None, typ, f"Invalid DFA: {e}"
            return obj, typ, None

    left_obj, left_t, left_err = parse_and_build_saved("left")
    right_obj, right_t, right_err = parse_and_build_saved("right")

    if left_err:
        st.error("Left automaton error: " + left_err)
    if right_err:
        st.error("Right automaton error: " + right_err)

    def apply_binary(op_name: str):
        if left_obj is None or right_obj is None:
            st.error("Both automata must be valid before performing an operation.")
            return
        try:
            result = None
            if isinstance(left_obj, DFA) and isinstance(right_obj, DFA):
                if op_name == "union":
                    result = left_obj.union(right_obj)
                elif op_name == "intersection":
                    result = left_obj.intersection(right_obj)
                elif op_name == "difference":
                    result = left_obj.difference(right_obj)
                elif op_name == "symdiff":
                    result = left_obj.symmetric_difference(right_obj)
                elif op_name == "product":
                    result = left_obj.product(right_obj)
                else:
                    st.error(f"Unsupported operation '{op_name}' for DFAs.")
                    return
            else:
                la = left_obj if isinstance(left_obj, NFA) else left_obj.get_nfa()
                ra = right_obj if isinstance(right_obj, NFA) else right_obj.get_nfa()
                if op_name == "union":
                    result = la.union(ra)
                elif op_name == "intersection":
                    result = la.intersection(ra)
                elif op_name == "product":
                    result = la.product(ra)
                else:
                    st.error(f"Unsupported operation '{op_name}' for NFAs.")
                    return

            st.session_state.automaton_obj = result
            st.session_state.automaton_type = "DFA" if isinstance(result, DFA) else "NFA"
            st.session_state.inputs = {
                "type": st.session_state.automaton_type,
                "states": ",".join(sorted(result.q)),
                "symbols": ",".join(sorted(result.sigma)),
                "start": result.initial_state,
                "finals": ",".join(sorted(result.f)),
                "transitions": ""
            }
            st.success("Binary operation applied; returning to operations page to view result.")
            st.session_state.page = "ops"
        except Exception as e:
            st.error(f"Error applying operation: {e}")

    col_ops1, col_ops2 = st.columns([1, 1])
    with col_ops1:
        if st.button("Union"):
            apply_binary("union")
    with col_ops2:
        if st.button("Intersection"):
            apply_binary("intersection")

    col_ops3, col_ops4 = st.columns([1, 1])
    with col_ops3:
        if st.button("Product"):
            apply_binary("product")
    with col_ops4:
        if st.button("Difference (DFA only)"):
            apply_binary("difference")

    st.markdown("---")
    if st.button("Cancel and return to page2"):
        st.session_state.page = "ops"

# -----------------------------------------------------------------------------
# Routing
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    page = st.session_state.page

    # choose page
    if page == "define":
        page_define()
    elif page == "ops":
        page_ops()
    elif page == "binary":
        page_binary()
    else:
        st.session_state.page = "define"
        page_define()

    # --- Prepare automaton context for chatbot ---
    automaton_info = None
    if "automaton_obj" in st.session_state and st.session_state.automaton_obj is not None:
        automaton_info = {
            "type": st.session_state.automaton_type,
            "states": list(st.session_state.automaton_obj.q),
            "alphabet": list(st.session_state.automaton_obj.sigma),
            "start": st.session_state.automaton_obj.initial_state,
            "finals": list(st.session_state.automaton_obj.f),
            "transitions": st.session_state.automaton_obj.delta,
        }

    # --- Render chatbot popup (always visible) ---
    render_chatbot(automaton_info)

if __name__ == "__main__":
    main()
