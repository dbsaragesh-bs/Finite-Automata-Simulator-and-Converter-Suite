# Automathon - Interactive Finite Automata Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)

An interactive, educational web application for working with **Deterministic Finite Automata (DFA)** and **Non-Deterministic Finite Automata (NFA)**. Built with Python and Streamlit, this tool provides a comprehensive environment for defining, visualizing, manipulating, and testing finite automata with an integrated AI chatbot tutor.

## 🌟 Features

### Core Functionality
- **DFA & NFA Support**: Full implementation of both deterministic and non-deterministic finite automata
- **Visual Editor**: Interactive interface for defining states, transitions, and automata properties
- **Real-time Validation**: Instant feedback on automaton validity
- **Interactive Visualization**: Graphical representation using Graphviz with animated string acceptance

### Automata Operations
- ✅ **Conversion**: NFA to DFA conversion
- ✅ **Minimization**: DFA minimization using Hopcroft's algorithm
- ✅ **Complement**: Generate complement automata
- ✅ **Union**: Combine two automata (accepts strings accepted by either)
- ✅ **Intersection**: Combine two automata (accepts strings accepted by both)
- ✅ **Product**: Cartesian product of two automata
- ✅ **Difference**: Set difference operation (DFA only)
- ✅ **Symmetric Difference**: XOR operation (DFA only)

### Advanced Features
- **Epsilon Transitions**: Full support for ε-transitions in NFAs
- **Epsilon Closure**: Automatic computation of epsilon closures
- **String Acceptance Testing**: Test whether strings are accepted by the automaton
- **Animated Validation**: Visual step-by-step string acceptance animation
- **AI Chatbot Tutor**: Integrated chatbot powered by Groq LLM for answering questions about automata theory

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository** (or download the project):
```bash
git clone <your-repository-url>
cd automathon
```

2. **Install required dependencies**:
```bash
pip install streamlit graphviz groq
```

3. **Install Graphviz system library** (required for visualization):

   - **Windows**: Download from [graphviz.org](https://graphviz.org/download/) and add to PATH
   - **macOS**: 
     ```bash
     brew install graphviz
     ```
   - **Linux (Ubuntu/Debian)**:
     ```bash
     sudo apt-get install graphviz
     ```

### Running the Application

Launch the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage Guide

### 1. Defining an Automaton

The application starts with the **Definition Page** where you can specify your automaton:

#### Input Fields
- **Type**: Choose between NFA or DFA
- **States**: Comma-separated list of states (e.g., `q0,q1,q2`)
- **Symbols**: Comma-separated alphabet symbols (e.g., `0,1`)
- **Start State**: Initial state (e.g., `q0`)
- **Final States**: Comma-separated accepting states (e.g., `q2`)
- **Transitions**: One transition per line in the format:
  ```
  state,symbol -> destination_state(s)
  ```

#### Transition Format Examples

**DFA Example:**
```
q0,0 -> q0
q0,1 -> q1
q1,0 -> q2
q1,1 -> q0
```

**NFA Example (multiple destinations):**
```
q0,0 -> q0
q0,1 -> q0,q1
q1,0 -> q2
```

**Epsilon Transitions (use ε or epsilon):**
```
q0,ε -> q1
q1,0 -> q2
```

### 2. Visualizing and Operating

After validation, you'll see the **Operations Page** with:

#### Left Panel - Visualization
- Graphical representation of your automaton
- States shown as circles (double circles for final states)
- Transitions labeled with symbols
- Automaton summary (states, alphabet, start, finals)

#### Right Panel - Operations

**For NFAs:**
- Convert to DFA
- Minimize (converts to DFA, minimizes, then back to NFA)
- Complement
- Test string acceptance (dict-based, with animation option)

**For DFAs:**
- Minimize DFA
- Complement DFA
- Convert to NFA
- Test string acceptance (with animation option)

### 3. Binary Operations

Click **"Binary Operation"** to combine two automata:

1. The left automaton is pre-filled with your current automaton
2. Define a right automaton using the same interface
3. Choose from operations:
   - **Union**: L(A) ∪ L(B)
   - **Intersection**: L(A) ∩ L(B)
   - **Product**: Cartesian product
   - **Difference**: L(A) - L(B) (DFA only)

### 4. AI Chatbot Tutor

The chatbot button (💬) appears in the bottom-right corner:
- Click to expand the chat interface
- Ask questions about automata theory
- Get explanations about your current automaton
- Learn concepts like epsilon closures, minimization, etc.

## 🧮 Example: Accepting Binary Strings Ending in "10"

### Define the NFA
```
Type: NFA
States: q0,q1,q2
Symbols: 0,1
Start: q0
Finals: q2
Transitions:
q0,0 -> q0
q0,1 -> q0,q1
q1,0 -> q2
```

### Test Strings
- `"110"` → **Accepted** ✅
- `"10"` → **Accepted** ✅
- `"101"` → **Rejected** ❌
- `"0010"` → **Accepted** ✅

### Convert to DFA
1. Click "Convert to DFA"
2. View the equivalent DFA with subset construction
3. Minimize for optimal state count

## 🏗️ Project Structure

```
automathon/
├── streamlit_app.py          # Main Streamlit application
├── finite_automata.py        # Abstract base class for FA
├── dfa.py                    # DFA implementation
├── nfa.py                    # NFA implementation
├── utils.py                  # Helper functions
├── chatbot.py                # AI chatbot integration
├── __init__.py               # Package initialization
├── automathon/               # Package directory
│   ├── finite_automata/      # Automata modules
│   │   ├── dfa.py
│   │   ├── nfa.py
│   │   └── ...
└── __pycache__/              # Compiled Python files
```

## 🔧 API Reference

### DFA Class

```python
from dfa import DFA

# Create a DFA
dfa = DFA(
    q={'q0', 'q1', 'q2'},              # States
    sigma={'0', '1'},                   # Alphabet
    delta={                             # Transitions
        'q0': {'0': 'q0', '1': 'q1'},
        'q1': {'0': 'q2', '1': 'q0'},
        'q2': {'0': 'q2', '1': 'q2'}
    },
    initial_state='q0',                 # Start state
    f={'q2'}                            # Final states
)

# Test string acceptance
dfa.accept("010")  # Returns: True/False

# Minimize the DFA
minimized = dfa.minimize()

# Complement
complement_dfa = dfa.complement()

# Union with another DFA
union_dfa = dfa1.union(dfa2)
```

### NFA Class

```python
from nfa import NFA

# Create an NFA
nfa = NFA(
    q={'q0', 'q1', 'q2'},
    sigma={'0', '1'},
    delta={
        'q0': {'0': {'q0'}, '1': {'q0', 'q1'}},
        'q1': {'0': {'q2'}}
    },
    initial_state='q0',
    f={'q2'}
)

# Convert to DFA
dfa = nfa.get_dfa()

# Remove epsilon transitions
nfa_without_epsilon = nfa.remove_epsilon_transitions()

# Union with another NFA
union_nfa = nfa1.union(nfa2)
```

## 🎯 Key Algorithms Implemented

### 1. **NFA to DFA Conversion (Subset Construction)**
- Implements the powerset construction algorithm
- Handles epsilon transitions
- Generates minimal state names

### 2. **DFA Minimization (Hopcroft's Algorithm)**
- Partitions states into equivalence classes
- Eliminates unreachable states
- Produces minimal DFA

### 3. **Epsilon Closure Computation**
- Recursive depth-first search
- Handles cyclic epsilon transitions
- Used in NFA operations

### 4. **Product Construction**
- Cartesian product of state sets
- Supports multiple final state conditions
- Used for union, intersection, difference

## 🤖 Chatbot Integration

The application includes an AI-powered chatbot tutor using Groq's LLM API:

### Features
- Context-aware responses about your current automaton
- Explanations of automata theory concepts
- Step-by-step guidance for operations
- Educational support for learning formal languages

### Setup
The chatbot requires a Groq API key. To use your own:
1. Get an API key from [Groq Console](https://console.groq.com/)
2. Update `chatbot.py` with your key:
```python
client = Groq(api_key="your-api-key-here")
```

## 🎨 Visualization

Automata are visualized using Graphviz with:
- **Circles**: Regular states
- **Double circles**: Final/accepting states
- **Arrows**: Transitions labeled with input symbols
- **Color coding** (during animation):
  - 🟡 Yellow: Current state being processed
  - 🟢 Green: Accepted final state
  - 🔴 Red: Rejected state
  - ⚪ White: Default state

## 📚 Educational Use Cases

This tool is perfect for:
- **Students** learning automata theory and formal languages
- **Instructors** demonstrating concepts in lectures
- **Researchers** prototyping automata-based algorithms
- **Developers** understanding state machines and pattern matching

### Topics Covered
- Deterministic vs. Non-deterministic automata
- Regular languages and regular expressions
- State minimization and equivalence
- Closure properties of regular languages
- Automata composition and decomposition

## 🔍 Testing Examples

### Example 1: Binary String Divisibility
Design a DFA that accepts binary strings divisible by 3.

### Example 2: Pattern Matching
Create an NFA that accepts strings containing "101" as a substring.

### Example 3: Language Intersection
Given two automata, find strings accepted by both.

## 🛠️ Troubleshooting

### Common Issues

**1. Graphviz not rendering:**
```bash
# Ensure Graphviz is installed and in PATH
dot -V  # Should show version
```

**2. Module import errors:**
```bash
# Reinstall dependencies
pip install --upgrade streamlit graphviz groq
```

**3. Animation not working:**
- Animations require Streamlit's runtime
- Run the app with `streamlit run streamlit_app.py`
- Don't run Python scripts directly

**4. Chatbot not responding:**
- Check your Groq API key
- Ensure internet connection
- Verify API rate limits

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional automata types (Pushdown Automata, Turing Machines)
- Regular expression to automata conversion
- More visualization options
- Performance optimizations for large automata
- Unit tests and test coverage

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Project developed for educational purposes in Formal Languages and Automata Theory

## 🙏 Acknowledgments

- **Graphviz** for visualization capabilities
- **Streamlit** for the interactive web framework
- **Groq** for AI chatbot integration
- Theory based on classic texts:
  - *Introduction to Automata Theory, Languages, and Computation* by Hopcroft, Motwani, and Ullman
  - *Introduction to the Theory of Computation* by Michael Sipser

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Use the integrated AI chatbot for automata theory questions
- Check the examples and documentation

## 🔮 Future Enhancements

- [ ] Regular expression to automata conversion
- [ ] Pushdown Automata (PDA) support
- [ ] Turing Machine simulator
- [ ] Export/import automata definitions (JSON/XML)
- [ ] Batch string testing
- [ ] Performance metrics and complexity analysis
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support

---

**Happy Automating! 🤖✨**

*Learn, Visualize, and Master Finite Automata with Automathon!*
