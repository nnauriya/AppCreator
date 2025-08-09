import streamlit as st
import ast
import re
import json
import requests
import io
import base64
import os
from dotenv import load_dotenv
from utils.llm_helpers import query_llm
from utils.file_helpers import write_text_log
import streamlit.components.v1 as components


# Load environment variables from a .env file
load_dotenv()


FRAMEWORKS = ["LangGraph", "CrewAI", "AutoGen", "MetaGPT","LangChain"]
PROGRAMMING_LANGUAGES = ["Python", "JavaScript", "Rust", "Go", "Java", "CSharp"]

# --- Caching Wrapper for LLM Calls ---
@st.cache_data(show_spinner="Calling the LLM... please wait.")
def cached_query_llm(prompt: str, provider: str, model: str, max_tokens: int, temperature: float) -> str:
    """Wrapper around query_llm to enable Streamlit's caching."""
    return query_llm(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def parse_framework_suggestion(response: str):
    lines = response.strip().splitlines()
    framework = None
    justification = ""
    for line in lines:
        lower_line = line.lower()
        if lower_line.startswith("framework:"):
            framework = line.split(":", 1)[1].strip()
        elif lower_line.startswith("justification:"):
            justification = line.split(":", 1)[1].strip()
    return framework, justification


def parse_tool_discovery_code_block(response: str):
    """Parse Python code block and extract internal_tools, external_tools."""
    # Corrected regex to remove the leading code fence
    cleaned = re.sub(r"^.*?```[a-zA-Z]*\n?", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```.*?$", "", cleaned, flags=re.DOTALL).strip()
    exec_vars = {}
    try:
        exec(cleaned, {}, exec_vars)
        internal_tools = exec_vars.get("internal_tools", [])
        external_tools = exec_vars.get("external_tools", [])
    except Exception:
        internal_tools, external_tools = [], []
    return internal_tools, external_tools


def extract_list_or_json(response: str):
    """Extract a Python list or JSON array from the model output, handling code fences and extra text."""
    cleaned = response.strip()
    # Corrected regex to remove the leading code fence
    cleaned = re.sub(r"^.*?```[a-zA-Z]*\n?", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```.*?$", "", cleaned, flags=re.DOTALL).strip()
    bracket_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if bracket_match:
        cleaned = bracket_match.group(0)
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return []
            
def render_mermaid_diagram(mermaid_code):
    """
    Renders a Mermaid diagram using st.components.v1.html to ensure it works reliably.
    Increased height to prevent cutoff issues.
    """
    html_code = f"""
    <div class="mermaid">
      {mermaid_code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true}});</script>
    """
    # Increased height to a generous 1200 pixels to accommodate large diagrams with long text blocks.
    components.html(html_code, height=1200, scrolling=True)


def main():
    st.title("Agentic AI Framework Selector & Tool Discovery MVP")

    if st.button("Start New Project"):
        keys_to_clear = [
            "objective", "what", "who", "why", "where", "when", "what_output",
            "user_framework", "agent_framework", "framework_justification",
            "final_framework", "internal_tools", "external_tools",
            "tools_comparison", "framework_comparison", "block_diagram_mermaid",
            "generated_code",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()

    if st.button("Clear LLM Cache"):
        st.cache_data.clear()
        st.success("Cache cleared! Please rerun your query.")

    # ---- Step 1: Project Objective (5W Expanded) ----
    st.header("1️⃣ Define Project Objective (Answer the 6 Key Questions)")

    # Initialize in session state if not present
    for key in ["what", "who", "why", "where", "when", "what_output"]:
        if key not in st.session_state:
            st.session_state[key] = ""

    st.session_state.what = st.text_area(
        "1. What do you want this application to do? (Describe core task or goal in 1–2 sentences)",
        st.session_state.what,
        height=80,
        key="what_input"
    )
    st.session_state.who = st.text_area(
        "2. Who or what is the subject of this task? (Is the app acting on data, users, documents, markets, etc.?)",
        st.session_state.who,
        height=50,
        key="who_input"
    )
    st.session_state.why = st.text_area(
        "3. Why is this objective important to you or your users? (What value does this application bring?)",
        st.session_state.why,
        height=50,
        key="why_input"
    )
    st.session_state.where = st.text_area(
        "4. Where does this app need to get its information from? (E.g., user input, APIs, websites, databases, files, etc.)",
        st.session_state.where,
        height=50,
        key="where_input"
    )
    st.session_state.when = st.text_area(
        "5. When should this task be performed? (E.g., on demand, scheduled, triggered by an event, etc.)",
        st.session_state.when,
        height=50,
        key="when_input"
    )
    st.session_state.what_output = st.text_area(
        "6. What output or result do you expect from this app? (What should the app produce or return once it completes the task?)",
        st.session_state.what_output,
        height=80,
        key="what_output_input"
    )

    # Validate all fields are filled
    required_fields = [
        st.session_state.what,
        st.session_state.who,
        st.session_state.why,
        st.session_state.where,
        st.session_state.when,
        st.session_state.what_output
    ]

    if not all([x and x.strip() for x in required_fields]):
        st.info("Please answer all 6 questions above to continue.")
        st.stop()

    # Compose full objective string from 5W inputs
    st.session_state.objective = (
        f"What: {st.session_state.what}\n"
        f"Who: {st.session_state.who}\n"
        f"Why: {st.session_state.why}\n"
        f"Where: {st.session_state.where}\n"
        f"When: {st.session_state.when}\n"
        f"What Output: {st.session_state.what_output}"
    )

    st.markdown("**Full Composed Objective:**")
    st.code(st.session_state.objective)

    # ---- Step 2: Framework selection and suggestion ----
    st.header("2️⃣ Choose and Validate Agentic AI Framework")
    user_framework = st.selectbox(
        "Choose your preferred Agentic AI Framework:",
        FRAMEWORKS,
        key="user_framework_select"
    )

    if st.button("Validate and Suggest Framework", key="btn_validate_framework"):
        with st.spinner("Asking the LLM to validate your framework..."):
            prompt = f'''
Given the user's objective: "{st.session_state.objective}"
and the selected framework: "{user_framework}",


Assess if the selected framework is the optimal choice for this project.
If it is the best or an equally strong choice, state that it is the recommended framework.
If another framework from this list: {FRAMEWORKS} is demonstrably better, state the name of that alternative.
You must only choose one framework name.


Respond in this format exactly, no extra text:
Framework: <framework_name>
Justification: <reason>
'''
            response = cached_query_llm(
                prompt=prompt,
                provider="groq",
                model="llama3-70b-8192",
                max_tokens=512,
                temperature=0.2,
            )
            write_text_log(f"Framework Suggestion Prompt:\n{prompt}\nResponse:\n{response}", filename="framework_suggestion.log")
            agent_framework, justification = parse_framework_suggestion(response)
            if not agent_framework:
                agent_framework = user_framework
                justification = "No alternative suggestion found."
            st.session_state.user_framework = user_framework
            st.session_state.agent_framework = agent_framework
            st.session_state.framework_justification = justification

    if st.session_state.get("agent_framework") and st.session_state.get("user_framework"):
        final_choice = st.selectbox(
            "Choose framework to proceed with:",
            options=[st.session_state.user_framework, st.session_state.agent_framework],
            index=0,
            key="final_framework_choice"
        )
        st.session_state.final_framework = final_choice
        st.markdown("#### Agent Suggestion Justification:")
        st.info(st.session_state.framework_justification)

    # ---- Step 3: Tool Discovery ----
    if "final_framework" in st.session_state:
        st.header("3️⃣ Tool Discovery")

        if st.button("Discover Tools", key="btn_discover_tools"):
            with st.spinner("Asking the LLM to discover tools..."):
                discover_prompt = f'''
Given the objective: "{st.session_state.objective}" and the chosen framework: "{st.session_state.final_framework}",


List internal tools/modules (built-in to the framework) and external tools/modules (required outside the framework) needed.


Respond with valid Python code defining exactly two variables:


internal_tools = [...]
external_tools = [...]


IMPORTANT: Output ONLY the code – no explanation, markdown, or extra prose.
'''
                response = cached_query_llm(
                    prompt=discover_prompt,
                    provider="groq",
                    model="llama3-70b-8192",
                    max_tokens=512,
                    temperature=0.7,
                )
                write_text_log(f"Tool Discovery Prompt:\n{discover_prompt}\nResponse:\n{response}", filename="tool_discovery.log")
                try:
                    internal_tools, external_tools = parse_tool_discovery_code_block(response)
                    st.session_state.internal_tools = internal_tools
                    st.session_state.external_tools = external_tools
                except Exception:
                    st.error("Failed to parse tool discovery response. Please try again.")
                    st.caption(response)

        # Show cached results always
        if st.session_state.get("internal_tools"):
            st.markdown("#### Internal Tools:")
            for tool in st.session_state.internal_tools:
                st.markdown(f"- {tool}")

        if st.session_state.get("external_tools"):
            st.markdown("#### External Tools:")
            for tool in st.session_state.external_tools:
                st.markdown(f"- {tool}")

    # ---- Step 4: External Tool Lookup & Comparison ----
    if st.session_state.get("external_tools"):
        st.header("4️⃣ External Tool Lookup & Comparison")

        if st.button("Lookup & Compare External Tools", key="btn_lookup_tools_comp"):
            with st.spinner("Asking the LLM to research external tools..."):
                lookup_prompt = f'''
For the following external tools required: {st.session_state.external_tools}

For each:
- Name the tool, state if it's Open Source, Free Closed Source, or Paid (or a combination).
- Explain its main features and pricing.
- If it's paid, suggest a cheaper OR open-source alternative if one exists, and give a brief comparison.


Respond as a Python list or JSON array exactly as shown below, with no additional text or markdown:


[
  {{
    "tool": "ToolName",
    "type": "Open Source"/"Free"/"Paid",
    "main_features": "...",
    "pricing": "...",
    "alternative": {{
      "name": "AlternativeName",
      "type": "...",
      "main_features": "...",
      "how_it_compares": "..."
    }} # omit or set null if no alternative
  }},
  ...
]


IMPORTANT: Output ONLY the array itself (no prose, no markdown, no explanation, no triple backticks).
'''
                response = cached_query_llm(
                    prompt=lookup_prompt,
                    provider="groq",
                    model="llama3-8b-8192",
                    max_tokens=1500,
                    temperature=0.7,
                )
                write_text_log(f"Extended Tool Lookup:\n{lookup_prompt}\nResponse:\n{response}", filename="external_tool_lookup.log")
                try:
                    tools_comp = extract_list_or_json(response)
                    st.session_state.tools_comparison = tools_comp  # Cache parsed results
                except Exception:
                    st.error("Could not parse enhanced tool comparison response (see below).")
                    st.caption(response)

        # Show cached comparison results
        if st.session_state.get("tools_comparison"):
            tools_comp = st.session_state.tools_comparison
            table_md = (
                "### Comparison Table\n"
                "| Tool | Type | Main Features | Pricing | Cheaper/Open Alternative | Alt Features | Difference |\n"
                "|------|------|---------------|---------|-------------------------|--------------|------------|\n"
            )
            for tool in tools_comp:
                tool_type = tool.get("type", "")
                features = tool.get("main_features", "").replace("\n", " ")
                pricing = tool.get("pricing", "").replace("\n", " ")
                alt = tool.get("alternative")
                if alt and isinstance(alt, dict):
                    alt_name = (alt.get("name") or "").replace("\n", " ")
                    alt_type = (alt.get("type") or "").replace("\n", " ")
                    alt_feat = (alt.get("main_features") or "").replace("\n", " ")
                    comparison = (alt.get("how_it_compares") or "").replace("\n", " ")
                    alt_summary = f"{alt_name} ({alt_type})"
                else:
                    alt_summary = ""
                    alt_feat = ""
                    comparison = ""
                table_md += f"| {tool.get('tool','')} | {tool_type} | {features} | {pricing} | {alt_summary} | {alt_feat} | {comparison} |\n"
            st.markdown(table_md)

    # ---- Step 5: Framework Comparison ----
    if all(k in st.session_state for k in ("user_framework", "agent_framework", "final_framework")):
        st.header("5️⃣ Framework Comparison")

        if st.button("Compare Frameworks", key="btn_compare_frameworks"):
            with st.spinner("Asking the LLM to compare frameworks..."):
                comp_prompt = f'''
Compare the following two Agentic AI frameworks for the project objective: "{st.session_state.objective}"

User selected framework: {st.session_state.user_framework}
Agent suggested framework: {st.session_state.agent_framework}

Please compare these two across these 7 benchmarks:
1. Integration & Extensibility: How easily can the framework be integrated with our existing enterprise systems (e.g., custom databases, internal APIs, authentication protocols like OAuth2) and extended with proprietary, in-house tools?
2. Scalability & Performance: What is the framework's architecture for handling production-level loads? Does it support stateless, horizontally scalable deployments, and can it manage long-running or asynchronous tasks efficiently?
3. Observability & Debugging: What tools and patterns does the framework provide for tracing an agent's decision-making process? How can we effectively monitor, log, and debug complex agent interactions in a production environment (e.g., integration with Datadog, OpenTelemetry)?
4. Governance & Security: How does the framework manage sensitive data, secrets, and API keys? What mechanisms are in place for auditing agent actions and enforcing operational guardrails or business policies?
5. Developer Velocity & Maintainability: How steep is the learning curve for our existing development team? Does the framework's level of abstraction (e.g., high-level and opinionated vs. low-level and flexible) accelerate or hinder rapid prototyping and long-term maintenance?
6. Control & Determinism: How much control does the developer retain over the workflow and agent execution path? How does the framework handle state management to ensure predictable, repeatable outcomes where required for testing and validation?
7. Ecosystem & Total Cost of Ownership (TCO): What is the maturity of the framework's ecosystem (community support, official documentation, pre-built integrations)? What are the primary drivers of its TCO beyond direct LLM API costs (e.g., infrastructure needs, specialized skill requirements)?

Respond as a Python list of dictionaries, exactly as shown below, with no additional text or markdown:

[
    {{
        "benchmark": "Integration & Extensibility?",
        "framework1_name": "{st.session_state.user_framework}",
        "framework1_value": "...",
        "framework2_name": "{st.session_state.agent_framework}",
        "framework2_value": "...",
        "justification": "..."
    }},
    ...
]
'''
                comparison_response = cached_query_llm(
                    prompt=comp_prompt,
                    provider="groq",
                    model="llama3-8b-8192",
                    max_tokens=1024,
                    temperature=0.7,
                )
                write_text_log(f"Framework Comparison Prompt:\n{comp_prompt}\nResponse:\n{comparison_response}", filename="framework_comparison.log")
                
                try:
                    comparison_data = extract_list_or_json(comparison_response)
                    st.session_state.framework_comparison = comparison_data
                except Exception:
                    st.error("Failed to parse framework comparison response. Displaying raw output.")
                    st.caption(comparison_response)
                    st.session_state.framework_comparison = None

        # Show cached framework comparison
        if st.session_state.get("framework_comparison") and isinstance(st.session_state.framework_comparison, list):
            st.markdown("#### Framework Comparison Result:")
            
            # Get the framework names from the first item
            framework1_name = st.session_state.framework_comparison[0]["framework1_name"]
            framework2_name = st.session_state.framework_comparison[0]["framework2_name"]

            # Generate the Markdown table header dynamically
            header = f"| Benchmark | {framework1_name} | {framework2_name} | Justification |\n"
            separator = "|---|---|---|---|\n"
            table_rows = []

            for row in st.session_state.framework_comparison:
                benchmark = row.get("benchmark", "").replace("\n", " ").replace("|", "\\|")
                val1 = row.get("framework1_value", "").replace("\n", " ").replace("|", "\\|")
                val2 = row.get("framework2_value", "").replace("\n", " ").replace("|", "\\|")
                justification = row.get("justification", "").replace("\n", " ").replace("|", "\\|")
                table_rows.append(f"| {benchmark} | {val1} | {val2} | {justification} |")
            
            table_md = header + separator + "\n".join(table_rows)
            st.markdown(table_md)
        elif st.session_state.get("framework_comparison"):
             st.markdown("#### Framework Comparison Result:")
             st.markdown(st.session_state.framework_comparison)

    # ---- Step 6: Generate Block Diagram ----
    if st.session_state.get("final_framework"):
        st.header("6️⃣ Generate Block Diagram")
        st.info("The diagram will be generated using Mermaid syntax and rendered as a vector graphic. This will ensure the text is clear and readable.")
        
        if st.button("Generate Block Diagram", key="btn_generate_diagram"):
            with st.spinner("Generating Mermaid diagram..."):
                # FINAL REPLACEMENT PROMPT FOR STEP 6

                mermaid_prompt = f'''
Act as a business analyst. Your goal is to create a simple business logic flow diagram for a non-technical audience, DONT use Technical jargans.

This blueprint should map out the entire process from the initial business problem to the final solution. Use the following information:

- **The Business Goal:** "{st.session_state.objective}"
- **The Core Engine (Framework):** "{st.session_state.final_framework}"
- **Key Resources & Helpers (Tools):**
  - Internal: {st.session_state.internal_tools}
  - External: {st.session_state.external_tools}

Create a clear, step-by-step workflow diagram that shows how the Core Engine and Key Resources work together to achieve the Business Goal.

---
**Technical Formatting & Syntax Rules:**
1.  The output must be ONLY the raw Mermaid syntax for a top-down graph (`graph TD`). Do not add explanations.
2.  Every step in the blueprint must have a unique ID (e.g., `A`, `B`, `C1`).
3.  **Link Label Syntax:** When labeling a connection, the format MUST be `A -->|label text| B`. Do NOT include any other characters like `>` inside the label's pipes.
    - **CORRECT:** `Start -->|Sends Data| Process`
    - **INCORRECT:** `Start -->|Sends Data|> Process`
'''

                mermaid_code = cached_query_llm(
                    prompt=mermaid_prompt,
                    provider="groq",
                    model="llama3-70b-8192",
                    max_tokens=1024,
                    temperature=0.7,
                )
                st.text_area("LLM Raw Output", mermaid_code, height=300) 
                # New, more robust cleaning logic for the Mermaid code
                cleaned_code = ""
                lines = mermaid_code.strip().splitlines()
                in_code_block = False
                for line in lines:
                    line = line.strip()
                    # Start/end of a code block
                    if line.startswith("```mermaid") or line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    # Keep lines if we are inside a code block or they start with a valid mermaid keyword
                    if in_code_block or line.startswith("graph ") or line.startswith("flowchart "):
                        cleaned_code += line + "\n"

                # Fallback to a simpler cleaning if the robust method yields nothing
                if not cleaned_code.strip():
                    cleaned_code = mermaid_code.strip()
                    if cleaned_code.startswith("```mermaid"):
                        cleaned_code = cleaned_code[len("```mermaid"):].strip()
                    if cleaned_code.endswith("```"):
                        cleaned_code = cleaned_code[:-len("```")].strip()
                
                st.session_state.block_diagram_mermaid = cleaned_code

        # Display the diagram if it's in session state
        if st.session_state.get("block_diagram_mermaid"):
            st.markdown("### Solution Block Diagram")
            render_mermaid_diagram(st.session_state.block_diagram_mermaid)

    # ---- Step 7: Generate Project Code ----
    if st.session_state.get("final_framework"):
        st.header("7️⃣ Generate Project Code")
        st.warning("Note: Read the README.md file for setup instructions and API details.")

        # Let the user choose the programming language and a mock save location
        programming_language = st.selectbox(
            "Select the programming language for the project:",
            options=PROGRAMMING_LANGUAGES,
            key="programming_language_select"
        )
        save_location = st.text_input(
            "Enter a mock location to save the code (e.g., /my_projects/agentic_app/main.py):",
            key="save_location_input"
        )
        st.session_state.save_location = save_location
        st.session_state.programming_language = programming_language

        if st.button("Generate Code", key="btn_generate_code"):
            code_prompt = f'''
Based on the project objective: "{st.session_state.objective}", the chosen framework: "{st.session_state.final_framework}", and the following tools:
Internal Tools: {st.session_state.internal_tools}
External Tools: {st.session_state.external_tools}
Write a complete, runnable code example for this project in {programming_language}. The code should implement the solution and use the specified framework and tools.
Ensure the code is fully commented, well-structured, and includes any necessary imports.
The output should be a single code block, but well explained in comments, so that it can be easily understood by a developer who is not familiar with the framework.
Do not include any additional text or markdown formatting, just the code itself.
Include the requirements.txt and README.md files as well, with appropriate content for the project.
Make sure to include any necessary setup instructions in the README.Also ADD details about API Providers along with free/paid information, urls, authentication, and any other relevant information that a developer would need to run this project.
Just one more very important thing when user selects Python as programming language, please use the version of Python 3.13.5 to build the python code.
'''
            with st.spinner(f"Generating {programming_language} code..."):
                generated_code = cached_query_llm(
                    prompt=code_prompt,
                    provider="groq",
                    model="llama3-70b-8192",
                    max_tokens=2048,
                    temperature=0.7,
                )
                st.session_state.generated_code = generated_code

        # Display the generated code if it's in session state
        if st.session_state.get("generated_code"):
            st.markdown("### Generated Code")
            st.code(st.session_state.generated_code, language=st.session_state.programming_language)


if __name__ == "__main__":
    main()
