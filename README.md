# AI Research Assistant with Multi-Agent Workflows

A hands-on project demonstrating **agentic AI workflow patterns** using **Google's Agent Development Kit (ADK)**.

## Testing Your Implementation

### Run the Complete Workflow

```bash
python main.py
```

### Test with Custom Research Queries

```bash
export RESEARCH_QUERY="What are the smartest birds on the planet?"
python main.py
```

Or edit the `queries` list in `main.py` to add your own research topics.

### Expected Output

If your implementation is correct, you should see:

1. All 7 stages execute successfully
2. LoopAgent, ParallelAgent, and SequentialAgent objects created
3. LLM output from Gemini
4. Performance evaluation metrics displayed
5. A generated `research_report.md` file with comprehensive findings

## Student notes

### Custom query

> Tell me about the history of the Python programming language, including evidences.

### Included files

```txt
docs/
  - 1. Project_Overview.md
  - 2. Environment_Setup.md
  - 3. Accessing_Google_Cloud_Platform_Credentials.md
  - 4. Configuring_Vertex_AI_Service_Account.md
  - 5. Instructions.md
  - 6. Starter_Code_and_Workspace.md
  - research_report.md (the final product of the agentic workflow)
img/
  - screenshots/ -> contains several screenshots showing the workflow output before and after the full implementation
terminal/
  - terminal output - full run.txt (the full workflow output after the first successful full run)
```

### Rubrics validation (via local LLM)

See document [8. Rubrics_Validation.md](./project-starter/docs/8.%20Rubrics_Validation.md)
