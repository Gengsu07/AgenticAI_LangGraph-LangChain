import json
import glob
import os

directory = r"d:\PROJECTS\AgenticAI_LangGraph-LangChain\Lecturer\Section 14-Different Workflow in LangGraph"
notebooks = glob.glob(os.path.join(directory, "*.ipynb"))
notebooks.sort()

with open(os.path.join(directory, "extracted_content.md"), "w", encoding="utf-8") as out:
    for nb in notebooks:
        out.write(f"# Notebook: {os.path.basename(nb)}\n\n")
        with open(nb, "r", encoding="utf-8") as f:
            data = json.load(f)
            for cell in data.get("cells", []):
                cell_type = cell.get("cell_type")
                source = "".join(cell.get("source", []))
                if cell_type == "markdown":
                    out.write(f"{source}\n\n")
                elif cell_type == "code":
                    out.write(f"```python\n{source}\n```\n\n")
        out.write("---\n\n")
print(f"Extraction complete. Found {len(notebooks)} notebooks. Output saved to extracted_content.md")
