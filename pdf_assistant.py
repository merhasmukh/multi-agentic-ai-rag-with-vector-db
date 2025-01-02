import typer
from typing import Optional
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.chroma import ChromaDb
from rich.prompt import Prompt
from phi.agent import Agent
from phi.model.groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')


knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=ChromaDb(collection="recipes"),
)


knowledge_base.load()

def pdf_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        model=Groq(
        id="llama3-groq-70b-8192-tool-use-preview"
        ),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=True,
    )
    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(pdf_agent)