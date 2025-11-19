from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse

load_dotenv()


tavily_search = TavilySearch()
tools = [tavily_search]

prompt = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "input",
        "tools",
        "tool_names",
        "agent_scratchpad",
        "format_instructions",
    ],
).partial(format_instructions="")


llm = ChatOpenAI(model="gpt-4o", temperature=0)
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)

extract_output = RunnableLambda(lambda x: x["output"])
# Structured output LLM for final formatting
llm_structured = llm.with_structured_output(AgentResponse)

agent = agent_executor | extract_output | llm_structured


def main():
    print("Hello from langchain-course!")

    # Run the agent to get the answer with sources
    result = agent.invoke(
        {
            "input": "What is the temperature of Gurgaon sector 46 now? Search for real-time data."
        }
    )

    print(result)


if __name__ == "__main__":
    main()
