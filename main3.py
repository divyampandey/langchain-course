from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse

load_dotenv()

parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "input",
        "tools",
        "tool_names",
        "agent_scratchpad",
        "format_instructions",
    ],
).partial(format_instructions=parser.get_format_instructions())
print(react_prompt)

tavily_search = TavilySearch()
tools = [tavily_search]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
react_agent = create_react_agent(llm, tools=tools, prompt=react_prompt)
chain = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: parser.parse(x))

agent = chain | extract_output | parse_output


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "input": """Can you please tell authentic and popular
     sources widely used to prepare for the 
     product sense interviews for the 
     Data Scientist/ ML Engineer role for product based companies in 2025. Give best answer as per user reviews and content interaction and sort them by confidence?"""
        }
    )
    print(result)


if __name__ == "__main__":
    main()
