import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

tavily_client = TavilyClient()


class Source(BaseModel):
    """Source class for the input"""

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Agent response class for the output"""

    response: str = Field(description="The response from the agent")
    sources: List[Source] = Field(
        default_factory=list, description="The sources of the response"
    )


# @tool
# def search(query: str) -> str:
#     """Tool that searches the web for information
#     Args:
#         query: The query to search the web for
#     Returns:
#         The information for the query
#     """
#     print(f"Searching the web for {query}")
#     response = tavily_client.search(query)

#     return f"The information for {query} is {response}"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {"messages": [HumanMessage(content="What is gurgaon sector 46 AQI today?")]}
    )

    for msg in result["messages"]:
        print(msg)


if __name__ == "__main__":
    main()
