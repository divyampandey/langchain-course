from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schema import AgentResponse

load_dotenv()

tavily_search = TavilySearch()
tools = [tavily_search]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the temperature of Gurgaon sector 46 now?"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
