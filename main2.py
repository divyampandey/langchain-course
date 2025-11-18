from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tavily_search = TavilySearch()
tools = [tavily_search]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
react_agent = create_react_agent(
    llm, tools=[tavily_search], prompt=hub.pull("hwchase17/react")
)
chain = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

agent = chain


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"input": "What is the AQI of Gurgaon sector 46 now?"})
    print(result)


if __name__ == "__main__":
    main()
