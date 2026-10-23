
import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "AI agent"

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """Fetch the current weather data for a given city."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weather API key is missing."

    url = "https://api.weatherstack.com/current"

    try:
        response = requests.get(
            url,
            params={"access_key": api_key, "query": city},
            timeout=15,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as error:
        return f"Weather request failed: {error}"


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=500,
)

# Create the modern tool-calling agent
agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful assistant. "
        "Use DuckDuckGoSearchRun to search for current or factual information. "
        "Use get_weather_data when asked for current weather. "
        "Give a clear answer based on the tool results. "
        "If a tool fails, explain that instead of inventing information."
    ),
)

# Invoke the agent
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the birth place of kalpana chawla and the current temperature?"}
    ]
})

print(response["messages"][-1].content)
