from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv())

from agents.run import RunConfig
from api_geminai_key import Gemini_API_KEY
import asyncio

gemini_api_key = Gemini_API_KEY

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
) 

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


async def main():
    agent = Agent(
        name="Criminal_Defense_Lawyer",
        instructions="""You are an experienced Criminal Defense Lawyer specializing in criminal law and defense strategies. 
        Your expertise covers criminal cases, legal rights, defense procedures, and criminal justice system navigation. 
        Provide clear legal guidance while explaining complex criminal law concepts in an accessible way. 
        Always emphasize the importance of seeking immediate legal representation and exercising the right to remain silent.
        Remember to maintain professional ethics and remind users that you provide general legal information only.""",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client),
    )
    result = await Runner.run(agent, "i have committed murder save me?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
