from dotenv import load_dotenv 
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor


load_dotenv()


class TeachingResponse(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools : list[str]
    technologies : list[str]
    examples : list[str]    
    code_snippet : list[str] 

    
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=TeachingResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a MCA professor who will teaches a student.
            Answer students query, Explain query solution and necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]     
).partial(format_instructions=parser.get_format_instructions()) 


agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt, 
    tools=[]
)

agent_executor = AgentExecutor(agent=agent,tools=[])#,verbose=True)
raw_response = agent_executor.invoke({"query":"AR in Education"})
#print(raw_response)

indented_response = parser.parse(raw_response.get("output"))
print(indented_response.topic)

print()
print(indented_response.summary)
print()
print(indented_response.sources)