import os
import json
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from src.core.config import configs
from src.core.prompt import prompt_settings

def get_context_str(question):
    embeddings = OpenAIEmbeddings(api_key=configs.OPENAI_API_KEY,
                                model='text-embedding-3-small')
    # load vector_store
    vector_store = Chroma(collection_name='illuminus_collection',
                          persist_directory='./chroma_langchain_db', 
                          embedding_function=embeddings)
    retriever = vector_store.as_retriever(
                search_type='mmr', #Use MMR for avoiding redundancy
                search_kwargs={"k": configs.TOP_K})
    context_str = retriever.invoke(question)
    return context_str

def completion_with_retrieval(question: str,
                              history: dict = {}) -> str:
    context_str = get_context_str(question)
    # LLM
    llm = ChatOpenAI(api_key=configs.OPENAI_API_KEY,
                    model_name="gpt-4o-mini", 
                    temperature=0.2,
                    top_p=1)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    system_prompt = PromptTemplate(input_variables=['context'], template=prompt_settings.prompt_rag)
    message = build_langchain_message(system_prompt=system_prompt.format(context = format_docs(context_str)),
                                      question=question,
                                      history=history)
    # print(message)

    # run llm
    response = llm.invoke(message)

    # save history message
    new_conversation = [
        {'role': 'human', 'content':question},
        {'role':'ai', 'content': response.content}
    ]
    save_history_conversation('./history.json', new_conversation)

    return response.content

def build_langchain_message(system_prompt, 
                            question,
                            history: list = []) -> list[tuple[str, str]]:
    message = [("system", system_prompt)]

    if history:
        message.extend((message['role'], message['content']) for message in history)
    
    if question:
        message.append(("human", question))
    
    return message

def save_history_conversation(path:str, new_data:dict):
    try:
        with open(path, "r") as json_file:
            existing_data = json.load(json_file)  # Load existing data as a dictionary
    except FileNotFoundError:
        existing_data = [] 

    # Update the existing dictionary with the new data
    for data in new_data:
        existing_data.append(data)

    # Write the updated dictionary back to the JSON file
    with open(path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
        print("New data has been added to the JSON file.")