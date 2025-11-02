"""
Auto-generated pipeline.py extracted from Eco_Tourism.ipynb
Includes deterministic seed setting for reproducible outputs.
WARNING: This is a best-effort conversion — manual edits may be needed.
"""
import os
import random
import numpy as np
# Set deterministic seeds for reproducibility
SEED = int(os.environ.get('REVIEW_GENIE_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass

# Collected imports from notebook (best-effort)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import csv
import getpass
import getpass  # Importing getpass to securely input sensitive information
import gradio as gr
import kagglehub
import numpy as np
import os
import os  # Importing the os module to interact with environment variables
import pandas as pd

# Concatenated code cells (wrapped into functions where appropriate)

def cell_0():
    pass

def cell_1():
    # Importing the KaggleHub library to interact with datasets and models available on Kaggle.
    import kagglehub
    
    # Importing the CSV module for reading and writing CSV files.
    import csv
    
    # Importing pandas for data manipulation and analysis.
    import pandas as pd
    
    # Importing numpy for numerical operations and handling arrays efficiently.
    import numpy as np
    
    # Importing os to interact with the operating system, such as environment variables and file paths.
    import os
    
    # Importing getpass to securely handle user input (e.g., API keys or passwords).
    import getpass

def cell_2():
    # Mount Google Drive
    
    drive.mount("/content/gdrive")

def cell_3():
    # Loading the data
    df = pd.read_csv("/content/gdrive/MyDrive/tourism_resource_dataset.csv",index_col=0)

def cell_4():
    # Viewing the data
    df.head(10)

def cell_5():
    # Importing RecursiveCharacterTextSplitter from LangChain for chunking large text into smaller, manageable pieces.
    # This helps in optimizing text for processing and retrieval.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Importing OpenAIEmbeddings from LangChain to generate numerical vector representations (embeddings) of text.
    # These embeddings capture the semantic meaning of the text for efficient similarity searches.
    from langchain_openai import OpenAIEmbeddings
    
    # Importing FAISS (Facebook AI Similarity Search) from LangChain's community package.
    # FAISS is used for storing and retrieving embeddings efficiently by finding similar vectors.
    from langchain_community.vectorstores import FAISS
    
    # Importing the OpenAI library to interact with OpenAI's API services.
    from openai import OpenAI
    
    import os  # Importing the os module to interact with environment variables
    import getpass  # Importing getpass to securely input sensitive information
    
    # Prompting the user to securely enter their OpenAI API key without displaying it on the screen
    OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ")
    
    # Setting the OpenAI API key as an environment variable.
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def cell_6():
    # Setting the OpenAI API key as an environment variable.
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    # Convert each row into a readable string for embeddings
    texts = df.apply(lambda row: (
        f"Location {row['location_id']} had {row['visitor_count']} visitors. "
        f"Resource usage rate: {row['resource_usage_rate']}%. "
        f"Temperature: {row['temperature']}°C, AQI: {row['air_quality_index']}, Noise level: {row['noise_level']} dB. "
        f"Season: {row['season']}, Peak hour: {row['peak_hour_flag']}. "
        f"Visitor satisfaction: {row['visitor_satisfaction']}, Resource prediction: {row['resource_prediction']} units."
    ), axis=1).tolist()
    
    # Split text into documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    
    documents = text_splitter.create_documents(texts)
    
    # Show a sample document
    print(documents[0])

def cell_7():
    # Create an embedding model using LangChain.
    # One option is using https://python.langchain.com/docs/integrations/text_embedding/openai/
    # See https://python.langchain.com/docs/integrations/text_embedding/ for a list of available embedding models on LangChain
    embeddings = OpenAIEmbeddings()

def cell_8():
    # Create a vector store using the created chunks and the embeddings model
    vector = FAISS.from_documents(documents, embeddings)

def cell_9():
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableMap

def cell_10():
    # Initializing the ChatOpenAI model to interact with OpenAI's GPT model.
    llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model = 'gpt-4o-mini')

def cell_11():
    # Importing the output parser to process and format the model's response into a readable string format.
    output_parser = StrOutputParser()
    chat_history = []
    # Creating a prompt template that instructs the LLM to act as a customer service agent.
    # The prompt takes two parameters:
    #   1. {context} - Relevant information retrieved from the document store.
    #   2. {input} - The user's question.
    # The model is instructed to base its answer solely on the provided context.
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the following context and chat history to answer the question.
    
    <context>
    {context}
    </context>
    
    <chat_history>
    {chat_history}
    </chat_history>
    
    Question: {input}"""
    )
    
    # Create a retriever from the vector store for fetching relevant documents
    # See https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
    retriever = vector.as_retriever()
    
    # Creating a document processing chain using the LLM and the defined prompt
    # template.
    # This chain takes a list of retrieved documents and passes them as context to
    # the model for generating responses.
    
    
    output_parser = StrOutputParser()
    
    rag_chain = (
        RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: "\n".join([doc.page_content for doc in retriever.invoke(x["input"])])
        })
        | prompt
        | llm
        | output_parser
    )
    
    def format_chat_history(history):
        return "\n".join([f"User: {q}\nBot: {a}" for q, a in history])

def cell_12():
    # Invoking the retrieval chain to process the user's query.
    # The query "what are some of the best shoes available?" is passed as input.
    # The retrieval chain first fetches relevant product descriptions from the vector store,
    # then processes them using the document chain to generate a meaningful LLM response.
    user_query = "what location has the top visitors?"
    formatted_history = format_chat_history(chat_history)
    
    response = rag_chain.invoke({
        "input": user_query,
        "chat_history": formatted_history
    })
    
    chat_history.append((user_query, response))
    print(response)

def cell_13():
    # Fetching the final answer from the retrieval chain by invoking it with a user query.
    # The ['answer'] key extracts the final LLM-generated answer from the response dictionary.
    user_query = "what season has the top visitors?"
    formatted_history = format_chat_history(chat_history)
    
    response = rag_chain.invoke({
        "input": user_query,
        "chat_history": formatted_history
    })
    
    chat_history.append((user_query, response))
    print(response)

def cell_14():
    
    
    # Define a function to run the full RAG process
    def final_response(user_query):
        formatted_history = format_chat_history(chat_history)
    
        # Step 1: Retrieve context
        context_docs = retriever.invoke(user_query)
        context_text = "\n".join([doc.page_content for doc in context_docs])
    
        # Step 2: Format the prompt with all variables
        prompt_input = {
            "input": user_query,
            "context": context_text,
            "chat_history": formatted_history
        }
        formatted_prompt = prompt.invoke(prompt_input)
    
        # Step 3: Run LLM on formatted prompt
        response = llm.invoke(formatted_prompt)
        parsed_response = output_parser.invoke(response)
    
        # Step 4: Save to history
        chat_history.append((user_query, parsed_response))
    
        return parsed_response
    
    def chatbot_fn(message, history):
        # Format history for prompt
        formatted_history = format_chat_history(chat_history)
    
        # Run RAG pipeline
        response = rag_chain.invoke({
            "input": message,
            "chat_history": formatted_history
        })
    
        # Save to history
        chat_history.append((message, response))
    
        return response

def cell_15():
    import gradio as gr
    chatbot_ui = gr.ChatInterface(
        fn=chatbot_fn,
        title="Echo Tourism RAG Bot",
        description="Ask me anything about eco-tourism and sustainability!",
        theme="soft",
    )
    
    chatbot_ui.launch()

def cell_16():
    pass

def run_all():
    """Run all extracted code cells in order. Modify as needed."""
    try:
        cell_0()
    except Exception as e:
        print('Error running cell_0:', e)

    try:
        cell_1()
    except Exception as e:
        print('Error running cell_1:', e)

    try:
        cell_2()
    except Exception as e:
        print('Error running cell_2:', e)

    try:
        cell_3()
    except Exception as e:
        print('Error running cell_3:', e)

    try:
        cell_4()
    except Exception as e:
        print('Error running cell_4:', e)

    try:
        cell_5()
    except Exception as e:
        print('Error running cell_5:', e)

    try:
        cell_6()
    except Exception as e:
        print('Error running cell_6:', e)

    try:
        cell_7()
    except Exception as e:
        print('Error running cell_7:', e)

    try:
        cell_8()
    except Exception as e:
        print('Error running cell_8:', e)

    try:
        cell_9()
    except Exception as e:
        print('Error running cell_9:', e)

    try:
        cell_10()
    except Exception as e:
        print('Error running cell_10:', e)

    try:
        cell_11()
    except Exception as e:
        print('Error running cell_11:', e)

    try:
        cell_12()
    except Exception as e:
        print('Error running cell_12:', e)

    try:
        cell_13()
    except Exception as e:
        print('Error running cell_13:', e)

    try:
        cell_14()
    except Exception as e:
        print('Error running cell_14:', e)

    try:
        cell_15()
    except Exception as e:
        print('Error running cell_15:', e)

    try:
        cell_16()
    except Exception as e:
        print('Error running cell_16:', e)


if __name__ == '__main__':
    run_all()
