from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
import random


def get_vectorstore():
    """
    Loads a vector store using the specified model and parameters.
    """

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )  # Consider using a sentence transformer model optimized for retrieval

    vectorstore = FAISS.load_local(
        "vector_store", embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore

def rqa(vectorstore):
  """
  Creates a RetrievalQA chain with ConversationBufferMemory.
  """

  retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
  memory = ConversationBufferMemory()  # Create the memory instance

  rqa = RetrievalQA.from_chain_type(
      llm=OpenAI(openai_api_key="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"),
      chain_type="map_reduce",
      retriever=retriever,
      memory=memory,  # Add memory to the chain
      verbose=True
  )

  return rqa


#  def rqa(vectorstore):
    # """
    # Creates a RetrievalQA chain using the provided vector store and OpenAI API key.
    # """
# 
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# 
    # rqa = RetrievalQA.from_chain_type(
        # llm=OpenAI(openai_api_key="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"),
        # chain_type="map_reduce",
        # retriever=retriever,
        # verbose=True
    # )

    # return rqa


# dental_prompts = [
#     "As a licensed dentist, I can tell you that {question}.",
#     "From a dental perspective, {question} is because {answer}.",
#     "Here's some information on {question} that might be helpful for you. {answer}",
# ]

# def result(question, rqa):
#   """
#   Invokes the RetrievalQA chain and generates a response with a prompt template.
#   """
#   result = rqa.invoke(question)
#   answer = result["result"]

#   # Choose a random prompt template using the imported random module
#   prompt_template = random.choice(dental_prompts)

#   # Format the final response with the prompt and answer
#   final_answer = prompt_template.format(question=question, answer=answer)
#   return final_answer


def result(question, rqa):
    """
    Invokes the RetrievalQA chain to generate a response to the user's question.
    """

    result = rqa.invoke(question)
    answer = result["result"]
    return answer
