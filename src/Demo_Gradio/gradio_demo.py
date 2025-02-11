import gradio as gr
import sys
import sys
sys.path.append('ofx-chat-bot/src/') 
from model.chatbot import get_vectorstore,rqa,result

def generate_response(question):    
    """
    This function retrieves the vectorstore, creates the RetrievalQA chain,         
    and calls the result function to generate a response to the user's question
    (designed for use with Gradio interface).
    """

    # Load the vectorstore (consider caching for efficiency)
    vectorstore = get_vectorstore()
    rqa1 = rqa(vectorstore)

    return result(question, rqa1)


# Define the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(lines=4, label="Ask me a question")],
    outputs="text",
    title="chat bot ",
    theme="huggingface"  # Optional: Use a Hugging Face-themed UI
)

# Launch the Gradio interface
iface.launch() 
