import os
from typing import Optional, Tuple

import gradio as gr
from langchain.llms import OpenAIChat
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from threading import Lock


def load_chain():
    prefix_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who is very good at problem solving and thinks step by step. You are about to receive a complex set of instructions to follow for the remainder of the conversation. Good luck!"
        }
    ]

    llm = OpenAIChat(model_name="gpt-3.5-turbo-0301", temperature=0.8, prefix_messages=prefix_messages)

    prompt = PromptTemplate(
        input_variables=['history', 'input'],
        output_parser=None,
        template='Current conversation:\n{history}\n\nUser: """""\n{input}"""""\n\nAssistant: ',
        template_format='f-string'
    )

    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationBufferMemory(human_prefix="User", ai_prefix="Assistant")
    )

    return chain


def load_prompt(prompt_selection: str):
    """Load the selected initializing prompt."""
    path = f"prompts/{prompt_selection}/prompt.txt"

    with open(path, "r") as f:
        init_prompt = f.read()
        print(f"Loading {path.split('/')[-2]} from: {path}...") # e.g. Loading proposal-gen from: prompts/work/proposal-gen/prompt.txt

    chain = load_chain()
    chain.predict(input=init_prompt)
    print(f"Done! Loaded {len(chain.memory.buffer)} characters.")
    return chain

def fetch_prompts():
    """Iterates recursively through the prompts directory, returning a list of prompts.

    This is used to populate the dropdown menu in the Gradio interface.
    """
    available_prompts = []
    for root, dirs, files in os.walk("prompts"):
        if "prompt.txt" in files:
            available_prompts.append(root.replace("prompts/", "").replace("/prompt.txt", "")) # remove the "prompts/" prefix and the "/prompt.txt" suffix
            available_prompts.sort()
            
    return available_prompts

def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print("API key set.")
        # chain = load_chain() # loads the chain.
        chain = load_prompt(selected_prompt.value)
        # os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []            
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            
            # Run chain and append input.
            output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h2><center>PromptLib</center></h2>")

        with gr.Tab('Prompt'):

            selected_prompt = gr.Dropdown(
                choices=fetch_prompts(),
                type="value",
                value="work/proposal-gen",
                label="Base prompt",
                interactive=True
            )

            reload_prompt= gr.Button(
                value="Reload",
                variant="secondary"
            )

        with gr.Tab('API Key'):
            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key (sk-...)",
                show_label=False,
                lines=1,
                type="password",
            )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="Message",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What can you do? What command(s) are available?",
            "Please suggest some sample commands.",
        ],
        inputs=message,
    )

    gr.HTML(
        "<center>Josh Pazmino | <a href='https://github.com/jmpaz'>GitHub</a> • <a href='https://twitter.com/fjpaz_'>Twitter</a> • <a href='https://linkedin.com/in/fjpazmino'>LinkedIn</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])


    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

    selected_prompt.change(
        load_prompt,
        inputs=[selected_prompt],
        outputs=[agent_state]
    )

    reload_prompt.click(load_prompt, inputs=[selected_prompt], outputs=[agent_state])


block.launch(debug=True)