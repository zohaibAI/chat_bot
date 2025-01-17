from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

app = FastAPI(title="Groq Chat API")

# Pydantic models for request/response validation
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    history: List[Message]

def convert_to_langchain_messages(messages: List[Message]):
    """Convert the chat history to LangChain message format"""
    langchain_messages = []
    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    return langchain_messages

def setup_groq_chat():
    """Initialize and return the Groq chat configuration"""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")
    
    model = 'llama3-8b-8192'
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that accepts a question and optional chat history.
    
    Request format:
    {
        "question": "string",  // The user's question
        "history": [          // Optional chat history
            {
                "role": "user" | "assistant",
                "content": "string"
            }
        ]
    }
    """
    try:
        groq_chat = setup_groq_chat()
        system_prompt = 'You are a friendly conversational chatbot'
        conversational_memory_length = 5

        # Initialize memory with existing history if provided
        memory = ConversationBufferWindowMemory(
            k=conversational_memory_length,
            memory_key="chat_history",
            return_messages=True
        )

        # Add existing history to memory
        if request.history:
            langchain_messages = convert_to_langchain_messages(request.history)
            for i in range(0, len(langchain_messages), 2):
                if i + 1 < len(langchain_messages):
                    memory.save_context(
                        {"input": langchain_messages[i].content},
                        {"output": langchain_messages[i+1].content}
                    )

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        # Create conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory
        )

        # Get response
        response = conversation.predict(human_input=request.question)

        # Update history with new interaction
        updated_history = request.history + [
            Message(role="user", content=request.question),
            Message(role="assistant", content=response)
        ]

        return ChatResponse(
            response=response,
            history=updated_history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9002)