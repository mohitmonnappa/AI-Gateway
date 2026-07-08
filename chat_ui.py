import streamlit as st
import requests
import json

st.set_page_config(page_title="RouteWise AI Chat", page_icon="🤖", layout="wide")

st.title("RouteWise AI Gateway Chat")
st.markdown("Chat with the AI gateway. It will automatically route your prompt to either the Fast model or the Capable model.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("Routing Metadata"):
                st.json(message["metadata"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send request to FastAPI backend
    with st.spinner("Waiting for response..."):
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"prompt": prompt},
                timeout=90
            )
            response.raise_for_status()
            data = response.json()
            
            reply = data.get("response", "")
            
            metadata = {
                "model_used": data.get("model_used"),
                "routing_reason": data.get("routing_reason"),
                "routing_confidence": data.get("routing_confidence"),
                "routing_latency_ms": data.get("routing_latency_ms"),
                "total_latency_ms": data.get("total_latency_ms"),
                "cache_hit": data.get("cache_hit"),
                "cache_similarity": data.get("cache_similarity")
            }
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(reply)
                with st.expander("Routing Metadata"):
                    st.json(metadata)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "metadata": metadata
            })
            
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to the gateway server. Is it running on http://localhost:8000?")
        except Exception as e:
            st.error(f"An error occurred: {e}")
