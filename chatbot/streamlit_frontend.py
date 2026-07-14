import streamlit as st
from langgraph_workflow import chatbot
from langchain_core.messages import HumanMessage
config= {"configurable":{'thread_id':'1'}}

# streamlit has session state
if 'messages' not in st.session_state:
    st.session_state['messages']=[]
#loading the conversation
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            st.markdown(message['content'])
        else:
            st.text(message['content'])

user_input = st.chat_input('Enter your query')

if user_input:
    st.session_state['messages'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    query = [HumanMessage(content=user_input)]
    response = chatbot.invoke({'messages':query},config= config)
    ai_message = response['messages'][-1].content
    st.session_state['messages'].append({'role':'assistant','content':ai_message})
    with st.chat_message('ai'):
        st.markdown(ai_message)