import streamlit as st
from QAgen import process_query

def main():
    st.set_page_config(page_title="CUDA Query Assistant", page_icon="🚀", layout="wide")
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.image("https://developer.nvidia.com/sites/default/files/akamai/cuda/images/cuda_logo.jpg", width=200)
    st.sidebar.title("CUDA Query Assistant")
    st.sidebar.info(
        "This app uses a hybrid retrieval system combining BM25 and dense retrieval, "
        "followed by cross-encoder reranking and GPT-3.5 for answer generation. "
        "It's designed to answer questions about CUDA programming based on the official documentation."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("Ask Your CUDA Questions")
        query = st.text_input("Enter your CUDA-related question:", key="query_input")
        
        if 'history' not in st.session_state:
            st.session_state.history = []

        if st.button("Get Answer", key="submit_button"):
            if query:
                with st.spinner("Processing your query..."):
                    answer = process_query(query)
                    st.session_state.history.append((query, answer))
                st.success("Answer generated!")
            else:
                st.warning("Please enter a question.")
        
        if st.session_state.history:
            st.subheader("Latest Answer")
            st.info(st.session_state.history[-1][1])

    with col2:
        st.subheader("Query History")
        for i, (q, a) in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {q[:50]}{'...' if len(q) > 50 else ''}"):
                st.write(f"A: {a}")

    st.markdown("---")
    st.markdown("Developed with ❤️ by Your Team | Powered by Streamlit")

if __name__ == "__main__":
    main()