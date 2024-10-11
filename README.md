# Prerequisites: 
python>=3.10
# Instruction and Information:
- Install dependencies packages by using `pip install -r requirements.txt`
- Create vector store and save locally in folder `chroma_langchain_db` if there are more documents by running `python create_vectorstore.py`
- Update your OPENAI API KEY in `.env` with variable `OPENAI_API_KEY = "your-key"`
- The history chat is stored in `history.json`
- Run programatically `python main.py`. Then go to Swagger UI to test your question by the link `http://127.0.0.1:8000/docs`
# Limitation:
- Not yet handle log 
- Not yet implement hybrid search
- Not yet self-refection RAG (halluciation, irrelevant question to documents)