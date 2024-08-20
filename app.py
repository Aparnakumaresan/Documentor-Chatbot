from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import uuid

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
parser = StrOutputParser()

# Initialize models
Model = "llama-3.1-70b-versatile"
model = ChatGroq(api_key=groq_api_key, model=Model, temperature=0)
llm = model | parser
key = os.getenv('key')
parser = StrOutputParser()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'fab3536a6bda9fb8bc80f4f5fcf279f756b8982af701d839'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global storage
pdf_text_storage = {}
chat_history = {}
current_conversation_id = None
previous_conversations = {}

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Function to split text into chunks for vectorization
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store vector embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context". Don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_files')
def upload_files_page():
    return render_template('upload.html')

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    file_ids = []

    for file in files:
        if file.filename == '':
            continue

        if file and file.filename.endswith('.pdf'):
            file_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.pdf')
            file.save(file_path)
            pdf_text_storage[file_id] = extract_text_from_pdf(file_path)
            file_ids.append(file_id)
        else:
            return jsonify({"error": "Invalid file type"}), 400

    combined_text = "\n\n".join(pdf_text_storage.values())
    text_chunks = get_text_chunks(combined_text)
    get_vector_store(text_chunks)

    return jsonify({"message": "Files uploaded, text extracted, and vector store created", "file_ids": file_ids}), 200

@app.route('/cancel/<file_id>', methods=['DELETE'])
def cancel_file(file_id):
    if file_id in pdf_text_storage:
        del pdf_text_storage[file_id]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.pdf')
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"message": "File upload canceled"}), 200
    else:
        return jsonify({"error": "File not found"}), 404

@app.route('/query')
def query_page():
    conversation_id = request.args.get('conversation_id')
    if conversation_id:
        global current_conversation_id
        current_conversation_id = conversation_id
    return render_template('query.html', chat_history=chat_history.get(current_conversation_id, []))

@app.route('/ask', methods=['POST'])
def ask_question():
    global chat_history
    global current_conversation_id
    if current_conversation_id is None:
        current_conversation_id = str(uuid.uuid4())
        chat_history[current_conversation_id] = []

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not pdf_text_storage:
        return jsonify({"error": "PDF text not available"}), 400

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    if current_conversation_id not in chat_history:
        chat_history[current_conversation_id] = []
    
    chat_history[current_conversation_id].append({"question": question, "response": response["output_text"]})

    return jsonify({"response": response["output_text"]}), 200

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history.clear()  # Clear the chat history
    global current_conversation_id
    current_conversation_id = None  # Clear the current conversation ID
    return jsonify({"message": "Chat history cleared"}), 200

@app.route('/delete_all_files', methods=['DELETE'])
def delete_all_files():
    try:
        for file_id in list(pdf_text_storage.keys()):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.pdf')
            if os.path.exists(file_path):
                os.remove(file_path)
            del pdf_text_storage[file_id]
        return jsonify({"message": "All files deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete', methods=['POST'])
def delete_files():
    files_deleted = []
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            files_deleted.append(file)
    return jsonify({"message": "Files deleted.", "files": files_deleted})

@app.route('/clear_all', methods=['POST'])
def clear_all():
    pdf_text_storage.clear()
    chat_history.clear()
    return jsonify({"message": "All data cleared"}), 200

@app.route('/conversations')
def list_conversations():
    return jsonify({"conversations": list(chat_history.keys())})

@app.route('/conversation/<conversation_id>')
def get_conversation(conversation_id):
    return jsonify(chat_history.get(conversation_id, []))

@app.route('/view_conversations')
def view_conversations():
    return render_template('view_conversations.html', previous_conversations=previous_conversations)

@app.route('/view_conversation/<conversation_id>')
def view_conversation(conversation_id):
    return render_template('query.html', chat_history=chat_history.get(conversation_id, []))

@app.route('/start_new_conversation', methods=['POST'])
def start_new_conversation():
    global chat_history, previous_conversations, current_conversation_id
    # Save the current conversation
    if current_conversation_id:
        previous_conversations[current_conversation_id] = chat_history[current_conversation_id]
    # Clear current conversation and start a new one
    current_conversation_id = str(uuid.uuid4())
    chat_history[current_conversation_id] = []
    return jsonify({'message': 'New conversation started'})

if __name__ == "__main__":
    app.run(debug=True)
