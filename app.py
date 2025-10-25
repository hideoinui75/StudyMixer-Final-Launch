# app.py

# 1. IMPORTS
import streamlit as st
import os
import google.generativeai as genai # 公式推奨のインポート
import io 
from pathlib import Path # ファイルパス操作用

# LangChain and PDF processing imports (PDF処理にのみ使用)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# 2. THEME CONFIG
# NOTE: This must be at the very top of the script execution
st.set_page_config(
    page_title="Study-Mixer",
    page_icon="📚", 
    layout="wide"
)

# 3. SESSION STATE INITIALIZATION
if 'generated_content' not in st.session_state:
    st.session_state['generated_content'] = ""

# 4. APP SETUP
st.title("💡 Study-Mixer - 資料形式を選ばないAI学習支援")
st.markdown("---")

# 5. API KEY CONFIGURATION (公式推奨の方法)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError: # More specific error handling
    st.error("エラー: .streamlit/secrets.toml に GEMINI_API_KEY が設定されていません。")
    st.stop()
except Exception as e:
    st.error(f"APIキーの設定中に予期せぬエラーが発生しました: {e}")
    st.stop()

# 6. UI CONTROLS (Sidebar)
with st.sidebar:
    st.header("⚙️ 問題生成オプション")
    difficulty = st.selectbox("難易度を選択:", ("標準", "難しい (応用・論述)", "易しい (基本・用語)"))
    format_type = st.selectbox("問題の形式を選択:", ("論述形式", "一問一答形式", "選択式（4択）"))
    professor_focus = st.text_area("先生が特に強調していた点を入力（任意）:", "（例：過去の社会問題との関連性を問う）", height=100)
    generate_button = st.button("問題を生成する")

# 7. FILE UPLOADER
uploaded_file = st.file_uploader(
    "講義のシラバス、板書、資料（PDF/画像/音声）をアップロード",
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"] 
)

# 8. AI PROCESSING LOGIC
if uploaded_file is not None and generate_button:
    
    st.session_state['processing_done'] = False # Reset flag on new generation
    st.session_state['generated_content'] = "" # Clear previous results

    # Processing Status
    with st.spinner("資料を解析し、AIが問題を生成中です..."):
        
        file_extension = Path(uploaded_file.name).suffix.lower() 
        contents_for_model = [] 
        gemini_uploaded_file = None 
        temp_file_path = f"temp_file{file_extension}" # Use unique temp file name

        try:
            # 1. Save uploaded file temporarily
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. Upload file to Gemini (using genai.upload_file)
            st.info(f"{file_extension.upper()} ファイルをアップロード中...")
            gemini_uploaded_file = genai.upload_file(path=temp_file_path)
            st.info("アップロード完了。AIによる解析を開始します...")

            # 3. Prepare content list based on file type
            if file_extension == ".pdf":
                # PDF processing (using LangChain for text extraction only)
                try:
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
                    texts = text_splitter.split_documents(documents)
                    context_text = "\n\n".join([t.page_content for t in texts])
                    contents_for_model.append(context_text)
                    contents_for_model.append(gemini_uploaded_file) 
                except Exception as pdf_error:
                    st.error(f"PDF解析エラー: {pdf_error}")
                    st.stop()

            elif file_extension in [".jpg", ".jpeg", ".png"]:
                contents_for_model.append("以下の画像は講義の板書または重要な図です。この画像の内容を完全に理解し、それに基づいた問題を生成してください。")
                contents_for_model.append(gemini_uploaded_file)

            elif file_extension in [".mp3", ".wav"]:
                contents_for_model.append("以下の音声ファイルは講義の録音です。まず内容を完全に文字起こしし、その文字起こし内容だけを参照して問題を生成してください。")
                contents_for_model.append(gemini_uploaded_file)
                
            else:
                st.warning("サポートされていないファイル形式です。")
                if os.path.exists(temp_file_path): os.remove(temp_file_path) # Clean up temp file
                st.stop()

        except Exception as e:
            st.error(f"ファイル処理またはアップロードエラー: {e}")
            if os.path.exists(temp_file_path): os.remove(temp_file_path)
            st.stop()

        # --- Final Prompt Construction and Execution ---
        
        final_prompt_text = f"""
        あなたは**{Path(uploaded_file.name).stem}**の専門家です。
        【生成ルール】: 難易度: {difficulty} / 形式: {format_type} / 焦点: {professor_focus}
        このルールに従い、問題と模範解答を計5問作成してください。
        """
        contents_for_model.insert(0, final_prompt_text)

        # Initialize the generative model
        # Use a model known for multimodal capabilities if needed, e.g., gemini-1.5-flash
        model = genai.GenerativeModel('models/gemini-2.5-flash') # Ensure multimodal model

        # Generate content request
        try:
            response = model.generate_content(contents_for_model)
            # Handle potential response errors or blocks
            if response.parts:
                 st.session_state['generated_content'] = response.text
            else:
                 # Attempt to access prompt_feedback for blocking reasons
                 feedback_reason = "不明な理由"
                 try:
                     if response.prompt_feedback and response.prompt_feedback.block_reason:
                         feedback_reason = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
                 except Exception:
                     pass # Ignore if feedback structure is unexpected
                 st.error(f"AIが応答を生成できませんでした。理由: {feedback_reason}")
                 st.session_state['generated_content'] = f"エラー: AI応答の取得に失敗しました ({feedback_reason})"

        except Exception as e:
            st.error(f"AI生成エラーが発生しました: {e}")
            st.session_state['generated_content'] = f"エラー: {e}" # Store error message
            st.stop()

        finally: 
             # --- Cleanup ---
            if gemini_uploaded_file:
                try:
                    genai.delete_file(gemini_uploaded_file.name) 
                except Exception as cleanup_error:
                    st.warning(f"Geminiファイル削除中にエラー: {cleanup_error}") 
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path) 
            

# 9. DISPLAY AI GENERATED RESULT
if st.session_state['generated_content']:
    st.header("--- AI生成結果 ---")
    st.markdown(st.session_state['generated_content'])
