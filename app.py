# app.py

# 1. IMPORTS
import streamlit as st
import os
import google.generativeai as genai # 公式推奨のインポート
import io 
from pathlib import Path # ファイルパス操作用
import time

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
    st.header("⚙️ 実行したいタスクを選択")
    
    selected_task = st.radio(
        "タスク:",
        ("問題を生成する", "要約を作成する", "音声を文字起こしする"),
        key="task_selection",
        index=0 
    )

    # 問題生成オプションのデフォルト値を設定
    difficulty = "標準"
    format_type = "論述形式"
    professor_focus = ""
    
    if selected_task == "問題を生成する":
        st.header("⚙️ 問題生成オプション")
        difficulty = st.selectbox("難易度を選択:", ("標準", "難しい (応用・論述)", "易しい (基本・用語)"))
        format_type = st.selectbox("問題の形式を選択:", ("論述形式", "一問一答形式", "選択式（4択）"))
        professor_focus = st.text_area("先生が特に強調していた点を入力（任意）:", "（例：過去の社会問題との関連性を問う）", height=100)
    
    button_label = selected_task 
    generate_button = st.button(button_label) 


# 7. FILE UPLOADER
uploaded_file = st.file_uploader(
    "講義のシラバス、板書、資料（PDF/画像/音声）をアップロード",
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"] 
)

# 8. AI PROCESSING LOGIC
if uploaded_file is not None and generate_button:
    
    st.session_state['processing_done'] = False # 処理開始時にフラグをリセット
    st.session_state['generated_content'] = "" # 前回の結果をクリア

    # --- プログレスバーの初期化 ---
    progress_bar = st.progress(0, text="処理を開始します...")
    # ---------------------------

    try:
        file_extension = Path(uploaded_file.name).suffix.lower() 
        contents_for_model = [] 
        gemini_uploaded_file = None 
        temp_file_path = f"temp_file{file_extension}" 

        # 1. Save uploaded file temporarily
        progress_bar.progress(10, text="ファイルを一時保存中...")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Upload file to Gemini (using genai.upload_file)
        progress_bar.progress(30, text=f"{file_extension.upper()} ファイルをGeminiにアップロード中...")
        gemini_uploaded_file = genai.upload_file(path=temp_file_path)
        progress_bar.progress(50, text="アップロード完了。解析準備中...") 

        # 3. Prepare content list based on file type
        if file_extension == ".pdf":
            progress_bar.progress(60, text="PDFテキストを抽出・分割中...")
            try:
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
                texts = text_splitter.split_documents(documents)
                context_text = "\n\n".join([t.page_content for t in texts])
                contents_for_model.append(context_text)
                contents_for_model.append(gemini_uploaded_file) 
                progress_bar.progress(70, text="PDF解析完了。AIに指示を出します...") 
            except Exception as pdf_error:
                st.error(f"PDF解析エラー: {pdf_error}")
                st.stop()

        elif file_extension in [".jpg", ".jpeg", ".png"]:
            progress_bar.progress(70, text="画像解析準備完了。AIに指示を出します...") 
            contents_for_model.append(gemini_uploaded_file) # 画像の場合はファイル参照のみでOK

        elif file_extension in [".mp3", ".wav"]:
            progress_bar.progress(70, text="音声解析準備完了。AIに指示を出します...") 
            contents_for_model.append(gemini_uploaded_file) # 音声の場合もファイル参照のみでOK
            
        else:
            st.warning("サポートされていないファイル形式です。")
            st.stop()

        # --- Final Prompt Construction based on selected_task - Step 2の修正を適用 ---
        
        final_prompt_text = "" 
        model_name = 'models/gemini-1.5-flash' # デフォルトモデル
        
        if selected_task == "問題を生成する":
            final_prompt_text = f"""
            あなたは**{Path(uploaded_file.name).stem}**の専門家です。
            【生成ルール】: 難易度: {difficulty} / 形式: {format_type} / 焦点: {professor_focus}
            このルールに従い、問題と模範解答を計5問作成してください。
            """
        elif selected_task == "要約を作成する":
            final_prompt_text = f"""
            以下の資料（ファイル名: {uploaded_file.name}）の内容を理解し、重要なポイントを箇条書きで300字程度に要約してください。
            """
            if file_extension in [".mp3", ".wav"]:
                 st.info("音声を文字起こししてから要約します...")
                 # Gemini 1.5 Flashは音声入力から直接要約可能なので、特別な指示は不要な場合が多い

        elif selected_task == "音声を文字起こしする":
            if file_extension in [".mp3", ".wav"]:
                final_prompt_text = f"""
                以下の音声ファイルの内容を正確に文字起こししてください。話者分離は不要です。テキストのみを出力してください。
                """
                # model_name = 'models/gemini-1.5-flash' # Flashでも可能
            else:
                st.warning("文字起こしは音声ファイル（MP3, WAV）のみ対応しています。")
                if gemini_uploaded_file: 
                    try: genai.delete_file(gemini_uploaded_file.name) 
                    except Exception: pass
                if os.path.exists(temp_file_path): os.remove(temp_file_path)
                st.stop()
                
        else:
            st.error("未定義のタスクが選択されました。")
            st.stop()
        # ------------------------------------
            
        # 最終指示をリストの先頭に追加
        contents_for_model.insert(0, final_prompt_text)

        # Initialize the generative model
        model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025') 

        # Generate content request
        progress_bar.progress(80, text="AIが処理中です... (時間がかかる場合があります)") 
        try:
            response = model.generate_content(contents_for_model)
            if response.parts:
                 st.session_state['generated_content'] = response.text
                 progress_bar.progress(95, text="AIによる処理完了！") 
            else:
                 feedback_reason = "不明な理由"
                 try:
                     if response.prompt_feedback and response.prompt_feedback.block_reason:
                         feedback_reason = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
                 except Exception:
                     pass 
                 st.error(f"AIが応答を生成できませんでした。理由: {feedback_reason}")
                 st.session_state['generated_content'] = f"エラー: AI応答の取得に失敗しました ({feedback_reason})"
                 progress_bar.empty() # エラー時はバーを消す
                 st.stop()

        except Exception as e:
            st.error(f"AI生成エラーが発生しました: {e}")
            st.session_state['generated_content'] = f"エラー: {e}" 
            progress_bar.empty()
            st.stop()

    except Exception as e:
        st.error(f"ファイル処理またはアップロードエラー: {e}")
        progress_bar.empty() # エラー時はバーを消す
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path): os.remove(temp_file_path) # Ensure cleanup on error
        st.stop() 

    finally: 
         # --- Cleanup ---
         # Cleanup happens regardless of success or failure in try block if file was uploaded
         if gemini_uploaded_file:
             try:
                 genai.delete_file(gemini_uploaded_file.name) 
             except Exception as cleanup_error:
                 st.warning(f"Geminiファイル削除中にエラー: {cleanup_error}") 
         
         if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             os.remove(temp_file_path) 
         
         # Update progress bar only if no critical error stopped execution earlier
         if 'st' in locals() and hasattr(st, 'session_state') and st.session_state.get('generated_content', "") and not st.session_state['generated_content'].startswith("エラー:"):
              progress_bar.progress(100, text="処理完了！") 
              time.sleep(1) # Show complete message briefly
              progress_bar.empty() # Clear the progress bar
         elif 'progress_bar' in locals():
              progress_bar.empty() # Clear progress bar on error too
         
         st.session_state['processing_done'] = True # Always mark as done in finally


# 9. DISPLAY AI GENERATED RESULT
if st.session_state['generated_content']:
    st.header("--- AI生成結果 ---")
    st.markdown(st.session_state['generated_content'])
