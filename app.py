# app.py

# 1. ライブラリのインポート (全てここに集める)
import streamlit as st
import os
from google import genai
import io 

# LangChain and PDF processing imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# 2. ページ全体の基本設定とテーマ（白黒）
st.set_page_config(
    page_title="Study-Mixer",
    page_icon="📚", 
    layout="wide"
)

# 3. セッションステートの初期化 (記憶の箱の準備)
if 'generated_content' not in st.session_state:
    st.session_state['generated_content'] = ""

# 4. APP SETUP
st.title("💡 Study-Mixer - 資料形式を選ばないAI学習支援")
st.markdown("---")

# 5. API KEY AND MODEL INITIALIZATION
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("エラー: .streamlit/secrets.toml に APIキーが設定されていません。")
    st.stop()
    
client = genai.Client(api_key=API_KEY)


# 6. UI CONTROLS (Sidebar)
with st.sidebar:
    st.header("⚙️ 問題生成オプション")
    
    difficulty = st.selectbox(
        "難易度を選択:",
        ("標準", "難しい (応用・論述)", "易しい (基本・用語)")
    )
    
    format_type = st.selectbox(
        "問題の形式を選択:",
        ("論述形式", "一問一答形式", "選択式（4択）")
    )

    professor_focus = st.text_area(
        "先生が特に強調していた点を入力（任意）:",
        "（例：過去の社会問題との関連性を問う）",
        height=100
    )

    generate_button = st.button("問題を生成する")


# 7. FILE UPLOADER
uploaded_file = st.file_uploader(
    "講義のシラバス、板書、資料（PDF/画像/音声）をアップロード",
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"] 
)

# 8. AI PROCESSING LOGIC
if uploaded_file is not None and generate_button:
    
    # 処理中の表示
    with st.spinner("資料を解析し、AIが問題を生成中です..."):
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        contents_to_gemini = [] # AIに渡すコンテンツリスト
        uploaded_gemini_file = None # Gemini APIにアップロードされたファイルオブジェクト
        temp_file_name = f"temp_file.{file_extension}" # 一時ファイル名を設定

        # --- File Type Handling ---
        try:
            # 1. データを一時ファイルとしてローカルに保存（Geminiのアップロード機能に対応させるため）
            with open(temp_file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. Geminiにファイルをアップロードし、一時的なGemini File Objectを取得
            uploaded_gemini_file = client.upload_file(file=temp_file_name)
            
            # 3. コンテンツリストの構築
            if file_extension == "pdf":
                st.info("PDFファイルを解析中...")
                
                # PDFテキストの分割とRAGコンテキストの構築
                loader = PyPDFLoader(temp_file_name)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
                texts = text_splitter.split_documents(documents)
                context_text = "\n\n".join([t.page_content for t in texts])
                
                # テキストコンテキストとアップロード済みファイルオブジェクトを渡す
                contents_to_gemini.append(context_text)
                contents_to_gemini.append(uploaded_gemini_file) # PDFファイル自体も参照させる

            elif file_extension in ["jpg", "jpeg", "png"]:
                st.info("画像を解析中...")
                contents_to_gemini.append("以下の画像は講義の板書または重要な図です。この画像の内容を完全に理解し、それに基づいた問題を生成してください。")
                contents_to_gemini.append(uploaded_gemini_file)

            elif file_extension in ["mp3", "wav"]:
                st.info("音声を文字起こし・解析中...（大容量ファイルは時間がかかる場合があります）")
                contents_to_gemini.append("以下の音声ファイルは講義の録音です。まず内容を完全に文字起こしし、その文字起こし内容だけを参照して問題を生成してください。")
                contents_to_gemini.append(uploaded_gemini_file)
                
            else:
                st.warning("サポートされていないファイル形式です。PDF、画像、音声ファイルをご利用ください。")
                
        except Exception as e:
            st.error(f"ファイル処理エラーが発生しました: {e}")
            st.stop()
            
        # --- Final Prompt Construction and Execution ---
        
        final_prompt_text = f"""
        あなたは**{uploaded_file.name.replace(f".{file_extension}", "")}**の専門家です。
        【生成ルール】: 難易度: {difficulty} / 形式: {format_type} / 焦点: {professor_focus}
        このルールに従い、問題と模範解答を計5問作成してください。
        """
        # 最終指示をリストの先頭に追加
        contents_to_gemini.insert(0, final_prompt_text)

        # Gemini API Request
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_to_gemini
            )
            # Save result to session state
            st.session_state['generated_content'] = response.content
        except Exception as e:
            st.error(f"AI生成エラーが発生しました: {e}")
            st.stop()


    # --- Cleanup ---
    if uploaded_gemini_file:
        # Geminiサーバー上の一時ファイルを削除
        client.delete_file(uploaded_gemini_file.name)
    
    # ローカルの一時ファイルを削除 (ファイル名が正しく設定されていることを確認)
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name) 

    # Rerun to display latest result instantly
    st.rerun()

# 9. DISPLAY AI GENERATED RESULT (At the end of the script)
if st.session_state['generated_content']:
    st.header("--- AI生成結果 ---")
    st.markdown(st.session_state['generated_content'])
