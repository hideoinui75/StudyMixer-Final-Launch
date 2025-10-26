# app.py

# --- 1. IMPORTS ---
import streamlit as st
import os
import google.generativeai as genai # 公式推奨
import io 
from pathlib import Path 
import time 

# LangChain and PDF processing imports (PDF処理にのみ使用)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# --- 2. THEME CONFIG ---
st.set_page_config(
    page_title="Study-Mixer",
    page_icon="📚", 
    layout="wide"
)

# --- 3. SESSION STATE INITIALIZATION ---
if 'generated_content' not in st.session_state:
    st.session_state['generated_content'] = ""
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = [] 
if 'displayed_history_index' not in st.session_state:
    st.session_state.displayed_history_index = None
if 'last_generated_ref' not in st.session_state: 
    st.session_state.last_generated_ref = None
if 'just_saved_history' not in st.session_state:
    st.session_state.just_saved_history = False

# --- FUNCTION: HISTORY CALLBACK (ファイルの先頭に定義) ---
def save_history_entry(generated_text, ai_success, selected_task, difficulty, format_type, professor_focus, summary_length, uploaded_file):
    """AI処理が成功した場合にのみ履歴を保存する関数"""
    
    if not ai_success:
        st.sidebar.warning("AI処理が成功しなかったため、履歴は保存されませんでした。", icon="⚠️")
        return 

    try:
        current_options = {}
        if selected_task == "問題を生成する":
             current_options = {"難易度": difficulty, "形式": format_type, "焦点": professor_focus}
        elif selected_task == "要約を作成する":
             current_options = {"長さ": summary_length}
        elif selected_task == "リアクションペーパー作成": # ★ このelifブロックを追加 ★
             current_options = {"文字数": report_words, "語彙/トーン": report_vocab}     

        history_entry = {
            "file_name": uploaded_file.name, "task": selected_task,
            "options": current_options, "result": generated_text
        }

        # 厳密な重複チェック (最新のエントリと比較)
        is_duplicate = False
        if st.session_state['analysis_history']:
            last_entry = st.session_state['analysis_history'][0]
            if (last_entry['file_name'] == history_entry['file_name'] and
                last_entry['task'] == history_entry['task'] and
                len(last_entry['result']) == len(history_entry['result'])):
                is_duplicate = True

        if not is_duplicate:
            st.session_state['analysis_history'].insert(0, history_entry)
            st.session_state.just_saved_history = True # 成功フラグを立てる

            # 履歴の最大件数を制限
            MAX_HISTORY = 10
            if len(st.session_state['analysis_history']) > MAX_HISTORY:
                st.session_state['analysis_history'] = st.session_state['analysis_history'][:MAX_HISTORY]
        
    except Exception as hist_e:
         st.sidebar.error(f"履歴保存中にエラーが発生しました: {hist_e}")

# --- 4. APP SETUP ---
st.title("📚 Study-Mixer - AI学習支援")
st.markdown("---")

# --- 5. API KEY CONFIGURATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError: 
    st.error("エラー: .streamlit/secrets.toml に GEMINI_API_KEY が設定されていません。")
    st.stop()
except Exception as e:
    st.error(f"APIキーの設定中に予期せぬエラーが発生しました: {e}")
    st.stop()

# --- 6. UI CONTROLS (Sidebar) ---
with st.sidebar:
    st.header("⚙️ 実行したいタスクを選択")
    
    selected_task = st.radio(
        "タスク:",
        ("問題を生成する", "要約を作成する", "音声を文字起こしする", "リアクションペーパー作成"), # ★★★ この行を修正 ★★★
        key="task_selection",
        index=0 
    )

    # Initialize options outside the conditional block
    difficulty = "標準"
    format_type = "論述形式"
    professor_focus = ""
    summary_length = "普通" 
    report_words = 400 # ★ 初期値を追加 ★
    report_vocab = "学術的、客観的" # ★ 初期値を追加 ★
    
    if selected_task == "問題を生成する":
        st.header("⚙️ 問題生成オプション")
        difficulty = st.selectbox("難易度を選択:", ("標準", "難しい (応用・論述)", "易しい (基本・用語)"), key="difficulty_select")
        format_type = st.selectbox("問題の形式を選択:", ("論述形式", "一問一答形式", "選択式（4択）"), key="format_select")
        professor_focus = st.text_area("先生が特に強調していた点を入力（任意）:", "（例：過去の社会問題との関連性を問う）", height=100, key="focus_input")
    elif selected_task == "要約を作成する":
         st.header("⚙️ 要約オプション")
         summary_length = st.select_slider("要約の長さ:", ["短め", "普通", "長め"], value="普通", key="summary_slider")
    elif selected_task == "リアクションペーパー作成": # ★ このelifブロックを追加 ★
         st.header("📝 ペーパー作成オプション")
         # 文字数をスライダーで指定
         report_words = st.slider("文字数 (目安):", min_value=100, max_value=1500, value=400, step=50, key="report_words")
         # 語彙/トーンを選択肢で指定
         report_vocab = st.selectbox("語彙/トーン:", ("学術的、客観的", "意欲的、前向き", "批判的、分析的", "簡潔、論理的","簡潔な感想、学生向け"), key="report_vocab")

    button_label = selected_task 
    generate_button = st.button(button_label, key="generate_button") 

    # History Selection (Display in Sidebar)
    st.markdown("---") 
    st.header("📄 分析履歴")

    if not st.session_state['analysis_history']:
        st.caption("まだ履歴はありません。")
    else:
        # Generate display titles (with less truncation than before)
        history_titles = []
        for i, entry in enumerate(st.session_state['analysis_history']):
            display_str = f"{entry['task']} ({entry['file_name'][:15]}...)"

            if entry['task'] == "問題を生成する":
                # ... (オプションのif/elif構造は正しくインデントされていることを確認) ...
                difficulty_str = entry['options'].get('難易度', '不明')
                format_str = entry['options'].get('形式', '不明')
                display_str = f"[{difficulty_str} / {format_str}] {entry['file_name'][:15]}..."
            elif entry['task'] == "要約を作成する":
                 length_str = entry['options'].get('長さ', '不明')
                 display_str = f"[要約: {length_str}] {entry['file_name'][:15]}..."
            elif entry['task'] == "リアクションペーパー作成":
                 words_str = entry['options'].get('文字数', '不明')
                 vocab_str = entry['options'].get('語彙/トーン', '不明')
                 display_str = f"[リアぺ: {words_str}字 / {vocab_str}] {entry['file_name'][:15]}..."


            history_titles.append(f"{i+1}: {display_str}") # ← for ループの処理はここで終了


        options_with_placeholder = ["履歴を選択..."] + history_titles # ★★★ この行のインデントを左に揃える ★★★

        selected_history_display = st.selectbox(
            "過去の分析結果を選択:",
            options=options_with_placeholder,
            index=0, 
            key="history_selectbox_display"
        )

        if selected_history_display != "履歴を選択...":
            try:
                # Find the corresponding entry
                selected_index = options_with_placeholder.index(selected_history_display) - 1 # -1 for placeholder
                selected_entry = st.session_state['analysis_history'][selected_index]
                
                # --- サイドバーでの詳細表示 ---
                st.subheader(f"選択履歴 {selected_index + 1} の詳細:")
                st.caption(f"**タスク:** {selected_entry['task']}")
                st.caption(f"**ファイル:** {selected_entry['file_name']}")
                
                if selected_entry['options']:
                    st.markdown("---")
                    st.caption("**実行オプション:**")
                    for k, v in selected_entry['options'].items():
                         st.write(f"- **{k}**: {v}")
                
                # --- メインエリアの表示を更新 ---
                if st.session_state.get('displayed_history_index') != selected_index:
                    st.session_state['generated_content'] = selected_entry['result']
                    st.session_state.displayed_history_index = selected_index
                    st.rerun() 

            except (ValueError, IndexError):
                 st.sidebar.warning("履歴の表示中にエラーが発生しました。", icon="⚠️")

# --- 7. FILE UPLOADER ---
def reset_history_selection_on_upload():
    st.session_state.displayed_history_index = None

uploaded_file = st.file_uploader(
    "講義のシラバス、板書、資料（PDF/画像/音声）をアップロード",
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"],
    key="file_uploader",
    on_change=reset_history_selection_on_upload
)

# --- 8. AI PROCESSING LOGIC ---
if generate_button and uploaded_file is not None:

    # Reset states
    st.session_state.just_saved_history = False # Reset flag for display
    st.session_state['generated_content'] = "" 
    st.session_state.displayed_history_index = None 

    progress_bar = st.progress(0, text="処理を開始します...")

    # Initialize variables used across try/except/cleanup
    generated_text = "" 
    ai_success = False
    gemini_uploaded_file = None
    temp_file_path = "" 

    try: # Outer try for all processing steps
        file_extension = Path(uploaded_file.name).suffix.lower() 
        contents_for_model = [] 
        temp_file_path = f"temp_file{file_extension}" 

        progress_bar.progress(10, text="ファイルを一時保存中...")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar.progress(30, text=f"{file_extension.upper()} ファイルをGeminiにアップロード中...")
        gemini_uploaded_file = genai.upload_file(path=temp_file_path, display_name=uploaded_file.name)
        progress_bar.progress(50, text="アップロード完了。解析準備中...") 

        # 3. Prepare content list (File Handling)
        if file_extension == ".pdf":
            progress_bar.progress(60, text="PDFテキストを抽出・分割中...")
            try:
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 
                texts = text_splitter.split_documents(documents)
                if not texts: 
                     raise ValueError("PDFからテキストを抽出できませんでした。")
                context_text = "\n\n".join([t.page_content for t in texts])
                contents_for_model.append(context_text)
                contents_for_model.append(gemini_uploaded_file) 
                progress_bar.progress(70, text="PDF解析完了。AIに指示を出します...") 
            except Exception as pdf_error:
                raise Exception(f"PDF解析エラー: {pdf_error}") 

        elif file_extension in [".jpg", ".jpeg", ".png"]:
            progress_bar.progress(70, text="画像解析準備完了。AIに指示を出します...") 
            contents_for_model.append(gemini_uploaded_file) 

        elif file_extension in [".mp3", ".wav"]:
            progress_bar.progress(70, text="音声解析準備完了。AIに指示を出します...") 
            contents_for_model.append(gemini_uploaded_file) 
            
        else: 
            st.warning("サポートされていないファイル形式です。")
            raise Exception("Unsupported file format") 

        # --- Final Prompt Construction ---
        final_prompt_text = "" 
        
        if selected_task == "問題を生成する":
            final_prompt_text = f"あなたは**{Path(uploaded_file.name).stem}**の専門家です。\n【生成ルール】: 難易度: {difficulty} / 形式: {format_type} / 焦点: {professor_focus}\nこのルールに従い、問題と模範解答を計5問作成してください。"
        elif selected_task == "要約を作成する":
            length_map = {"短め": "150字程度", "普通": "300字程度", "長め": "500字程度"}
            final_prompt_text = f"以下の資料（ファイル名: {uploaded_file.name}）の内容を理解し、重要なポイントを箇条書きで{length_map.get(summary_length, '300字程度')}に要約してください。"
            if file_extension in [".mp3", ".wav"]:
                 st.info("音声を文字起こししてから要約します...")

        elif selected_task == "リアクションペーパー作成": # ★ このelifブロックを追加 ★
            tone_instruction = ""
            if report_vocab == "簡潔な感想、学生向け":
                 # 語尾と文体に対する具体的な指示を追加
                 tone_instruction = "語彙は『〜と思う』『〜と感じた』『〜と考えている』といった主観的な表現を多く含め、AIとわかりづらい文にしてください。"
            final_prompt_text = f"""以下の資料を基に、講義の核心とあなたの考察を含むリアクションペーパー（感想文）を作成してください。
            【生成ルール】: 1. 文字数目安: {report_words}文字 / 2. 語彙/トーン: {report_vocab} / 3. 結論から書き始めてください。"""        
          
        elif selected_task == "音声を文字起こしする":
            if file_extension in [".mp3", ".wav"]:
                final_prompt_text = "以下の音声ファイルの内容を正確に文字起こししてください。話者分離は不要です。テキストのみを出力してください。"
            else:
                st.warning("文字起こしは音声ファイル（MP3, WAV）のみ対応しています。")
                raise Exception("Transcription only supports audio files")
        
        else: 
            st.error("未定義のタスクが選択されました。")
            raise Exception("Undefined task selected")
            
        contents_for_model.insert(0, final_prompt_text)

        # --- AI Request ---
        model = genai.GenerativeModel('models/gemini-2.5-flash') 
        progress_bar.progress(80, text="AIが処理中です... (時間がかかる場合があります)") 

        try: # Inner try for AI request
            response = model.generate_content(contents_for_model, request_options={"timeout": 600}) 

            if hasattr(response, 'text') and response.text:
                 generated_text = response.text 
                 ai_success = True 
                 progress_bar.progress(95, text="AIによる処理完了！") 
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 feedback_reason = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
                 error_message = f"AIが応答を生成できませんでした。理由: {feedback_reason}"
                 st.error(error_message)
                 generated_text = error_message 
            else: 
                 error_message = "AIからの応答が空か、予期せぬ形式でした。"
                 st.error(error_message)
                 generated_text = error_message

        except Exception as ai_e: # Catch AI specific errors
            error_message = f"AI生成エラーが発生しました: {ai_e}"
            st.error(error_message)
            generated_text = error_message 

    # --- Catch All Processing Errors (Outer Except Block) ---
    except Exception as e:
        error_message = f"処理中にエラーが発生しました: {e}"
        st.error(error_message)
        generated_text = error_message # Ensure generated_text exists

    # --- Save Result to Session State (Always happens) ---
    st.session_state['generated_content'] = generated_text

    # --- 履歴保存処理の呼び出し (重複防止のため) ---
    # save_history_entry関数がファイル上部に定義されている前提
    save_history_entry(generated_text, ai_success, selected_task, difficulty, format_type, professor_focus, summary_length, uploaded_file)
    # -----------------------------------------

    # --- Final Cleanup Block ---
    # This block executes sequentially after history saving

    # --- Cleanup ---
    if gemini_uploaded_file:
        try: genai.delete_file(gemini_uploaded_file.name)
        except Exception as cleanup_error: st.warning(f"Geminiファイル削除中にエラー: {cleanup_error}")
    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # --- Progress Bar Update ---
    if ai_success:
            progress_bar.progress(100, text="処理完了！")
            time.sleep(1)
    if 'progress_bar' in locals(): progress_bar.empty()


# --- 9. DISPLAY AI GENERATED RESULT ---
if st.session_state['generated_content']:
    st.header("--- AI生成結果 ---")
    st.markdown(st.session_state['generated_content'])
