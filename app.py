# app.py

# --- 1. IMPORTS ---
import streamlit as st
import os
import google.generativeai as genai
import io 
from pathlib import Path 
import time 
# import json # 永続化機能削除のためコメントアウト

# LangChain and PDF processing imports (PDF処理にのみ使用)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# --- 2. THEME CONFIG ---
st.set_page_config(
    page_title="Study-Mixer",
    page_icon="📚", 
    layout="wide"
)
st.markdown(
"""
<style>
[data-testid="stSidebar"] {
width: 350px !important; /* ここで幅を指定 (例: 350px) */
}
</style>
""",
unsafe_allow_html=True,
)

# --- 3. PERSISTENCE FUNCTIONS (削除) ---
# load_data, save_data 関数を削除


# --- 4. HISTORY SAVE FUNCTION ---
# 履歴保存関数 (引数 file_name_str を受け取るように変更)
def save_history_entry(generated_text, ai_success, selected_task, difficulty, format_type, professor_focus, summary_length, file_name_str, selected_course, report_words=None, report_vocab=None, report_keywords=None):
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
        elif selected_task == "リアクションペーパー作成":
            current_options = {"文字数": report_words, "語彙/トーン": report_vocab, "強調キーワード": report_keywords}      

        history_entry = {
            "file_name": file_name_str, # 連結されたファイル名文字列
            "task": selected_task,
            "course": selected_course, # 講義フォルダ機能
            "options": current_options, 
            "result": generated_text
        }

        # 厳密な重複チェック (最新のエントリと比較)
        is_duplicate = False
        if st.session_state['analysis_history']:
            last_entry = st.session_state['analysis_history'][0]
            if (last_entry['file_name'] == history_entry['file_name'] and
                last_entry['task'] == history_entry['task'] and
                last_entry['course'] == history_entry['course'] and # 講義フォルダもチェック
                len(last_entry['result']) == len(history_entry['result'])):
                is_duplicate = True

        if not is_duplicate:
            st.session_state['analysis_history'].insert(0, history_entry)
            st.session_state.just_saved_history = True # 成功フラグを立てる

            # 履歴の最大件数を制限
            MAX_HISTORY = 100
            if len(st.session_state['analysis_history']) > MAX_HISTORY:
                st.session_state['analysis_history'] = st.session_state['analysis_history'][:MAX_HISTORY]
            
            # save_data() 呼び出し削除 (永続化機能削除)
            
    except Exception as hist_e:
         st.sidebar.error(f"履歴保存中にエラーが発生しました: {hist_e}")

# --- 5. SESSION STATE INITIALIZATION ---
# 永続化（load_data）を削除し、空のリストまたはデフォルト値で初期化
if 'generated_content' not in st.session_state:
    st.session_state['generated_content'] = ""
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = [] # ★ 空のリストで初期化
if 'displayed_history_index' not in st.session_state:
    st.session_state.displayed_history_index = None
if 'last_generated_ref' not in st.session_state: 
    st.session_state.last_generated_ref = None
if 'just_saved_history' not in st.session_state:
    st.session_state.just_saved_history = False

# ★ 講義フォルダ機能 (永続化なし) ★
if 'course_folders' not in st.session_state:
    st.session_state.course_folders = ["未分類"] # ★ デフォルト値で初期化
if 'selected_course' not in st.session_state:
    st.session_state.selected_course = "未分類"

# ★ 複数ファイルアップロード（エラー回避用）リスト ★
if 'uploaded_file_list' not in st.session_state:
    st.session_state.uploaded_file_list = []

# --- 6. APP SETUP ---
st.title("💡 Study-Mixer - その資料、10秒で試験問題に。")
st.markdown("---")

# --- 7. API KEY CONFIGURATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError: 
    st.error("エラー: .streamlit/secrets.toml に GEMINI_API_KEY が設定されていません。")
    st.stop()
except Exception as e:
    st.error(f"APIキーの設定中に予期せぬエラーが発生しました: {e}")
    st.stop()

# --- 8. UI CONTROLS (Sidebar) ---
with st.sidebar:
    # --- 講義フォルダ管理 ---
    st.header("📚 講義フォルダ管理")
    new_course_name = st.text_input("新しい講義名を追加:", placeholder="例: 経済学特講", key="new_course_input")
    
    if st.button("講義フォルダを追加"):
        if new_course_name and new_course_name not in st.session_state.course_folders:
            st.session_state.course_folders.append(new_course_name)
            st.session_state.selected_course = new_course_name # 追加したものを選択
            # save_data() 呼び出し削除
            st.rerun() # UI更新のためリラン
    
    st.markdown("---") # 区切り線
    
    # 「未分類」以外のフォルダを削除対象リストとして取得
    deletable_folders = [folder for folder in st.session_state.course_folders if folder != "未分類"]
    
    if deletable_folders: # 削除できるフォルダが1つ以上ある場合のみ表示
        st.subheader("🗑️ フォルダを削除")
        
        folder_to_delete = st.selectbox(
            "削除するフォルダを選択:",
            options=deletable_folders,
            index=0,
            key="delete_folder_select"
        )
        
        # 削除ボタン
        if st.button(f"「{folder_to_delete}」を削除", key="delete_folder_button", type="primary"):
            
            # 1. フォルダリストから削除
            st.session_state.course_folders.remove(folder_to_delete)
            
            # 2. 関連する履歴を「未分類」に移動
            for entry in st.session_state['analysis_history']:
                if entry.get('course') == folder_to_delete:
                    entry['course'] = "未分類"
            
            # 3. 現在選択中のフォルダを「未分類」に戻す
            st.session_state.selected_course = "未分類"
            
            st.success(f"フォルダ「{folder_to_delete}」を削除しました。中の履歴は「未分類」に移動しました。")
            st.rerun() # UIを更新

    
    # 選択中のフォルダがリストに存在するか確認
    current_folder_index = 0
    if st.session_state.selected_course in st.session_state.course_folders:
        current_folder_index = st.session_state.course_folders.index(st.session_state.selected_course)
    elif st.session_state.course_folders:
        st.session_state.selected_course = st.session_state.course_folders[0] # 存在しない場合は先頭を選択
    
    st.session_state.selected_course = st.selectbox(
        "分析結果を保存する講義フォルダ:",
        st.session_state.course_folders,
        index=current_folder_index,
        key="course_folder_select"
    )
    st.markdown("---")
    
    # --- タスク選択 ---
    st.header("⚙️ 実行したいタスクを選択")
    
    selected_task = st.radio(
        "タスク:",
        ("問題を生成する", "要約を作成する", "音声を文字起こしする", "リアクションペーパー作成"),
        key="task_selection",
        index=0 
    )

    # --- オプション初期化 ---
    difficulty = "標準"
    format_type = "論述形式"
    professor_focus = ""
    summary_length = "普通" 
    report_words = 400 
    report_vocab = "学術的、客観的"
    report_keywords = "" # 初期化
    
    # --- タスク別オプション ---
    if selected_task == "問題を生成する":
        st.header("⚙️ 問題生成オプション")
        difficulty = st.selectbox("難易度を選択:", ("標準", "難しい (応用・論述)", "易しい (基本・用語)"), key="difficulty_select")
        format_type = st.selectbox("問題の形式を選択:", ("論述形式", "一問一答形式", "選択式（4択）"), key="format_select")
        professor_focus = st.text_area("先生が特に強調していた点を入力（任意）:", "（例：過去の社会問題との関連性を問う）", height=100, key="focus_input")
    elif selected_task == "要約を作成する":
          st.header("⚙️ 要約オプション")
          summary_length = st.select_slider("要約の長さ:", ["短め", "普通", "長め"], value="普通", key="summary_slider")
    elif selected_task == "リアクションペーパー作成":
          st.header("📝 ペーパー作成オプション")
          report_words = st.slider("文字数 (目安):", min_value=100, max_value=1500, value=400, step=50, key="report_words")
          report_vocab = st.selectbox("語彙/トーン:", ("学術的、客観的", "意欲的、前向き", "批判的、分析的", "簡潔、論理的","簡潔な感想、学生向け"), key="report_vocab")
          report_keywords = st.text_input("強調したいキーワード (複数可):", placeholder="例: Web3, 分散型社会, 倫理", key="report_keywords")

    button_label = f"{selected_task} ( {len(st.session_state.uploaded_file_list)}件の資料 )"
    generate_button = st.button(button_label, key="generate_button", use_container_width=True) 

    # --- 履歴表示 ---
    st.markdown("---") 
    st.header("📄 分析履歴")

    # 講義フォルダで履歴をフィルタリング
    filtered_history = [e for e in st.session_state['analysis_history'] if e.get('course') == st.session_state.selected_course]

    if not filtered_history:
        st.caption(f"講義フォルダ **{st.session_state.selected_course}** に履歴はありません。")
    else:
        history_titles = []
        for i, entry in enumerate(filtered_history):
            # ファイル名が長い場合も考慮
            display_str = f"[{entry['task']}] {entry['file_name'][:25]}..."
            history_titles.append(f"{i+1}: {display_str}")

        options_with_placeholder = ["履歴を選択..."] + history_titles

        selected_history_display = st.selectbox(
            f"過去の分析結果を選択 (フォルダ: {st.session_state.selected_course}):",
            options=options_with_placeholder,
            index=0, 
            key="history_selectbox_display"
        )

        if selected_history_display != "履歴を選択..." and not generate_button:
            try:
                selected_index = options_with_placeholder.index(selected_history_display) - 1
                selected_entry = filtered_history[selected_index]
                
                st.subheader(f"選択履歴 {selected_index + 1} の詳細:")
                st.caption(f"**タスク:** {selected_entry['task']}")
                st.caption(f"**ファイル:** {selected_entry['file_name']}")
                st.caption(f"**フォルダ:** {selected_entry.get('course', '未分類')}")
                
                if selected_entry['options']:
                    st.markdown("---")
                    st.caption("**実行オプション:**")
                    for k, v in selected_entry['options'].items():
                         st.write(f"- **{k}**: {v}")
                
                if st.session_state.get('displayed_history_index') != selected_index:
                    st.session_state['generated_content'] = selected_entry['result']
                    st.session_state.displayed_history_index = selected_index
                    st.rerun() 

            except (ValueError, IndexError):
                 st.sidebar.warning("履歴の表示中にエラーが発生しました。", icon="⚠️")

# --- 9. FILE UPLOADER (複数ファイル蓄積方式) ---
def reset_history_selection_on_upload():
    st.session_state.displayed_history_index = None

# ★★★ ここが「multiple=True」のエラー回避ロジック ★★★
newly_uploaded_file = st.file_uploader(
    f"資料を1つずつアップロード (保存先: **{st.session_state.selected_course}**)",
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"],
    key="file_uploader_single", # キー名を変更
    on_change=reset_history_selection_on_upload
    # multiple=True は使用しない
)

# アップロードされたファイルをリストに蓄積する
if newly_uploaded_file is not None:
    # ファイルがまだリストに追加されていないかチェック (名前とサイズで簡易的に)
    is_already_added = False
    for f in st.session_state.uploaded_file_list:
        if f.name == newly_uploaded_file.name and f.size == newly_uploaded_file.size:
            is_already_added = True
            break
            
    if not is_already_added:
        st.session_state.uploaded_file_list.append(newly_uploaded_file)
        # ファイルを追加したら、アップローダーをリセットするためにRerun
        st.rerun()

# --- アップロード済みファイルリストの表示とクリアボタン ---
if st.session_state.uploaded_file_list:
    st.subheader(f"📤 分析対象の資料リスト ({len(st.session_state.uploaded_file_list)}件)")
    
    cols = st.columns([4, 1]) # ファイル名と削除ボタン用の列
    
    index_to_remove = None
    
    for i, f in enumerate(st.session_state.uploaded_file_list):
        cols[0].caption(f"- {f.name} ({f.size // 1024} KB)")
        if cols[1].button(f"削除", key=f"remove_{i}"):
            index_to_remove = i
            
    if index_to_remove is not None:
        st.session_state.uploaded_file_list.pop(index_to_remove)
        st.rerun() # リスト更新のためRerun

    if st.button("リストをすべてクリア", key="clear_all_uploads"):
        st.session_state.uploaded_file_list = []
        st.rerun() # リスト更新のためRerun
        
st.markdown("---")

# --- 10. AI PROCESSING LOGIC ---
# ★ 処理開始のトリガーをリストの存在確認に変更 ★
if generate_button and st.session_state.uploaded_file_list:

    # 処理対象のファイルリストを取得
    uploaded_files = st.session_state.uploaded_file_list
    
    # ★★★ NameError 修正 ★★★
    # file_names_str を try ブロックの外（できるだけ早く）で定義する
    file_names_str = " + ".join([f.name for f in uploaded_files])

    # Reset states
    st.session_state.just_saved_history = False
    st.session_state['generated_content'] = "" 
    st.session_state.displayed_history_index = None 

    progress_bar = st.progress(0, text="処理を開始します...")

    # Initialize variables
    generated_text = "" 
    ai_success = False
    uploaded_gemini_files = [] # Gemini APIにアップロードされたファイルオブジェクト
    temp_file_paths = []       # クリーンアップ用の一時ファイルパス
    contents_for_model = []    # AIに渡す全コンテンツ (テキストとファイルオブジェクト)

    try: # Outer try for all processing steps
       total_files = len(uploaded_files)
        
       # --- ループで全ファイルを処理 ---
       for i, uploaded_file in enumerate(uploaded_files):
            file_extension = Path(uploaded_file.name).suffix.lower() 
            temp_file_path = f"temp_{i}_{uploaded_file.name}" 
            temp_file_paths.append(temp_file_path)

            progress_percent = int((i / total_files) * 50) # 進捗

            progress_bar.progress(progress_percent + 10, text=f"[{i+1}/{total_files}] ファイルを一時保存中...")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar.progress(progress_percent + 30, text=f"[{i+1}/{total_files}] {file_extension.upper()} ファイルをGeminiにアップロード中...")
            gemini_uploaded_file = genai.upload_file(path=temp_file_path, display_name=uploaded_file.name)
            uploaded_gemini_files.append(gemini_uploaded_file) # アップロード結果をリストに保持
            
            progress_bar.progress(progress_percent + 50, text=f"[{i+1}/{total_files}] アップロード完了。解析準備中...") 

            # --- ファイルタイプ別処理 ---
            if file_extension == ".pdf":
                progress_bar.progress(progress_percent + 60, text=f"[{i+1}/{total_files}] PDFテキストを抽出・分割中...")
                try:
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 
                    texts = text_splitter.split_documents(documents)
                    if not texts: 
                         raise ValueError(f"ファイル {uploaded_file.name} からテキストを抽出できませんでした。")
                    
                    context_text = f"--- 資料名: {uploaded_file.name} の内容 ---\n\n" + "\n\n".join([t.page_content for t in texts])
                    contents_for_model.append(context_text)
                    
                    progress_bar.progress(progress_percent + 70, text=f"[{i+1}/{total_files}] PDF解析完了。") 
                except Exception as pdf_error:
                    raise Exception(f"ファイル {uploaded_file.name} でPDF解析エラー: {pdf_error}") 

            elif file_extension in [".jpg", ".jpeg", ".png", ".mp3", ".wav"]:
                progress_bar.progress(progress_percent + 70, text=f"[{i+1}/{total_files}] ファイル処理完了。") 

            else: 
                st.warning(f"ファイル {uploaded_file.name} はサポートされていない形式です。スキップします。", icon="⚠️")
                continue 
       
       # --- ループ終了後 ---
       
       contents_for_model.extend(uploaded_gemini_files)

       progress_bar.progress(75, text="すべての資料のアップロード完了。AIに指示を出します...") 
       
       # --- Final Prompt Construction ---
       final_prompt_text = "" 
       
       # ★ file_names_str の定義を try の外に移動したため、ここでは削除 ★
       
       if selected_task == "問題を生成する":
            final_prompt_text = f"あなたは**以下の資料群**の専門家です。\n資料群: {file_names_str}\n【生成ルール】: 難易度: {difficulty} / 形式: {format_type} / 焦点: {professor_focus}\nこれらすべての資料の内容を統合し、このルールに従って問題と模範解答を計5問作成してください。"
       elif selected_task == "要約を作成する":
            length_map = {"短め": "150字程度", "普通": "300字程度", "長め": "500字程度"}
            final_prompt_text = f"以下の資料群（ファイル名: {file_names_str}）の内容を理解し、重要なポイントを箇条書きで要約してください。要約の長さは**厳密に{length_map.get(summary_length, '300字程度')}**にしてください。**必ず文章の最後まで完全な形で**出力してください。"
            if any(Path(f.name).suffix.lower() in [".mp3", ".wav"] for f in uploaded_files):
                 st.info("音声を文字起こししてから要約します...")

       elif selected_task == "リアクションペーパー作成":
            tone_instruction = ""
            keyword_instruction = ""
            if report_vocab == "簡潔な感想、学生向け":
                tone_instruction = "語彙は『〜と思う』『〜と感じた』『〜と考えている』といった主観的な表現を多く含め、AIとわかりづDらい文にしてください。"
            if report_keywords:
                keyword_instruction = f"キーワード「{report_keywords}」を**必ず**文中に何度も使用し、あなたの考察の中心に据えてください。"
            
            final_prompt_text = f"""以下の資料群（{file_names_str}）を基に、講義の核心とあなたの考察を含むリアクションペーパー（感想文）を作成してください。
【生成ルール】: 1. 文字数目安: {report_words}文字 / 2. 語彙/トーン: {report_vocab} / 3. 結論から書き始めてください。
【追加指示】:
{tone_instruction}
{keyword_instruction}
"""
            
       elif selected_task == "音声を文字起こしする":
            if not all(Path(f.name).suffix.lower() in [".mp3", ".wav"] for f in uploaded_files):
                st.warning("文字起こしは音声ファイル（MP3, WAV）のみ対応しています。音声以外のファイルは無視されます。", icon="⚠️")
            final_prompt_text = "以下の音声ファイル群の内容を正確に文字起こししてください。ファイルごとに内容を分けて出力してください。話者分離は不要です。"
       
       else: 
            st.error("未定義のタスクが定義されました。")
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

       except Exception as ai_e:
            error_message = f"AI生成エラーが発生しました: {ai_e}"
            st.error(error_message)
            generated_text = error_message 

    # --- Catch All Processing Errors (Outer Except Block) ---
    except Exception as e:
        error_message = f"処理中にエラーが発生しました: {e}"
        st.error(error_message)
        generated_text = error_message

    # --- Save Result to Session State ---
    st.session_state['generated_content'] = generated_text

    # --- 履歴保存処理の呼び出し ---
    # 連結したファイル名を渡す (file_names_str は try の外で定義済み)
    save_history_entry(
        generated_text, 
        ai_success, 
        selected_task, 
        difficulty, 
        format_type, 
        professor_focus, 
        summary_length, 
        file_names_str, # 連結ファイル名文字列
        st.session_state.selected_course, # 講義フォルダ
        report_words=report_words,
        report_vocab=report_vocab,
        report_keywords=report_keywords
    )
    # -----------------------------------------

    # --- Final Cleanup Block ---
    
    # 処理が完了したら、アップロードリストをクリアする
    st.session_state.uploaded_file_list = []

    # Geminiにアップロードしたファイルを削除
    if uploaded_gemini_files:
        for f in uploaded_gemini_files:
            try: genai.delete_file(f.name)
            except Exception as cleanup_error: st.warning(f"Geminiファイル削除中にエラー: {cleanup_error}", icon="⚠️")
    
    # 一時ファイルを削除
    if temp_file_paths:
        for p in temp_file_paths:
            if os.path.exists(p):
                os.remove(p)

    # --- Progress Bar Update ---
    if ai_success:
        progress_bar.progress(100, text="処理完了！")
    else:
        progress_bar.empty()
    
    # 処理完了後、UI（特にファイルリスト）を更新するためにRerun
    st.rerun()


# --- 11. DISPLAY AI GENERATED RESULT ---
if st.session_state['generated_content']:
    st.header("--- AI生成結果 ---")
    st.markdown(st.session_state['generated_content'])
