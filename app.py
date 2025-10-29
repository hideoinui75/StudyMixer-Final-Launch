# app.py

# --- 1. IMPORTS ---
import streamlit as st
import os
import google.generativeai as genai
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

# ★ サイドバーのデフォルト幅を広げるCSS ★
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


# --- 3. HISTORY SAVE FUNCTION ---
# --- FUNCTION: HISTORY CALLBACK (履歴保存関数) ---
# 修正点: 引数を簡素化し、オプションを辞書(options)として受け取る
def save_history_entry(generated_text, ai_success, selected_task, options, uploaded_file_names, lecture_name):
    """AI処理が成功した場合にのみ履歴を保存する関数 (引数を簡素化)"""
    
    if not ai_success:
        st.sidebar.warning("AI処理が成功しなかったため、履歴は保存されませんでした。", icon="⚠️")
        return 

    try:
        # 2. 履歴エントリ作成
        file_list_str = ", ".join(uploaded_file_names)
        
        history_entry = {
            "file_name": file_list_str, 
            "task": selected_task,
            "options": options, # ★ 渡された辞書をそのまま保存
            "result": generated_text,
            "lecture_name": lecture_name,
        }

        # 3. 重複チェック (厳密なチェック)
        is_duplicate = False
        if st.session_state['analysis_history']:
            last_entry = st.session_state['analysis_history'][0]
            if (last_entry['file_name'] == history_entry['file_name'] and
                last_entry['task'] == history_entry['task'] and
                len(last_entry['result']) == len(history_entry['result'])):
                is_duplicate = True

        # 4. 重複でなければ追加
        if not is_duplicate:
            st.session_state['analysis_history'].insert(0, history_entry)
            st.session_state.just_saved_history = True # 成功フラグを立てる

            # 5. 履歴の最大件数を制限
            MAX_HISTORY = 10
            if len(st.session_state['analysis_history']) > MAX_HISTORY:
                st.session_state['analysis_history'] = st.session_state['analysis_history'][:MAX_HISTORY]
        
    except Exception as hist_e:
         st.sidebar.error(f"履歴保存中にエラーが発生しました: {hist_e}")

# --- 4. SESSION STATE INITIALIZATION ---
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
if 'chat_history' not in st.session_state: # ★★★ チャット履歴を追加 ★★★
    st.session_state.chat_history = [
        {"role": "model", "parts": [{"text": "チャットを開始するには、まず資料をアップロードし、サイドバーの実行ボタンを押して分析を実行してください。"}]}
    ]
if 'chat_context_title' not in st.session_state: # ★★★ コンテキストタイトルを追加 ★★★
    st.session_state.chat_context_title = "（資料未選択)"

# ★ 講義フォルダ機能 (永続化なし) ★
if 'course_folders' not in st.session_state:
    st.session_state.course_folders = ["未分類"] # ★ デフォルト値で初期化
if 'selected_course' not in st.session_state:
    st.session_state.selected_course = "未分類"

# ★ 複数ファイルアップロード（エラー回避用）リスト ★
if 'uploaded_file_list' not in st.session_state:
    st.session_state.uploaded_file_list = []

# --- 5. APP SETUP ---
st.title("💡 Study-Mixer - リアぺも問題も５秒で生成。")
st.markdown("---")

# --- 6. API KEY CONFIGURATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError: 
    st.error("エラー: .streamlit/secrets.toml に GEMINI_API_KEY が設定されていません。")
    st.stop()
except Exception as e:
    st.error(f"APIキーの設定中に予期せぬエラーが発生しました: {e}")
    st.stop()

# --- 7. UI CONTROLS (Sidebar) ---
with st.sidebar:
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

    # --- 生成ボタン ---
    button_label = f"{selected_task} ( {len(st.session_state.uploaded_file_list)}件の資料 )"
    generate_button = st.button(button_label, key="generate_button", use_container_width=True) 

    # --- 講義フォルダ管理 ---
    st.header("📚 講義フォルダ管理")
    new_course_name = st.text_input("新しい講義名を追加:", placeholder="例: 経済学特講", key="new_course_input")
    
    if st.button("講義フォルダを追加"):
        if new_course_name and new_course_name not in st.session_state.course_folders:
            st.session_state.course_folders.append(new_course_name)
            st.session_state.selected_course = new_course_name # 追加したものを選択
            # save_data() 呼び出し削除
            st.rerun() # UI更新のためリラン
    
    
    st.subheader("講義フォルダ")

    # ★ フォルダリストをボタンで描画 (選択と削除のUI) ★
    folder_to_delete = None # 削除が押されたフォルダを一時的に保持
    
    for folder in st.session_state.course_folders:
        cols = st.columns([5, 1]) # フォルダ名用の列 と 削除ボタン用の列
        
        # 現在選択中のフォルダかどうかでボタンのタイプを変える
        is_selected = (folder == st.session_state.selected_course)
        button_type = "primary" if is_selected else "secondary"
        
        # --- フォルダ選択ボタン ---
        if cols[0].button(folder, key=f"select_{folder}", use_container_width=True, type=button_type):
            st.session_state.selected_course = folder
            st.rerun() # 選択をUIに反映

        # --- 削除ボタン (「未分類」フォルダ以外に表示) ---
        if folder != "未分類":
            if cols[1].button("🗑️", key=f"delete_{folder}", help=f"「{folder}」を削除します"):
                folder_to_delete = folder # 削除対象としてマーク

    # ループの外で削除処理を実行
    if folder_to_delete:
        
        st.session_state.course_folders.remove(folder_to_delete)
        
        for entry in st.session_state['analysis_history']:
            if entry.get('course') == folder_to_delete:
                entry['course'] = "未分類"
        
        if st.session_state.selected_course == folder_to_delete:
            st.session_state.selected_course = "未分類"
        
        st.success(f"フォルダ「{folder_to_delete}」を削除しました。中の履歴は「未分類」に移動しました。")
        st.rerun() # UIを更新

    

    # --- 履歴表示 ---
    # History Selection (Display in Sidebar)
    st.markdown("---")
    
    # --- 履歴のフォルダビュー (講義別フィルター) ---
    st.header("📂 履歴リスト")
    
    # コールバック関数を定義 (フィルター設定用)
    def set_filter(lecture_name):
        """サイドバーボタンクリック時にフィルターを設定するコールバック関数"""
        st.session_state['current_lecture_filter'] = lecture_name
        st.rerun() 

    # --- 講義名リストとフィルターボタンの表示 ---
    if not st.session_state['analysis_history']:
        st.caption("まだ履歴はありません。")
    else:
        # 1. すべての講義名を抽出
        lecture_names = [entry.get('lecture_name', '（講義名なし）') for entry in st.session_state['analysis_history']]
        unique_lectures = sorted(list(set(lecture_names)))

        # 2. フィルタリング解除ボタン (すべての履歴)
        is_all_active = st.session_state.get('current_lecture_filter') is None
        st.button(
            "📚 すべての履歴を表示",
            on_click=set_filter,
            args=(None,),
            key="filter_all",
            use_container_width=True,
            type="primary" if is_all_active else "secondary"
        )
        
        st.markdown("---")
        st.caption("講義でフィルター:")

        # 3. 講義ごとのフォルダボタン
        for lecture in unique_lectures:
            is_active = st.session_state.get('current_lecture_filter') == lecture
            item_count = lecture_names.count(lecture)
            
            # 講義名ボタン
            st.button(
                f"📁 {lecture} ({item_count})",
                on_click=set_filter,
                args=(lecture,),
                key=f"filter_{lecture}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            )

        st.markdown("---") 
        st.subheader("📝 フィルターされた結果")

        # --- 4. フィルター処理 ---
        current_filter = st.session_state.get('current_lecture_filter', None)
        
        if current_filter is None:
            filtered_history = st.session_state['analysis_history']
            st.caption("現在: **すべての履歴**を表示中")
        else:
            filtered_history = [
                entry for entry in st.session_state['analysis_history'] 
                if entry.get('lecture_name') == current_filter
            ]
            st.caption(f"現在: 講義『**{current_filter}**』の履歴を {len(filtered_history)}件 表示中")


        # --- 5. 履歴の選択ボックス ---
        if not filtered_history:
             st.info("この講義の履歴はありません。", icon="ℹ️")
        else:
            history_titles = []
            for i, entry in enumerate(filtered_history):
                 # 難易度や形式、ファイル名を結合して表示名を作成
                display_str = f"[{entry['task']}] {entry['file_name'].split(',')[0][:15]}..." # 最初のファイル名で省略
                if entry['task'] == "問題を生成する" and entry['options']:
                    difficulty_str = entry['options'].get('難易度', '不明')
                    format_str = entry['options'].get('形式', '不明')
                    display_str = f"[問:{difficulty_str} / {format_str}] {entry['file_name'].split(',')[0][:10]}..."

                history_titles.append(f"{i+1}: {display_str}")

            options_with_placeholder = ["履歴を選択して表示..."] + history_titles

            selected_history_display = st.selectbox(
                "結果を選択:", 
                options=options_with_placeholder,
                index=0, 
                key="history_selectbox_display"
            )

            # --- 6. 履歴選択時の表示ロジック ---
            if selected_history_display != "履歴を選択して表示...":
                try:
                    # -1でインデックスを取得
                    selected_index_in_list = options_with_placeholder.index(selected_history_display) - 1 
                    selected_entry = filtered_history[selected_index_in_list]
                    
                    # メインエリアに表示する
                    st.session_state['generated_content'] = selected_entry['result']

                    # --- サイドバーでの詳細表示 ---
                    st.markdown("---")
                    st.subheader(f"選択履歴 {selected_index_in_list + 1} の詳細:")
                    st.caption(f"**タスク:** {selected_entry['task']}")
                    st.caption(f"**ファイル:** {selected_entry['file_name']}")
                    st.caption(f"**フォルダ:** {selected_entry.get('lecture_name', '未分類')}")
                    
                    if selected_entry['options']:
                        st.markdown("**実行オプション:**")
                        for k, v in selected_entry['options'].items():
                             st.write(f"- **{k}**: {v}")
                    
                    # メイン画面を更新
                    st.rerun() 

                except (ValueError, IndexError):
                     st.sidebar.error("履歴データの読み込み中にエラーが発生しました。", icon="🚨")

# --- 8. FILE UPLOADER (複数ファイル蓄積方式) ---
def reset_history_selection_on_upload():
    st.session_state.displayed_history_index = None

# ★「multiple=True」のエラー回避ロジック ★
newly_uploaded_file = st.file_uploader(
    f"資料を1つずつアップロード (保存先: **{st.session_state.selected_course}**)",
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"],
    key="file_uploader_single", # キー名を変更
    on_change=reset_history_selection_on_upload
    # multiple=True は使用しない
)

# アップロードされたファイルをリストに蓄積する
if newly_uploaded_file is not None:
    is_already_added = False
    for f in st.session_state.uploaded_file_list:
        if f.name == newly_uploaded_file.name and f.size == newly_uploaded_file.size:
            is_already_added = True
            break
            
    if not is_already_added:
        st.session_state.uploaded_file_list.append(newly_uploaded_file)
        st.rerun() # ファイル追加を即時反映

# --- アップロード済みファイルリストの表示とクリアボタン ---
if st.session_state.uploaded_file_list:
    st.subheader(f"📤 分析対象の資料リスト ({len(st.session_state.uploaded_file_list)}件)")
    
    cols = st.columns([4, 1])
    
    index_to_remove = None
    
    for i, f in enumerate(st.session_state.uploaded_file_list):
        cols[0].caption(f"- {f.name} ({f.size // 1024} KB)")
        if cols[1].button(f"削除", key=f"remove_{i}"):
            index_to_remove = i
            
    if index_to_remove is not None:
        st.session_state.uploaded_file_list.pop(index_to_remove)
        st.rerun() 

    if st.button("リストをすべてクリア", key="clear_all_uploads"):
        st.session_state.uploaded_file_list = []
        st.rerun()
        
st.markdown("---")

# --- 9. AI PROCESSING LOGIC ---
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
    uploaded_gemini_files = [] 
    temp_file_paths = []       
    contents_for_model = []    
    uploaded_file_names = []

    # ★★★ 講義名決定ロジックをここに移動 ★★★
    input_value = st.session_state.get('lecture_name_input', '')
    if not input_value and uploaded_files:
        lecture_name = uploaded_files[0].name
    elif not input_value:
        lecture_name = "（無題の分析）" 
    else:
        lecture_name = input_value 
    # -----------------------------------------------

    try: # Outer try for all processing steps
       total_files = len(uploaded_files)
        
       # --- ループで全ファイルを処理 ---
       for i, uploaded_file in enumerate(uploaded_files):
            file_extension = Path(uploaded_file.name).suffix.lower() 
            temp_file_path = f"temp_{i}_{uploaded_file.name}" 
            temp_file_paths.append(temp_file_path)

            progress_percent = int((i / total_files) * 50) 

            progress_bar.progress(progress_percent + 10, text=f"[{i+1}/{total_files}] ファイルを一時保存中...")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar.progress(progress_percent + 30, text=f"[{i+1}/{total_files}] {file_extension.upper()} ファイルをGeminiにアップロード中...")
            gemini_uploaded_file = genai.upload_file(path=temp_file_path, display_name=uploaded_file.name)
            uploaded_gemini_files.append(gemini_uploaded_file)
            
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
                tone_instruction = "語彙は『〜と思う』『〜と感じた』『〜と考えている』といった主観的な表現を多く含め、AIとわかりづらい文にしてください。"
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

# --- Save Result to Session State (Always happens) ---
    st.session_state['generated_content'] = generated_text

    # --- チャットコンテキストも同時に更新 (ここが新しい修正) ---
    if ai_success: # AIが成功した場合のみ
        st.session_state.chat_context = generated_text # 分析結果をチャットの知識ベースに設定
        st.session_state.chat_history = [] # 新しい分析結果でチャット履歴をリセット
        
        # ★★★ 修正箇所：uploaded_file_names が空でないかチェック ★★★
        if uploaded_file_names: # リストが空でないことを確認
            first_file_name = uploaded_file_names[0]
            display_file_name = first_file_name[:20] + "..." if len(first_file_name) > 20 else first_file_name
            st.session_state.chat_context_title = f"最新の分析結果 ({display_file_name})" 
            initial_message = f"「{display_file_name}」の分析完了！この資料について何か質問はありますか？"
        else:
            # リストが空だった場合の代替
            st.session_state.chat_context_title = "最新の分析結果" 
            initial_message = "分析完了！この資料について何か質問はありますか？"
        
        # ★★★ 修正箇所：成功時のAIからの最初の挨拶を、上記の if/else ブロックで定義した initial_message を使う ★★★
        st.session_state.chat_history.append({"role": "model", "parts": [{"text": initial_message}]})
        # ----------------------------------------------------

    else: # AIが失敗した場合
        st.session_state.chat_context = generated_text # エラーメッセージをコンテキストに設定
        st.session_state.chat_history = []
        st.session_state.chat_context_title = "分析エラー"
    # ------------------------------------------------
    
    # --- 履歴保存処理を関数呼び出しに置き換え (NameError解消の修正) ---
    # 履歴保存に必要な final_options をここで定義
    final_options = {}
    
    if selected_task == "問題を生成する":
        # Sidebarから取得した変数を使って辞書を作成
        final_options = {"難易度": difficulty, "形式": format_type, "焦点": professor_focus}
    elif selected_task == "要約を作成する":
        final_options = {"長さ": summary_length}
    elif selected_task == "リアクションペーパー作成":
        # report_keywords が Sidebar で定義されている前提
        final_options = {"文字数": report_words, "語彙/トーン": report_vocab, "強調キーワード": st.session_state.get('report_keywords', '')}

    # ★★★ この行が、重複なく履歴を保存する鍵です ★★★
    # lecture_name も AI処理ブロックの先頭で定義されている前提
    save_history_entry(generated_text, ai_success, selected_task, final_options, uploaded_file_names, lecture_name)
    # -----------------------------------------

    # --- Final Cleanup Block ---
    # This block executes sequentially after history saving
    
    # ... (この後にクリーンアップコードが続く) ...
    
    # --- 履歴保存処理を関数呼び出しに置き換え (NameError解消の修正) ---
    # 履歴保存に必要な final_options をここで定義
    final_options = {}
    
    if selected_task == "問題を生成する":
        # Sidebarから取得した変数を使って辞書を作成
        final_options = {"難易度": difficulty, "形式": format_type, "焦点": professor_focus}
    elif selected_task == "要約を作成する":
        final_options = {"長さ": summary_length}
    elif selected_task == "リアクションペーパー作成":
        # report_keywords が Sidebar で定義されている前提
        final_options = {"文字数": report_words, "語彙/トーン": report_vocab, "強調キーワード": st.session_state.get('report_keywords', '')}

    # ★★★ この行が、重複なく履歴を保存する鍵です ★★★
    # lecture_name も AI処理ブロックの先頭で定義されている前提
    save_history_entry(generated_text, ai_success, selected_task, final_options, uploaded_file_names, lecture_name)
    # -----------------------------------------

    # -----------------------------------------

    # --- Final Cleanup Block ---
    
    # 処理が完了したら、アップロードリストをクリアする
    st.session_state.uploaded_file_list = []

    if uploaded_gemini_files:
        for f in uploaded_gemini_files:
            try: genai.delete_file(f.name)
            except Exception as cleanup_error: st.warning(f"Geminiファイル削除中にエラー: {cleanup_error}", icon="⚠️")
    
    if temp_file_paths:
        for p in temp_file_paths:
            if os.path.exists(p):
                os.remove(p)

    if ai_success:
        progress_bar.progress(100, text="処理完了！")
    else:
        progress_bar.empty()
    
    st.rerun()


# --- 10. DISPLAY AI GENERATED RESULT ---
if st.session_state['generated_content']:
    st.header("--- AI生成結果 ---")
    st.markdown(st.session_state['generated_content'])

# --- 8.5 CHAT INTERFACE (Main Area) --- 
st.markdown("---")
st.subheader("💬 資料深掘りチャット") 
st.caption(f"コンテキスト: {st.session_state.chat_context_title}")

# ★★★ 修正箇所：チャット入力を条件分岐の前に移動 ★★★

# ユーザー入力のプレースホルダーを先に定義
placeholder_text = "この資料について質問を入力..."
disabled_state = False

if st.session_state.get('chat_context_title') == "（資料未選択）":
     if uploaded_files: # ファイルはあるが、まだ分析されていない
        st.info("サイドバーの実行ボタンを押して分析を完了すると、チャットが開始できます。", icon="💡")
        user_query = st.chat_input("分析を実行すると入力できます...", disabled=True)
     else: # ファイルもアップロードされていない
        st.info("チャットを開始するには、まず資料をアップロードし、分析を実行してください。", icon="💡")
        user_query = st.chat_input("資料をアップロードしてください...", disabled=True)
     
# ユーザー入力ウィジェット
user_query = st.chat_input(placeholder_text, disabled=disabled_state)

# 過去のチャット履歴を表示
for message in st.session_state.chat_history:
    role = message["role"] 

    if role == "model":
        avatar_icon = "🎓" # 脳（知識）のアイコン
    elif role == "user":
        avatar_icon = "🙂" # ユーザーアイコン
    else:
        continue 
    
    # ロールが user または model の場合のみ表示
    if role in ["user", "model"]: 
        with st.chat_message(role, avatar=avatar_icon): 
            st.markdown(message["parts"][0]["text"])


if user_query:
    
    # 1. ユーザーの質問をまず「表示用」の履歴に追加する
    st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_query}]})
    
    # 2. 以前のコンテキスト（分析結果）を取得
    context_content = st.session_state.get('chat_context', '（コンテキストなし）')
    context_title = st.session_state.get('chat_context_title', '不明な資料')
    
    # 3. AIがコンテキストに基づいた役割を果たすための「指示」を構築
    system_instruction = f"""
    あなたはユーザーの学習をサポートする専門家です。
    以下の[提供された資料]（タイトル: {context_title}）の内容に厳密に基づき、ユーザーの質問に回答してください。
    
    【重要】: 回答の際、「[分析コンテキスト]」や「[提供された資料]」という言葉は絶対に使わないでください。代わりに「この資料によると、」や「資料によれば、」といった自然な表現を使用してください。

    [提供された資料]: {context_content}
    """
    
    # 4. モデルの呼び出し (Gemini APIのチャット機能を使用)
    try:
        # 5. APIに送信するためのクリーンな会話履歴 (api_contents) を構築
        api_contents = []
        
        # 画面表示用の st.session_state.chat_history からAPI用のリストを作成
        for i, msg in enumerate(st.session_state.chat_history):
             # ロールを 'assistant' から 'model' に変換
             role_to_use = "model" if msg["role"] == "assistant" else msg["role"]
             
             # APIがサポートする 'user' と 'model' ロールのみを追加
             if role_to_use in ["user", "model"]:
                 
                 # ★★★ ここが最重要 ★★★
                 # もし、これが会話の「最初の」「ユーザー」のメッセージなら、システム指示を結合する
                 if i == len(st.session_state.chat_history) - 1 and role_to_use == "user":
                     original_prompt = msg["parts"][0]["text"]
                     # 最新の質問に「脳」(システム指示)を結合
                     integrated_prompt = f"{system_instruction}\n\n[ユーザーの最新の質問]: {original_prompt}"
                     api_contents.append({"role": "user", "parts": [{"text": integrated_prompt}]})
                 else:
                     # 過去の会話（AIの挨拶など）はそのまま追加
                     api_contents.append({"role": role_to_use, "parts": msg["parts"]})
        # --- ★★★ 会話履歴の構築ここまで ★★★ ---

        with st.spinner("AIが回答を作成中です..."):
             # system_instruction 引数は使わず、構築した contents のみを渡す
             response = genai.GenerativeModel('models/gemini-2.5-flash').generate_content(
                contents=api_contents # 修正された会話履歴
             )

        # 6. 応答の安全チェック
        is_blocked = (hasattr(response, 'candidates') and 
                      response.candidates and 
                      hasattr(response.candidates[0], 'finish_reason') and 
                      response.candidates[0].finish_reason.name == "SAFETY")

        if is_blocked:
             error_message = "AIの応答が安全性フィルタによりブロックされました。別の質問を試してください。"
             st.error(error_message, icon="🚫")
             ai_response_text = error_message
        elif hasattr(response, 'text') and response.text:
             ai_response_text = response.text
        else:
             error_message = "AIからの応答が空か、予期せぬ形式でした。時間をおいて再試行してください。"
             st.error(error_message, icon="❓")
             ai_response_text = error_message
    
    except Exception as e:
        ai_response_text = f"チャットエラー: {e}"
        # エラーも履歴として表示するために role は model を使用
        st.session_state.chat_history.append({"role": "model", "parts": [{"text": ai_response_text}]})
        st.rerun() # エラー発生時も再実行
    
    else: # tryブロックが成功した場合のみ、AIの回答を履歴に追加
        st.session_state.chat_history.append({"role": "model", "parts": [{"text": ai_response_text}]})
        st.rerun() # 正常終了時も再実行
