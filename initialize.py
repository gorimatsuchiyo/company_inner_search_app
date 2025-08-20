"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# --- SQLite 3.35+ shim for Streamlit Cloud ---
try:
    import pysqlite3  # provided by pysqlite3-binary
    import sys
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite"] = pysqlite3
except Exception:
    pass
# ---------------------------------------------
from langchain_community.vectorstores import Chroma
import constants as ct

############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

# 必須定数の存在チェック
required_constants = [
    "LOG_DIR_PATH", "LOGGER_NAME", "LOG_FILE", "CHUNK_SIZE", "CHUNK_OVERLAP",
    "RETRIEVER_K", "RAG_TOP_FOLDER_PATH", "WEB_URL_LOAD_TARGETS", "SUPPORTED_EXTENSIONS"
]
for const in required_constants:
    if not hasattr(ct, const):
        raise AttributeError(f"Missing required constant '{const}' in constants.py")

############################################################
# ユーティリティ
############################################################
def _get_secret(key: str, default=None):
    """
    Streamlit Secrets -> 環境変数 -> default の順で値を取得
    """
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        # ローカルで st.secrets が未設定でも落ちないように
        pass
    return os.getenv(key, default)

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()

def initialize_logger():
    """
    ログ出力の設定（書き込み不可の場合は /tmp/app_logs にフォールバック）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # 既に設定済みなら再設定しない
    if logger.hasHandlers():
        return

    # 1) まず constants のパスを試す
    log_dir = ct.LOG_DIR_PATH
    log_path = os.path.join(log_dir, ct.LOG_FILE)

    def _try_setup(target_dir, target_file):
        os.makedirs(target_dir, exist_ok=True)
        handler = TimedRotatingFileHandler(
            target_file, when="D", encoding="utf8"
        )
        # セッションIDをフォーマットに入れておく
        session_id = st.session_state.get("session_id", "N/A")
        formatter = logging.Formatter(
            f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={session_id}: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    try:
        _try_setup(log_dir, log_path)
    except Exception:
        # 2) 失敗したら /tmp/app_logs に切替
        fallback_dir = "/tmp/app_logs"
        fallback_path = os.path.join(fallback_dir, ct.LOG_FILE)
        os.makedirs(fallback_dir, exist_ok=True)
        _try_setup(fallback_dir, fallback_path)

    # 右側ログにも出したいとき（Cloudのログパネル）
    logger.addHandler(logging.StreamHandler())

def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex

def initialize_retriever():
    if "retriever" in st.session_state:
        return

    docs_all = load_data_sources()
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # ★ Secrets/環境変数から OPENAI_API_KEY を確実に渡す
    openai_api_key = _get_secret("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY が見つかりません。Streamlit Cloud の Secrets か .env に設定してください。"
        )

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # CSVファイルは統合処理済みなので、そのまま使用
    # その他のファイルのみテキスト分割を適用
    final_docs = []

    for doc in docs_all:
        # CSVファイルの場合は分割せずにそのまま使用
        if doc.metadata.get("file_type") == "csv":
            final_docs.append(doc)
        else:
            # その他のファイルは従来通り分割
            text_splitter = CharacterTextSplitter(
                chunk_size=ct.CHUNK_SIZE,
                chunk_overlap=ct.CHUNK_OVERLAP,
                separator="\n"
            )

            page_number = doc.metadata.get("page", None)
            chunks = text_splitter.split_text(doc.page_content)

            for chunk in chunks:
                new_meta = doc.metadata.copy()
                if page_number is not None:
                    new_meta["page"] = page_number
                final_docs.append(
                    doc.__class__(page_content=chunk, metadata=new_meta)
                )

    # ベクターストアを一括登録
    db = Chroma.from_documents(final_docs, embedding=embeddings)
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_K})

def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []

def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all

def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


import pandas as pd
from langchain.schema import Document

def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得し小文字化
    file_extension = os.path.splitext(path)[1].lower()

    # 拡張子がドット付きで定義されているか確認
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # CSVファイルの場合は特別処理
        if file_extension == ".csv":
            try:
                # CSVファイルを読み込み
                df = pd.read_csv(path, encoding='utf-8')

                # ファイル名から内容を推測してドキュメントを構築
                file_name = os.path.basename(path)

                if "社員名簿" in file_name or "従業員" in file_name:
                    # 社員名簿の場合の特別処理
                    content = create_employee_document(df, file_name)
                else:
                    # その他のCSVファイルの場合の汎用処理
                    content = create_generic_csv_document(df, file_name)

                # 統合されたドキュメントを作成
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": path,
                        "page": 1,
                        "file_type": "csv",
                        "total_rows": len(df)
                    }
                )
                docs_all.append(doc)

            except Exception as e:
                logger = logging.getLogger(ct.LOGGER_NAME)
                logger.error(f"Error loading CSV file '{path}': {e}")
        else:
            # CSV以外のファイルは従来通りの処理
            loader_func = ct.SUPPORTED_EXTENSIONS[file_extension]
            if callable(loader_func):
                try:
                    loader = loader_func(path)
                    docs = loader.load()

                    # PDFの場合、ページ番号を metadata に追記
                    if file_extension == ".pdf":
                        for i, doc in enumerate(docs):
                            doc.metadata["page"] = i + 1
                    elif file_extension == ".txt":
                        for doc in docs:
                            doc.metadata["page"] = 1

                    docs_all.extend(docs)
                except Exception as e:
                    logger = logging.getLogger(ct.LOGGER_NAME)
                    logger.error(f"Error loading file '{path}': {e}")
            else:
                logger = logging.getLogger(ct.LOGGER_NAME)
                logger.error(f"Loader for extension '{file_extension}' is not callable for file '{path}'")


def create_employee_document(df, file_name):
    """
    社員名簿CSVから検索に適したドキュメントを作成
    """
    content_parts = []

    # ファイルの概要情報（検索キーワードを強化）
    content_parts.append(f"【{file_name} - 社員名簿・従業員一覧・人事データベース】")
    content_parts.append(f"全社員数: {len(df)}名の従業員情報")
    content_parts.append("")

    # 部署別の詳細情報（人事部の情報を特に強調）
    if '従業員区分' in df.columns:
        dept_groups = df.groupby('従業員区分')
        content_parts.append("【部署別従業員詳細情報】")

        for dept_name, dept_df in dept_groups:
            content_parts.append(f"\n■ {dept_name}部門 ({len(dept_df)}名)")
            content_parts.append(f"{dept_name}に所属している従業員一覧:")

            for idx, (_, row) in enumerate(dept_df.iterrows(), 1):
                employee_details = []
                for col in dept_df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip() and col != '従業員区分':
                        employee_details.append(f"{col}:{row[col]}")

                if employee_details:
                    content_parts.append(f"  {idx}. {', '.join(employee_details)}")

        content_parts.append("")

    # 人事部専用の検索最適化セクション
    hr_employees = df[df['従業員区分'] == '人事部'] if '従業員区分' in df.columns else pd.DataFrame()
    if not hr_employees.empty:
        content_parts.append("【人事部所属従業員の完全一覧】")
        content_parts.append(f"人事部には{len(hr_employees)}名の従業員が所属しています。")
        content_parts.append("人事部メンバー詳細:")

        for idx, (_, row) in enumerate(hr_employees.iterrows(), 1):
            hr_details = []
            for col in hr_employees.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    hr_details.append(f"{col}:{row[col]}")

            if hr_details:
                content_parts.append(f"人事部員{idx}: {', '.join(hr_details)}")

        content_parts.append("")

    # 全従業員の統合リスト
    content_parts.append("【全従業員統合データ】")
    content_parts.append("社内の全従業員情報:")

    for index, row in df.iterrows():
        employee_info = []
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                employee_info.append(f"{col}:{row[col]}")

        if employee_info:
            content_parts.append(f"社員{index + 1}: {', '.join(employee_info)}")

    # 検索用キーワードを大幅に強化
    content_parts.append("\n【検索最適化キーワード】")
    keywords = [
        "従業員情報", "社員名簿", "人事部", "社員一覧", "従業員一覧",
        "社員データ", "人事データ", "社員情報", "従業員データベース",
        "社員リスト", "従業員リスト", "人事情報", "社員台帳"
    ]

    # 部署名もキーワードに追加
    if '従業員区分' in df.columns:
        unique_depts = df['従業員区分'].dropna().unique()
        for dept in unique_depts:
            keywords.extend([f"{dept}", f"{dept}部", f"{dept}所属", f"{dept}の従業員"])

    content_parts.append(f"関連キーワード: {', '.join(keywords)}")

    return "\n".join(content_parts)


def create_generic_csv_document(df, file_name):
    """
    一般的なCSVファイルから検索に適したドキュメントを作成
    """
    content_parts = []

    # ファイルの概要情報
    content_parts.append(f"【{file_name}】")
    content_parts.append(f"データ件数: {len(df)}件")
    content_parts.append(f"項目数: {len(df.columns)}項目")
    content_parts.append("")

    # 列名の情報
    content_parts.append("【データ項目】")
    content_parts.append(", ".join(df.columns.tolist()))
    content_parts.append("")

    # データの内容
    content_parts.append("【データ内容】")
    for index, row in df.iterrows():
        row_info = []
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                row_info.append(f"{col}: {row[col]}")

        if row_info:
            content_parts.append(f"{index + 1}. " + ", ".join(row_info))

    return "\n".join(content_parts)

def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    # OSがWindows以外の場合はそのまま返す
    return s
