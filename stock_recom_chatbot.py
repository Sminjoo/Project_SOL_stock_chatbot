import streamlit as st
import requests
import random
import time
import urllib.parse
import mplfinance as mpf
import FinanceDataReader as fdr
import tiktoken
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ✅ 1. 한글 폰트 설정
def set_korean_font():
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if not os.path.exists(font_path):  
        os.system("apt-get update -qq")
        os.system("apt-get install fonts-nanum* -qq")
    
    fe = fm.FontEntry(fname=font_path, name="NanumGothic")
    fm.fontManager.ttflist.insert(0, fe)  
    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False})

set_korean_font()  # ✅ 한 번만 실행

# ✅ 2. 메인 실행 함수
def main():
    st.set_page_config(page_title="Stock Analysis Chatbot", page_icon=":chart_with_upwards_trend:")
    st.title("_기업 정보 분석 주식 추천 :red[QA Chat]_ :chart_with_upwards_trend:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 주가 시각화에 필요한 세션 상태 추가 (기본값: 1day)
    if "selected_period" not in st.session_state:
        st.session_state.selected_period = "1day"

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        company_name = st.text_input("분석할 기업명 (코스피 상장)")
        process = st.button("분석 시작")

    if process:
        if not openai_api_key or not company_name:
            st.info("OpenAI API 키와 기업명을 입력해주세요.")
            st.stop()

        news_data = crawl_news(company_name)
        if not news_data:
            st.warning("해당 기업의 최근 뉴스를 찾을 수 없습니다.")
            st.stop()

        text_chunks = get_text_chunks(news_data)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = create_chat_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

        st.subheader(f"📈 {company_name} 최근 주가 추이")

        # 반응형 UI 버튼 추가 (선택한 기간을 즉시 반영)
        selected_period = st.radio(
            "기간 선택",
            options=["1day", "week", "1month", "1year"],
            horizontal=True
        )

        #  선택된 기간에 따라 주가 차트 업데이트
        visualize_stock(company_name, selected_period)
        
        with st.chat_message("assistant"):
            st.markdown("📢 최근 기업 뉴스 목록:")
            for news in news_data:
                st.markdown(f"- **{news['title']}** ([링크]({news['link']}))")

    if query := st.chat_input("질문을 입력해주세요."):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                result = st.session_state.conversation({"question": query})
                response = result['answer']

                st.markdown(response)
                with st.expander("참고 뉴스 확인"):
                    for doc in result['source_documents']:
                        st.markdown(f"- [{doc.metadata['source']}]({doc.metadata['source']})")

#✅ 3. 주가 시각화 & 티커 조회 함수
def get_ticker(company):
    """
    FinanceDataReader를 통해 KRX 상장 기업 정보를 불러오고,
    입력한 기업명에 해당하는 티커 코드를 반환합니다.
    환경에 따라 컬럼명이 다를 수 있으므로 여러 경우를 처리합니다.
    """
    try:
        listing = fdr.StockListing('KRX')
        if listing.empty:
            listing = fdr.StockListing('KOSPI')
        if listing.empty:
            st.error("KRX 혹은 KOSPI 상장 기업 정보를 불러올 수 없습니다.")
            return None

        # 여러 가지 컬럼 조합에 대해 처리합니다.
        if "Code" in listing.columns and "Name" in listing.columns:
            name_col = "Name"
            ticker_col = "Code"
        elif "Symbol" in listing.columns and "Name" in listing.columns:
            name_col = "Name"
            ticker_col = "Symbol"
        elif "종목코드" in listing.columns and "기업명" in listing.columns:
            name_col = "기업명"
            ticker_col = "종목코드"
        else:
            st.error("상장 기업 정보의 컬럼명이 예상과 다릅니다: " + ", ".join(listing.columns))
            return None

        # 좌우 공백 제거 후 비교
        ticker_row = listing[listing[name_col].str.strip() == company.strip()]
        if ticker_row.empty:
            st.error(f"입력한 기업명 '{company}'에 해당하는 정보가 없습니다.\n예시: '삼성전자' 입력 시 티커 '005930'을 반환합니다.")
            return None
        else:
            ticker = ticker_row.iloc[0][ticker_col]
            st.write(f"✅ 가져온 티커 코드: {ticker}")  # 🔥 티커 값 확인용 로그 추가
            # 숫자 형식인 경우 6자리 문자열로 변환 (예: 5930 -> '005930')
            return str(ticker).zfill(6)
    except Exception as e:
        st.error(f"티커 변환 중 오류 발생: {e}")
        return None

def visualize_stock(company, period):
    ticker = get_ticker(company)
    if not ticker:
        st.error("해당 기업의 티커 코드를 찾을 수 없습니다.")
        return

    try:
        df = fdr.DataReader(ticker, '2024-01-01')
        st.write(f"✅ 가져온 데이터 샘플:\n", df.head())  # 🔥 데이터가 정상적으로 들어오는지 확인
    except Exception as e:
        st.error(f"주가 데이터를 불러오는 중 오류 발생: {e}")
        return

    # ✅ 올바른 기간 필터링 (기본값 1day 유지)
    if period == "1day":
        df = df.tail(30)
    elif period == "week":
        df = df.resample('W').last()
    elif period == "1month":
        df = df.resample('M').last()
    elif period == "1year":
        df = df.resample('Y').last()
    
    # ✅ 차트 업데이트
    fig, _ = mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f"{company}({ticker}) 주가 ({period})",
        volume=True,
        returnfig=True
    )
    st.pyplot(fig)


#✅ 4. 뉴스 크롤링 & 챗봇 관련 함수
def crawl_news(company):
    today = datetime.today()
    start_date = (today - timedelta(days=5)).strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    encoded_query = urllib.parse.quote(company)
    url = f"https://search.naver.com/search.naver?where=news&query={encoded_query}&nso=so:r,p:from{start_date}to{end_date}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.select("ul.list_news > li")

    data = []
    for article in articles[:10]:
        title = article.select_one("a.news_tit").text
        link = article.select_one("a.news_tit")['href']
        content = article.select_one("div.news_dsc").text if article.select_one("div.news_dsc") else ""
        data.append({"title": title, "link": link, "content": content})

    return data

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(news_data):
    # 뉴스 요약 없이 제목과 내용을 그대로 사용
    texts = [f"{item['title']}\n{item['content']}" for item in news_data]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.create_documents(texts)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)

def create_chat_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h, return_source_documents=True)



# ✅ 실행
if __name__ == '__main__':
    main()
