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
import pandas as pd

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

        # ✅ 반응형 UI 버튼 추가 (선택한 기간을 즉시 반영)
        selected_period = st.radio(
            "기간 선택",
            options=["1day", "week", "1month", "1year"],
            horizontal=True
        )

        # ✅ 주가 데이터를 가져오고 시각화
        with st.spinner(f"📊 {company_name} ({selected_period}) 데이터 불러오는 중..."):
            ticker = get_ticker(company_name)
            if not ticker:
                st.error("해당 기업의 티커 코드를 찾을 수 없습니다.")
                st.stop()

            df = None
            try:
                if selected_period in ["1day", "week"]:
                    df = get_intraday_data_bs(ticker)  # ✅ Requests 기반 크롤링 적용
                else:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=30 if selected_period == "1month" else 365)).strftime('%Y-%m-%d')
                    df = fdr.DataReader(ticker, start_date, end_date)

                if df is None or df.empty:
                    st.warning(f"📉 {company_name} ({ticker}) - 해당 기간({selected_period})의 거래 데이터가 없습니다.")
                else:
                    visualize_stock(df, company_name, selected_period)

            except Exception as e:
                st.error(f"주가 데이터를 불러오는 중 오류 발생: {e}")

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
    """
    try:
        listing = fdr.StockListing('KRX')
        if listing.empty:
            listing = fdr.StockListing('KOSPI')
        
        if listing.empty:
            st.error("상장 기업 정보를 불러올 수 없습니다.")
            return None

        # 컬럼명 처리 (KRX 데이터 컬럼명 기준)
        for name_col, ticker_col in [("Name", "Code"), ("Name", "Symbol"), ("기업명", "종목코드")]:
            if name_col in listing.columns and ticker_col in listing.columns:
                ticker_row = listing[listing[name_col].str.strip() == company.strip()]
                if not ticker_row.empty:
                    ticker = str(ticker_row.iloc[0][ticker_col]).zfill(6)
                    st.write(f"✅ 가져온 티커 코드: {ticker}")
                    return ticker

        st.error(f"'{company}'에 해당하는 티커 정보를 찾을 수 없습니다. \n예: '삼성전자' 입력 시 '005930' 반환")
        return None

    except Exception as e:
        st.error(f"티커 조회 중 오류 발생: {e}")
        return None

# ✅ 1. 네이버 금융 시간별 시세 크롤링 함수
def get_intraday_data_bs(ticker):
    """
    네이버 금융 iframe에서 시간별 체결가 데이터를 가져와 DataFrame으로 반환
    :param ticker: 종목코드 (예: '035720' - 카카오)
    :return: DataFrame (Datetime, Close, Volume)
    """
    # 📌 현재 시간 기반으로 'thistime' 값 설정
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 📌 iframe URL 구성
    base_url = f"https://finance.naver.com/item/sise_time.naver?code={ticker}&thistime={now}&page="
    headers = {"User-Agent": "Mozilla/5.0"}

    data = []
    page = 1

    while True:
        url = base_url + str(page)
        res = requests.get(url, headers=headers)
        time.sleep(1)  # 서버 과부하 방지

        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table.type2 tr")

        # ✅ 데이터가 없거나 마지막 페이지면 종료
        if not rows or "체결시각" in rows[0].text:
            break

        page_data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue  # 데이터가 부족하면 무시

            try:
                time_str = cols[0].text.strip()  # HH:MM 형식의 시간
                close_price = int(cols[1].text.replace(",", ""))  # 체결가
                volume = int(cols[5].text.replace(",", ""))  # 거래량

                page_data.append([time_str, close_price, volume])
            except ValueError:
                continue
        
        # ✅ 시계열 정렬을 위해 page_data 순서를 뒤집음
        data.extend(reversed(page_data))
        page += 1  # 다음 페이지로 이동

    # ✅ DataFrame 생성 및 정리
    if not data:
        print("❌ 크롤링된 데이터 없음.")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["Time", "Close", "Volume"])
    df["Date"] = datetime.today().strftime("%Y-%m-%d")  # 날짜 추가
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])  # 시간 합치기
    df.set_index("Datetime", inplace=True)
    df = df[["Close", "Volume"]]  # 필요한 열만 남기기

    print("✅ 크롤링 완료, 데이터 샘플:")
    print(df.head())  # 가져온 데이터 샘플 출력

    return df

    
# ✅ 2. 주가 시각화 함수
def visualize_stock(df, company, period):
    """
    가져온 주가 데이터를 기반으로 시각화
    :param df: 주가 데이터 DataFrame
    :param company: 기업명
    :param period: 기간 (1day, week, 1month, 1year)
    """
    if df is None or df.empty:
        st.warning(f"📉 {company} - 해당 기간({period})의 거래 데이터가 없습니다.")
        return

    fig, _ = mpf.plot(df, type='line' if period in ["1day", "week"] else 'candle',
                       style='charles', title=f"{company} 주가 ({period})",
                       volume=True, returnfig=True)
    st.pyplot(fig)
    
#✅ 5. 뉴스 크롤링 & 챗봇 관련 함수
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
