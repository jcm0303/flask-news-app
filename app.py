from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# Flask 앱 초기화
app = Flask(__name__)

# KoBART 모델 로드
model_name = "gogamza/kobart-summarization"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 뉴스 본문 추출 함수
def extract_article_content(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL. Status code: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    # "자동완성 기능이 켜져 있습니다." 텍스트 제거
    content = content.replace("자동완성 기능이 켜져 있습니다.", "")

    return content

# 텍스트 전처리 함수
def preprocess_text(text):
    sentences = text.split(".")
    unique_sentences = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))

    return ". ".join(unique_sentences) + "."

# 요약 실행 함수
def summarize_text(text):
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer.encode(preprocessed_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, num_beams=4, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 라우트 설정
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    original_content = None
    summary = None

    if request.method == "POST":
        url = request.form["url"]
        try:
            # 뉴스 본문 추출
            article_content = extract_article_content(url)
            original_content = article_content

            # 요약 실행
            if article_content.strip():
                summary = summarize_text(article_content)
            else:
                error = "기사 본문이 비어 있습니다."

        except Exception as e:
            error = f"오류 발생: {e}"

    return render_template("index.html", error=error, original_content=original_content, summary=summary)

# Flask 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
