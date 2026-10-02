import os
import base64

from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

ALLOWED_TYPES = [
    "종이",
    "유리",
    "비닐",
    "플라스틱",
    "캔",
    "라벨이 붙어있는 플라스틱",
    "일반쓰레기",
    "판별불가"
]


@app.route("/")
def home():
    return "Recycling AI Server is running!"


@app.route("/analyze", methods=["POST"])
def analyze():

    # App Inventor에서 이미지가 제대로 전달되었는지 확인
    if "image" not in request.files:
        return jsonify({
            "error": "이미지가 없습니다."
        }), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    # 이미지를 Base64 형태로 변환
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # GPT에게 이미지 분석 요청
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """
이 사진에 있는 물체를 분리수거 기준에 따라
다음 8가지 중 가장 적절한 하나로 분류해라.

1. 종이
2. 유리
3. 비닐
4. 플라스틱
5. 캔
6. 라벨이 붙어있는 플라스틱
7. 일반쓰레기
8. 판별불가

판단 기준:

- 깨끗한 종이류는 '종이'
- 유리병이나 유리 용기는 '유리'
- 비닐봉지나 비닐 포장재는 '비닐'
- 라벨이 제거된 플라스틱 용기나 페트병은 '플라스틱'
- 캔이나 금속 음료캔은 '캔'
- 플라스틱이나 페트병에 라벨이 붙어 있으면
  '라벨이 붙어있는 플라스틱'
- 휴지, 물티슈, 칫솔, 고무 제품 등
  위의 재활용 분류에 해당하지 않는 일반적인 폐기물은
  '일반쓰레기'
- 사진이 너무 흐리거나 물체가 가려져 있거나
  어떤 종류인지 판단하기 어려우면 '판별불가'

특히 라벨이 붙어있는 페트병은
'플라스틱'이 아니라 '라벨이 붙어있는 플라스틱'으로 분류해라.

반드시 위 8가지 중 하나만 출력해라.
설명이나 다른 문장은 출력하지 마라.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    result = response.output_text.strip()

    # 허용된 결과가 아니면 판별불가 처리
    if result not in ALLOWED_TYPES:
        result = "판별불가"

    return jsonify({
        "result": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
