from langchain_ollama import ChatOllama
import os
import threading

# 모든 Ollama 호출을 직렬화하는 프로세스 전역 락.
# 메인 파이프라인(analyze_write)과 백그라운드 요약(_summarize_to_ready)이
# 같은 모델 슬롯을 동시에 요청하면, 단일 슬롯 Ollama에서는 스래싱/직렬화로
# 지연이 폭증해 후처리 워커가 멈춘 것처럼 보인다(전사·요약 동시 정지).
# 모든 .invoke() 호출부를 이 락으로 감싸 동시 호출을 원천 차단한다.
llm_lock = threading.Lock()

_model = os.getenv("LLM_MODEL", "driving-mentor")
_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
_num_ctx = int(os.getenv("LLM_NUM_CTX", "2048"))
_num_predict = int(os.getenv("LLM_NUM_PREDICT", "120"))

# 요약 전용 모델/토큰 한도. 메인 모델(driving-mentor)은 SYSTEM 프롬프트가 구조화
# JSON 출력을 강제하므로 요약 용도로는 부적합하다. 일반 생성 모델을 받았다면
# LLM_SUMMARY_MODEL로 분리 지정하고, 아니면 메인 모델을 재사용하되 호출부에서
# SystemMessage로 멘토 SYSTEM을 덮어써 평문 요약을 받는다(listen_list.py 참조).
_summary_model = os.getenv("LLM_SUMMARY_MODEL", _model)
_summary_num_predict = int(os.getenv("LLM_SUMMARY_NUM_PREDICT", "256"))

# 요청 타임아웃(초). GPU 메모리 부족으로 Ollama가 모델을 스래싱하면 단일 호출이
# 무한정 늘어져 후처리 워커가 영구히 막힌다("적재만" 동결). httpx 타임아웃을 주입해
# 이 시간을 넘기면 예외로 빠지게 하면, _downstream_loop의 핸들러가 해당 발화를 건너뛰고
# 다음 발화를 계속 처리한다(동결 대신 우아한 저하).
_request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "60"))
_client_kwargs = {"timeout": _request_timeout}

llm_structured = ChatOllama(
    model         = _model,
    temperature   = _temperature,
    format        = "json",
    num_ctx       = _num_ctx,
    num_predict   = _num_predict,
    client_kwargs = _client_kwargs,
)
llm = ChatOllama(
    model         = _model,
    temperature   = 0.7,
    num_ctx       = _num_ctx,
    num_predict   = _num_predict,
    client_kwargs = _client_kwargs,
)
llm_summary = ChatOllama(
    model         = _summary_model,
    temperature   = 0.0,
    num_ctx       = _num_ctx,
    num_predict   = _summary_num_predict,
    client_kwargs = _client_kwargs,
)
print(
    f"[LLM] provider=Ollama  model={_model}  summary_model={_summary_model}  "
    f"timeout={_request_timeout:.0f}s"
)
