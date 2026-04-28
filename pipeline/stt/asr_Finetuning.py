"""
Whisper 한국어 파인튜닝 파이프라인 / Whisper Korean ASR Finetuning Pipeline

데이터 흐름 / Data flow:
  HuggingFace Dataset (Common Voice ko / KsponSpeech 로컬)
      → KoreanDatasetLoader  (리샘플링 + 특징 추출 + 토크나이징)
      → WhisperDataCollator  (Seq2Seq 패딩 + -100 레이블 마스킹)
      → WhisperFinetuner     (Seq2SeqTrainer + WER 평가)
      → HF checkpoint
      → ModelConverter       (ct2-transformers-converter → faster-whisper 형식)

사용 예시 / Usage:
    python asr_Finetuning.py --model base --epochs 3 --batch-size 8 --convert
    python asr_Finetuning.py --model small --dataset ksponspeech --dataset-path /data/kspon
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset
from dotenv import load_dotenv
from evaluate import load as load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

load_dotenv()                               # .env (공통 설정, git 커밋 O)
load_dotenv(".env.local", override=True)    # .env.local (민감 정보, git 커밋 X)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    """환경변수 값을 반환. 없으면 default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """환경변수를 int로 반환. 없거나 변환 실패 시 default."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    """환경변수를 float로 반환. 없거나 변환 실패 시 default."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """환경변수를 bool로 반환. 'true'/'1'/'yes' → True, 나머지 → False."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class FinetuningConfig:
    """
    Whisper 파인튜닝 하이퍼파라미터 및 경로 설정.
    Finetuning hyperparameters and path configuration for Whisper.

    기존 .env의 ASR_MODEL, ASR_LANGUAGE 값을 재사용합니다.
    Reuses ASR_MODEL and ASR_LANGUAGE from the existing .env file.
    """

    # 모델
    model_name: str = field(
        default_factory=lambda: f"openai/whisper-{_env('ASR_MODEL', 'base')}"
    )
    language: str = field(default_factory=lambda: _env("ASR_LANGUAGE", "ko"))
    task: str = "transcribe"

    # 데이터셋
    dataset_name: str = field(
        default_factory=lambda: _env(
            "FT_DATASET_NAME", "mozilla-foundation/common_voice_17_0"
        )
    )
    dataset_config: str = field(
        default_factory=lambda: _env("FT_DATASET_CONFIG", "ko")
    )
    dataset_split_train: str = field(
        default_factory=lambda: _env("FT_DATASET_SPLIT_TRAIN", "train+validation")
    )
    dataset_split_test: str = field(
        default_factory=lambda: _env("FT_DATASET_SPLIT_TEST", "test")
    )
    audio_column: str = field(
        default_factory=lambda: _env("FT_AUDIO_COLUMN", "audio")
    )
    text_column: str = field(
        default_factory=lambda: _env("FT_TEXT_COLUMN", "sentence")
    )
    max_label_length: int = field(
        default_factory=lambda: _env_int("FT_MAX_LABEL_LENGTH", 448)
    )

    # 학습
    output_dir: str = field(
        default_factory=lambda: _env("FT_OUTPUT_DIR", "./checkpoints")
    )
    num_train_epochs: int = field(
        default_factory=lambda: _env_int("FT_EPOCHS", 3)
    )
    per_device_train_batch_size: int = field(
        default_factory=lambda: _env_int("FT_TRAIN_BATCH_SIZE", 8)
    )
    per_device_eval_batch_size: int = field(
        default_factory=lambda: _env_int("FT_EVAL_BATCH_SIZE", 8)
    )
    gradient_accumulation_steps: int = field(
        default_factory=lambda: _env_int("FT_GRAD_ACCUM_STEPS", 2)
    )
    learning_rate: float = field(
        default_factory=lambda: _env_float("FT_LEARNING_RATE", 1e-5)
    )
    warmup_steps: int = field(
        default_factory=lambda: _env_int("FT_WARMUP_STEPS", 500)
    )
    max_steps: int = field(
        default_factory=lambda: _env_int("FT_MAX_STEPS", -1)
    )
    fp16: bool = field(
        default_factory=lambda: torch.cuda.is_available()
    )
    predict_with_generate: bool = True
    generation_max_length: int = field(
        default_factory=lambda: _env_int("FT_MAX_LABEL_LENGTH", 448)
    )
    save_steps: int = field(
        default_factory=lambda: _env_int("FT_SAVE_STEPS", 1000)
    )
    eval_steps: int = field(
        default_factory=lambda: _env_int("FT_EVAL_STEPS", 1000)
    )
    logging_steps: int = field(
        default_factory=lambda: _env_int("FT_LOGGING_STEPS", 25)
    )
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False         # WER은 낮을수록 좋음
    save_total_limit: int = field(
        default_factory=lambda: _env_int("FT_SAVE_TOTAL_LIMIT", 3)
    )
    push_to_hub: bool = field(
        default_factory=lambda: _env_bool("FT_PUSH_TO_HUB", False)
    )

    # 변환 (CTranslate2)
    converted_output_dir: str = field(
        default_factory=lambda: _env("FT_CONVERTED_OUTPUT_DIR", "./faster-whisper-finetuned")
    )
    quantization: str = field(
        default_factory=lambda: _env("FT_QUANTIZATION", "int8")
    )

    @property
    def hf_model_size(self) -> str:
        """openai/whisper-{size} 에서 size 부분만 반환."""
        return self.model_name.split("-")[-1]


# ---------------------------------------------------------------------------
# 데이터 콜레이터
# ---------------------------------------------------------------------------

class WhisperDataCollator:
    """
    Seq2Seq 학습을 위한 커스텀 데이터 콜레이터.
    Custom data collator for Seq2Seq training.

    - input_features: feature_extractor.pad() 로 배치 내 최대 길이 패딩
    - labels: tokenizer.pad() 후 pad_token_id → -100 치환
              (손실 계산 시 패딩 위치 무시 / cross-entropy ignores -100)
    """

    def __init__(self, processor: WhisperProcessor) -> None:
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Union[List[int], np.ndarray]]]
    ) -> Dict[str, torch.Tensor]:
        # input_features와 labels를 분리하여 각각 패딩
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # pad_token_id → -100: 손실 계산에서 패딩 위치 제외
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # BOS 토큰이 앞에 추가되어 있으면 제거 (Seq2SeqTrainer가 내부적으로 처리)
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# 데이터셋 로더
# ---------------------------------------------------------------------------

class KoreanDatasetLoader:
    """
    한국어 음성 데이터셋 로더 및 전처리기.
    Korean speech dataset loader and preprocessor.

    지원 데이터셋:
    - mozilla-foundation/common_voice_11_0 (ko)  — 기본 fallback
    - AIHub / KsponSpeech (로컬 경로, config.dataset_name에 경로 지정 시)
    """

    SAMPLE_RATE: int = 16_000

    def __init__(self, config: FinetuningConfig) -> None:
        self.config = config
        self.processor = WhisperProcessor.from_pretrained(
            config.model_name,
            language="korean",
            task="transcribe",
        )

    def load(self) -> DatasetDict:
        """
        데이터셋 로드 → 전처리 → DatasetDict 반환.
        Load dataset → preprocess → return DatasetDict.
        """
        logger.info(
            f"[KoreanDatasetLoader] 데이터셋 로드: {self.config.dataset_name}"
        )
        dataset = self._load_raw()

        # 오디오 자동 리샘플링 (16kHz)
        dataset = dataset.cast_column(
            self.config.audio_column, Audio(sampling_rate=self.SAMPLE_RATE)
        )

        # Windows에서 multiprocessing 사용 시 deadlock 위험 → num_proc=1
        num_proc = 1 if sys.platform == "win32" else os.cpu_count()

        logger.info("[KoreanDatasetLoader] 특징 추출 및 토크나이징 중...")
        dataset = dataset.map(
            self._prepare_dataset,
            remove_columns=dataset.column_names["train"],
            num_proc=num_proc,
        )

        # 너무 긴 레이블 필터링
        before = {k: len(v) for k, v in dataset.items()}
        dataset = dataset.filter(
            self._filter_long_labels, num_proc=num_proc
        )
        for split, before_count in before.items():
            after_count = len(dataset[split])
            if before_count != after_count:
                logger.warning(
                    f"[KoreanDatasetLoader] '{split}' split: "
                    f"{before_count - after_count}개 샘플이 "
                    f"max_label_length({self.config.max_label_length}) 초과로 제거됨"
                )

        logger.info(
            f"[KoreanDatasetLoader] 완료 — "
            f"train: {len(dataset['train'])}개, "
            f"test: {len(dataset['test'])}개"
        )
        return dataset

    def _load_raw(self) -> DatasetDict:
        """HuggingFace Hub 또는 로컬 디렉터리에서 원시 데이터셋 로드."""
        is_local = os.path.isdir(self.config.dataset_name)

        if is_local:
            # 로컬 KsponSpeech 등: audiofolder 형식 가정
            raw = load_dataset(
                "audiofolder",
                data_dir=self.config.dataset_name,
            )
        else:
            raw = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split={
                    "train": self.config.dataset_split_train,
                    "test": self.config.dataset_split_test,
                },
            )

        # Common Voice에 있는 불필요한 컬럼 제거 (메모리 절약)
        _drop_cols = [
            "accent", "age", "client_id", "down_votes", "gender",
            "locale", "path", "segment", "up_votes", "variant",
        ]
        for split in raw:
            existing = [c for c in _drop_cols if c in raw[split].column_names]
            if existing:
                raw[split] = raw[split].remove_columns(existing)

        return raw

    def _prepare_dataset(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 샘플 전처리 (dataset.map()에 전달).
        Single-sample preprocessing passed to dataset.map().

        - 오디오 배열 → log-mel spectrogram (80, 3000) — input_features
        - 텍스트 → 토큰 ID 리스트 — labels
        """
        audio = batch[self.config.audio_column]

        batch["input_features"] = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        batch["labels"] = self.processor.tokenizer(
            batch[self.config.text_column]
        ).input_ids

        return batch

    def _filter_long_labels(self, batch: Dict[str, Any]) -> bool:
        """max_label_length 초과 샘플 제거."""
        return len(batch["labels"]) <= self.config.max_label_length


# ---------------------------------------------------------------------------
# 파인튜너
# ---------------------------------------------------------------------------

class WhisperFinetuner:
    """
    Whisper 파인튜닝 메인 클래스.
    Main class for Whisper finetuning using HuggingFace Seq2SeqTrainer.
    """

    def __init__(self, config: FinetuningConfig) -> None:
        self.config = config
        self.processor: Optional[WhisperProcessor] = None
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.wer_metric = load_metric("wer")

    def setup(self) -> None:
        """
        모델 및 프로세서 초기화.
        Initialize model and processor.
        """
        logger.info(f"[WhisperFinetuner] 모델 로드: {self.config.model_name}")
        self.processor = WhisperProcessor.from_pretrained(
            self.config.model_name,
            language="korean",
            task="transcribe",
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name
        )

        # 한국어 forced decoder IDs 설정 (언어 고정: 자동 감지 비활성화)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="korean", task="transcribe"
        )
        self.model.config.forced_decoder_ids = forced_decoder_ids
        self.model.config.suppress_tokens = []

        # generation_config도 동기화 (predict_with_generate 사용 시 필요)
        self.model.generation_config.language = "korean"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = forced_decoder_ids
        self.model.generation_config.suppress_tokens = []

        logger.info("[WhisperFinetuner] 모델 초기화 완료")

    def train(self, dataset: DatasetDict) -> str:
        """
        학습 실행 후 최적 체크포인트 경로 반환.
        Run training and return the best checkpoint path.

        Args:
            dataset: {'train': Dataset, 'test': Dataset}

        Returns:
            최적 HF 체크포인트 디렉터리 경로
        """
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "setup()을 먼저 호출하세요. Call setup() before train()."
            )

        training_args = self._build_training_args()
        data_collator = WhisperDataCollator(self.processor)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            processing_class=self.processor.feature_extractor,  # 저장 시 함께 보존
        )

        logger.info("[WhisperFinetuner] 학습 시작...")
        trainer.train()

        best_ckpt = trainer.state.best_model_checkpoint or self.config.output_dir
        logger.info(f"[WhisperFinetuner] 학습 완료 — 최적 체크포인트: {best_ckpt}")

        # 프로세서를 체크포인트와 함께 저장 (재로드 편의)
        self.processor.save_pretrained(best_ckpt)
        return best_ckpt

    def _build_training_args(self) -> Seq2SeqTrainingArguments:
        """Seq2SeqTrainingArguments 생성."""
        return Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            fp16=self.config.fp16,
            predict_with_generate=self.config.predict_with_generate,
            generation_max_length=self.config.generation_max_length,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            save_total_limit=self.config.save_total_limit,
            push_to_hub=self.config.push_to_hub,
            eval_strategy="steps",
            save_strategy="steps",
            report_to="none",       # wandb / tensorboard 사용 시 변경
        )

    def _compute_metrics(self, pred) -> Dict[str, float]:
        """
        WER 계산 콜백. WER computation callback for Seq2SeqTrainer.

        pred.label_ids의 -100 → pad_token_id 로 복원 후 디코딩하여 WER 계산.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100 → pad_token_id (배치 디코딩을 위한 복원)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = self.processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = 100 * self.wer_metric.compute(
            predictions=pred_str, references=label_str
        )
        return {"wer": wer}


# ---------------------------------------------------------------------------
# 모델 변환기 (HF → faster-whisper CTranslate2)
# ---------------------------------------------------------------------------

class ModelConverter:
    """
    HuggingFace Whisper 체크포인트 → faster-whisper CTranslate2 형식 변환기.
    Converts HF Whisper checkpoint to faster-whisper CTranslate2 format.

    변환 명령 / Conversion command:
        ct2-transformers-converter --model <hf_dir>
            --output_dir <output_dir> --quantization int8 --force

    변환 후 모델은 PipelineConfig(model="<output_dir>") 로 추론에 사용 가능.
    """

    def __init__(self, config: FinetuningConfig) -> None:
        self.config = config

    def convert(
        self,
        hf_checkpoint_dir: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        HF 체크포인트를 CTranslate2 형식으로 변환.
        Convert HF checkpoint to CTranslate2 format.

        Args:
            hf_checkpoint_dir: HuggingFace 형식 체크포인트 경로
            output_dir: 변환 결과 저장 경로 (None이면 FinetuningConfig 기본값 사용)

        Returns:
            변환된 faster-whisper 모델 디렉터리 경로

        Raises:
            FileNotFoundError: hf_checkpoint_dir가 존재하지 않을 때
            RuntimeError: ct2-transformers-converter 실행 실패 시
        """
        if not os.path.isdir(hf_checkpoint_dir):
            raise FileNotFoundError(
                f"체크포인트 디렉터리가 없습니다: {hf_checkpoint_dir}"
            )
        self._check_converter_available()

        target_dir = output_dir or self.config.converted_output_dir
        cmd = [
            "ct2-transformers-converter",
            "--model", hf_checkpoint_dir,
            "--output_dir", target_dir,
            "--quantization", self.config.quantization,
            "--force",
        ]

        logger.info(f"[ModelConverter] 변환 명령 실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"ct2-transformers-converter 실패:\n{result.stderr}"
            )

        logger.info(f"[ModelConverter] 변환 완료 → {target_dir}")
        return target_dir

    @staticmethod
    def _check_converter_available() -> None:
        """ct2-transformers-converter CLI 존재 여부 확인."""
        if shutil.which("ct2-transformers-converter") is None:
            raise RuntimeError(
                "ct2-transformers-converter를 찾을 수 없습니다.\n"
                "pip install ctranslate2 를 실행하세요."
            )


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Whisper 한국어 ASR 파인튜닝 파이프라인 / "
            "Korean ASR Finetuning Pipeline"
        )
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper 모델 크기 (tiny/base/small/medium/large-v3)",
    )
    parser.add_argument(
        "--dataset", default="common_voice",
        choices=["common_voice", "ksponspeech"],
        help="학습 데이터셋 선택",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="KsponSpeech 등 로컬 데이터셋 경로 (--dataset ksponspeech 시 필수)",
    )
    parser.add_argument(
        "--output-dir", default="./checkpoints",
        help="HF 체크포인트 저장 디렉터리",
    )
    parser.add_argument(
        "--converted-output-dir", default="./faster-whisper-finetuned",
        help="CTranslate2 변환 결과 디렉터리",
    )
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="디바이스당 배치 크기 (GPU OOM 시 2–4로 줄이세요)",
    )
    parser.add_argument(
        "--convert", action="store_true",
        help="학습 완료 후 ct2-transformers-converter 자동 실행",
    )
    args = parser.parse_args()

    # FinetuningConfig 구성
    config = FinetuningConfig(
        model_name=f"openai/whisper-{args.model}",
        output_dir=args.output_dir,
        converted_output_dir=args.converted_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
    )

    if args.dataset == "ksponspeech":
        if args.dataset_path is None:
            parser.error("--dataset ksponspeech 는 --dataset-path 가 필요합니다.")
        config.dataset_name = args.dataset_path
        config.dataset_config = ""

    # ── 파이프라인 실행 ──────────────────────────────────────────────────────

    # 1. 데이터셋 로드 및 전처리
    loader = KoreanDatasetLoader(config)
    dataset = loader.load()

    # 2. 파인튜닝
    finetuner = WhisperFinetuner(config)
    finetuner.setup()
    best_ckpt = finetuner.train(dataset)

    # 3. (선택) CTranslate2 변환
    if args.convert:
        converter = ModelConverter(config)
        fw_dir = converter.convert(best_ckpt)
        logger.info(
            f"[Main] 변환 완료. faster-whisper 모델 경로: {fw_dir}\n"
            f"  → PipelineConfig(model='{fw_dir}') 로 추론 가능"
        )
    else:
        logger.info(
            f"[Main] 학습 완료. HF 체크포인트: {best_ckpt}\n"
            f"  → --convert 플래그를 추가하면 자동 변환됩니다.\n"
            f"  → 수동 변환: ct2-transformers-converter "
            f"--model {best_ckpt} "
            f"--output_dir {config.converted_output_dir} "
            f"--quantization {config.quantization} --force"
        )