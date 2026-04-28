# convert_best.py
import os, json, ctranslate2

base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(base_dir, "checkpoints")

# trainer_state.json에서 best checkpoint 읽기
state_path = os.path.join(checkpoints_dir, "trainer_state.json")
with open(state_path) as f:
    state = json.load(f)

best_ckpt = state.get("best_model_checkpoint", checkpoints_dir)
best_ckpt = os.path.abspath(best_ckpt)  # 절대경로 보장
output_dir = os.path.join(base_dir, "faster-whisper-finetuned")

print(f"변환 대상: {best_ckpt}")
print(f"출력 경로: {output_dir}")

converter = ctranslate2.converters.TransformersConverter(
    model_name_or_path=best_ckpt,
    copy_files=["tokenizer.json", "preprocessor_config.json", "tokenizer_config.json"],
)
converter.convert(output_dir, quantization="int8", force=True)
print("변환 완료!")