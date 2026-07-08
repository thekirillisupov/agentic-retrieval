# Запуск сервиса
AGENT_CONFIG=configs/ift_inference.yaml PYTHONPATH=. .venv/bin/python -m uvicorn agent.service:app --host 127.0.0.1 --port 8090

# Одиночный запрос
curl -s -X POST http://127.0.0.1:8090/retrieve    -H 'content-type: application/json'    -d '{
     "messages": [{"role": "user", "content": "Когда исправили ошибку с отсутствием документов по спецсчету капремонта?"}],
     "search_params": {},
     "prompt_version": "v2_search_only",
     "include_trajectory": true
   }' | python3 -m json.tool > data/processed/test_trajectory.json


# CKR валидация (search config подхватывается из configs/search_config.v.0.2.4.json)
PYTHONPATH=. bash scripts/run_ckr_eval.sh

# или один датасет
PYTHONPATH=. python3 -m eval_.run_ckr_eval \
  --eval data/ckr_eval/ckr_eval.jsonl \
  --url http://127.0.0.1:8090 \
  --prompt-version v2_search_only \
  --search-config configs/search_config.v.0.2.4.json \
  --out data/processed/ckr_eval/ckr_eval_results.json