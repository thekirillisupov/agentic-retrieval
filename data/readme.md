cd /home/jovyan/isupov/agentic-retrieval
for f in \
  data/processed/rnd/grpo_train.parquet     data/processed/rnd/grpo_val.parquet \
  data/processed/musique/grpo_train.parquet data/processed/musique/grpo_val.parquet \
  data/processed/sbol/grpo_train.parquet    data/processed/sbol/grpo_val.parquet ; do
    python -m grpo.data_prep --update-parquet "$f" --prompt-version v2_search_only
done

for f in \
  data/processed/ckr/grpo_train.parquet data/processed/ckr/grpo_val.parquet ; do
    python -m grpo.data_prep --update-parquet "$f" --prompt-version v2
done

python -m grpo.build_union_data --sources musique sbol rnd ckr


Обучающая выборка и структура данных:

Сценарий - один источник PROMPT_V2_search_only
независимые пассажи (faq):
  sbol - формат диалога
  rnd - один запрос
  musique - один multi-hop запрос
связанные пассажи PROMPT_v2
  ckr - синт генерация 

TODO: фильтрация (grep) + поиск по оставшемуся  (потестить ручку)


Сценарий - несколько источников, например + web (нужно добавлять поле sources)
