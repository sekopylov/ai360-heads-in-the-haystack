# Colab: Qwen3-8B no-thinking smoke

Функциональная проверка существующего detection-кода, не полный survey:
три кейса, длина1000, глубина50%, max64, FP16, Flash SDPA prefill,
multiset и attention mass,1152Q-голов. Адаптер/промпт Qwen не изменяются.
Colab CUDA torch сохраняется; отдельные зависимости в requirements.txt.

Сессия smoke: `qwen3-smoke`, A10040GB. CLI установлен локально через
`uv tool install --python 3.12 google-colab-cli`; OAuth `--auth oauth2`.
Drive mount требует отдельного подтверждения пользователя.

Из корня репозитория, после создания сессии и подключения Drive:

```bash
python3 updated/colab/prepare_bundle.py --output /tmp/retrieval-heads-colab-smoke.zip
colab --auth oauth2 upload -s qwen3-smoke /tmp/retrieval-heads-colab-smoke.zip /content/retrieval-heads-smoke.zip
colab --auth oauth2 exec -s qwen3-smoke -f updated/colab/launch_smoke.py
colab --auth oauth2 exec -s qwen3-smoke -f updated/colab/status_smoke.py
```

Launcher запускает worker в фоне: установка зависимостей, скачивание
Qwen/Qwen3-8B с зафиксированным HF commit, detection, aggregation всех трёх
случаев и проверка полного набора голов, конечности scores и отсутствия think.
Низкий ROUGE не выдаётся за поломку программы; ответы/score сохраняются.
Повторный запуск поверх существующих результатов запрещён.

Drive: `MyDrive/retrieval-heads-colab/`:

- `qwen3-smoke/`: JSON/контексты/aggregation, runtime, лог, validation.
- `qwen3-smoke-bootstrap.log`: установка/скачивание/все выводы worker.
- `qwen3-smoke-status.json`: текущий этап или ошибка.
- `qwen3-smoke-checkpoint.json`: repo/commit скачанных весов.

Веса находятся в `/content/models/Qwen3-8B`, код —
`/content/retrieval-heads-smoke/updated`; они временные и теряются при
удалении VM. Сохранённые на Drive файлы остаются, но работающий процесс
после удаления VM не продолжится сам: resume в этом smoke не реализован.
A100 расходует compute units, пока VM выделена. После скачивания и проверки
результатов сессию можно остановить (`colab stop -s qwen3-smoke`), что удалит
временный диск и веса. Не останавливать до сохранения нужных файлов.

## Проверенный запуск 11 октября 2026

Три кейса завершены успешно: score100 каждый, EOS36/28/33токена,
thinking нет, обе метрики содержат1152finite heads с ненулевыми scores.
Runtime: A100-SXM4-40GB, torch2.11.0+cu130, transformers5.19.0.
HF commit: `b968826d9c46dd6066d109eabc6255188de91218`.
Результаты также скачаны в `updated/colab-results/qwen3-smoke/`; повторная
локальная validation passed. После копирования сессия освобождается.
98 существующих и5 новых локальных unit tests прошли.
