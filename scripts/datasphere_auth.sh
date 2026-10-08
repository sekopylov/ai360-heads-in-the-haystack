#!/usr/bin/env bash
# Выдаёт рабочий IAM-токен (`t1....`) для `datasphere`, стараясь не запускать `yc`
# без нужды.
#
# Почему это вообще нужно:
#   * аккаунт федеративный, и `yc` хранит в ~/.config/yandex-cloud/credentials/default
#     СЕССИОННЫЙ токен федерации (`yctk...`) со сроком жизни ~12 часов. Пока он
#     свежий, `yc iam create-token` работает молча. Когда сессия истекла, `yc`
#     обязан пройти веб-аутентификацию: печатает "Authentication web site will be
#     opened", в терминале ждёт "Press 'enter' to continue...", затем открывает
#     console.yandex.cloud через cmd.exe / powershell.exe (WSL) — это и есть
#     вкладки в браузере.
#   * песочница агента не даёт `yc` записать обновлённую сессию в ~/.config,
#     поэтому там он уходит на веб-аутентификацию при КАЖДОМ вызове.
#   * `YC_NO_BROWSER=1` в CLI 1.40.0 открытие вкладки НЕ подавляет (проверено).
#   * токен из хранилища (`yctk...`) нельзя подставлять в YC_IAM_TOKEN: DataSphere
#     API отвечает на него 401, это не IAM-токен.
#
# Использование (именно source, иначе экспорт не попадёт в текущий shell):
#
#   source scripts/datasphere_auth.sh      # -> export YC_IAM_TOKEN
#   .venv-datasphere/bin/datasphere project get --id bt1u5v72b71eesdhp9k5
#
# Переопределяемое: DATASPHERE_TOKEN_CACHE, DATASPHERE_PROJECT_ID (пусто = не
# проверять токен), DATASPHERE_TOKEN_MAX_AGE_MIN, YC_BIN.  Внутренние переменные
# префиксованы `_rh_`, чтобы скрипт не затирал одноимённые переменные
# вызывающего шелла (раньше он делал `unset YC_BIN`).

_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_rh_cache="${DATASPHERE_TOKEN_CACHE:-$_repo_root/.cache/datasphere/iam-token}"
_rh_project="${DATASPHERE_PROJECT_ID-bt1u5v72b71eesdhp9k5}"
_rh_yc="${YC_BIN:-$(command -v yc || echo "$HOME/yandex-cloud/bin/yc")}"
_rh_max_age="${DATASPHERE_TOKEN_MAX_AGE_MIN:-660}"   # ~11 часов из 12

# Проверка токена дешёвым запросом к DataSphere API (пустой PROJECT_ID отключает
# саму проверку, но не требование непустого токена).
#
# Токен передаётся curl через конфиг на stdin (`-K -`), а не аргументом, поэтому
# он не виден в `ps` и /proc/*/cmdline.
_token_is_valid() {
  [ -n "$1" ] || return 1
  [ -z "$_rh_project" ] && return 0
  local _rh_code
  _rh_code=$(printf 'header = "Authorization: Bearer %s"\nurl = "%s"\n' \
      "$1" "https://datasphere.api.cloud.yandex.net/datasphere/v2/projects/$_rh_project" \
    | curl -s -o /dev/null -w '%{http_code}' --max-time 15 -K -)
  [ "$_rh_code" = "200" ]
}

_token=""

# 1. Собственный кэш: файл свежее MAX_AGE_MIN минут и токен ещё рабочий.
if [ -s "$_rh_cache" ] && [ -z "$(find "$_rh_cache" -mmin +"$_rh_max_age" 2>/dev/null)" ]; then
  _cached=$(cat "$_rh_cache")
  if _token_is_valid "$_cached"; then
    _token="$_cached"
  else
    echo "datasphere_auth: кэш не прошёл проверку, обновляю" >&2
  fi
fi

# 2. Иначе — `yc` (при истёкшей федеративной сессии он откроет вкладку).
if [ -z "$_token" ]; then
  echo "datasphere_auth: запрашиваю токен через yc (если сессия истекла — откроется вкладка консоли)" >&2
  _candidate=$(YC_NO_BROWSER=1 "$_rh_yc" iam create-token 2>/dev/null | tail -1)
  if _token_is_valid "$_candidate"; then
    _token="$_candidate"
  else
    echo "datasphere_auth: yc не вернул рабочий IAM-токен" >&2
  fi
fi

if [ -z "$_token" ]; then
  echo "datasphere_auth: не удалось получить токен (проверь 'yc init' и сеть)" >&2
  unset _cached _candidate _token _rh_cache _rh_project _rh_yc _rh_max_age _repo_root
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$(dirname "$_rh_cache")"
# umask 077 in a subshell: the secret must never exist with the default 0644
# permissions, and a sourced script must not change the caller's umask.
( umask 077; printf '%s' "$_token" > "$_rh_cache" )
chmod 600 "$_rh_cache"   # covers a cache file left by an earlier version
export YC_IAM_TOKEN="$_token"
echo "datasphere_auth: YC_IAM_TOKEN готов (${#_token} символов, кэш: $_rh_cache)" >&2
unset _cached _candidate _token _rh_cache _rh_project _rh_yc _rh_max_age _repo_root
