# cpal Code Review

Reviewed: `server.py`, `git_tools.py`, `test_tools.py`, packaging/docs. Line numbers are approximate (functions named so you can locate exactly). Findings grouped per your priority order; summary table at the end.

The overall architecture is sound — per-session asyncio locks, "only commit session state on success," pre-validation before API spend, and the defense-in-depth path checks are all good decisions. The serious problems are concentrated in **session history lifecycle** (pruning, truncation, thinking-block hygiene), where several paths permanently poison a session so every later call 400s.

---

## 1. Correctness bugs

### F1. CRITICAL — Naive message pruning corrupts conversation structure → permanently poisoned sessions
`server.py`, `_consult`, ~L760: `updated_messages = updated_messages[-MAX_SESSION_MESSAGES:]`

The agentic loop appends 2 messages per tool round (assistant `tool_use` + user `tool_result`). With the default cap of **1000 tool calls per query**, a single consult can easily exceed `MAX_SESSION_MESSAGES=200`, so pruning fires routinely. Slicing by count can make the retained history begin with:

- an **assistant** message → Anthropic API requires the first message to be `user` → 400; or
- a **user message containing `tool_result` blocks** whose `tool_use_id`s reference an assistant turn that was just sliced away → 400 (`tool_result` without matching `tool_use`); or
- an assistant message whose trailing `tool_use` lost its results → 400.

Worse, the broken head **persists**: every subsequent call appends at the tail and re-slices, so the session is dead until TTL expiry or restart — silent, permanent corruption, exactly what the project philosophy forbids.

**Fix:** Prune on *turn boundaries*. Walk backward from the end; only cut at a `user` message whose content is the original query (text blocks, not `tool_result` blocks), guaranteeing the head is a clean user turn. Also consider pruning by estimated tokens, not message count. Add a unit test that builds a synthetic tool-use history >200 messages and asserts the pruned head is a plain user message.

### F2. HIGH — `stop_reason == "max_tokens"` (and unknown stop reasons) can persist an orphaned `tool_use` → session poisoned
`server.py`, `run_agentic_loop`, ~L600–630 (max_tokens branch) and the unknown-stop-reason fallthrough.

If the response is truncated mid-tool-call (or stops for an unhandled reason like `refusal`/`pause_turn`) the content may end with a `tool_use` block. The code appends that assistant message to history and returns — **without ever supplying a `tool_result`**. The next request in that session will be rejected by the API ("tool_use ids found without tool_result blocks"), and since failures don't modify history, the session is stuck forever.

**Fix:** In every loop-exit path that isn't the `tool_use` branch, either (a) strip trailing `tool_use` blocks before persisting, or (b) append a synthetic user message with `tool_result` blocks like `"[tool not executed: response truncated]"` for each dangling `tool_use_id`. (b) is better — it preserves valid structure and tells the model what happened. Also explicitly handle `pause_turn` (continue the loop per API docs) rather than treating it as terminal.

### F3. HIGH — Thinking blocks stored in history are never re-filtered → toggling `extended_thinking=False` after a thinking turn 400s
`server.py`, `_filter_thinking_blocks` + `_consult`, ~L520/~L740.

`_filter_thinking_blocks` is applied only **at append time**, based on the *current* call's flag. Sequence:

1. Turn 1: `extended_thinking=True` (the default) → assistant messages with `thinking` blocks saved to session.
2. Turn 2: `extended_thinking=False` → request has no `thinking` param, but the history sent contains `thinking` blocks → API rejects (the function's own docstring states "When disabled, they must be stripped or the API rejects them").

The session isn't corrupted (failure doesn't persist), but every `extended_thinking=False` call on that session fails with a confusing API error.

**Fix:** Filter at *request build time*: in `_consult`, when thinking is disabled, map over `session["messages"]` stripping thinking/redacted_thinking from all assistant messages before sending (don't mutate stored history — the stored blocks are needed if the user re-enables thinking).

### F4. MEDIUM (low-confidence on exact API behavior) — Model migration keeps signed thinking blocks from another model
`server.py`, `get_session`, ~L460.

Switching `model` on an existing session just swaps the model string; history keeps thinking blocks **signed by the previous model**. Thinking blocks from *completed* turns are generally ignored by the API, so this may work in the common case — but any path where signature validation applies (and any manual↔adaptive boundary differences) turns the "migration" into a 400. At minimum it's untested, undocumented risk.

**Fix:** On migration, strip thinking/redacted_thinking blocks from all stored assistant messages (cheap, always safe — they're never needed across turns). Log the migration as you already do.

### F5. MEDIUM — Lock acquire/cleanup race can create two locks for one session
`server.py`, `get_session_lock` + `cleanup_old_sessions`, ~L320–360.

Race: coroutine A calls `get_session_lock("x")` and gets lock L but hasn't awaited `acquire()` yet → cleanup runs (triggered by another session's `get_session`), sees L unlocked, deletes it from `_session_locks` → coroutine B calls `get_session_lock("x")` and creates L2. A and B now hold *different* locks for the same session and run the agentic loop concurrently → interleaved history writes. The existing test (`TestCleanupPreservesHeldLocks`) covers only the *held* case, not the got-but-not-yet-acquired window.

**Fix:** Canonical-lock recheck:
```python
while True:
    lock = get_session_lock(sid)
    async with lock:
        if get_session_lock(sid) is lock:
            ...do work...
            break
```
or simply never delete locks (they're tiny; bound them by the same TTL sweep but only when you also hold them).

### F6. MEDIUM — Sessions can be reaped mid-consult → completed work silently discarded
`server.py`, `get_session` (sets `last_access` only at turn start) + `cleanup_old_sessions`.

`last_access` is bumped once, at the start of `_consult`. A long agentic run (1000 tool calls × multi-second API round-trips can exceed an hour) leaves `last_access` stale; if another request pushes `len(sessions) > 100`, cleanup deletes the session **while its lock is held and the loop is running**. The loop then writes `session["messages"] = ...` into an orphaned dict — the entire turn's history is silently lost and the next call starts a fresh session. Silent data loss, against project philosophy.

**Fix:** Skip cleanup for sessions whose lock is currently held (you already have the lock map; check `lock.locked()` before deleting the *session*, not just the lock), and/or refresh `last_access` after the loop completes.

### F7. MEDIUM — Module-level `asyncio.Lock()` binds to the first event loop; breaks across loops (including your own test suite)
`server.py`, `_models_lock = asyncio.Lock()` at import, ~L100.

Since 3.10, an asyncio.Lock binds to the loop that first acquires it and raises `RuntimeError("... bound to a different event loop")` if acquired from another loop. Production FastMCP uses one loop, fine — but pytest-asyncio (default function-scoped loops, and no `asyncio_mode`/loop-scope config exists in the repo) creates a fresh loop per test. `test_get_aliases_falls_back_when_no_api_key` and `test_fallback_not_cached` both go through the lock path in different loops; the second should blow up with RuntimeError. Either the suite is currently red/flaky, or it passes only by accidental loop reuse — both bad. Same hazard for `_session_locks` entries reused across loops in any embedding scenario.

**Fix:** Create the lock lazily inside the running loop (`global _models_lock; if _models_lock is None: _models_lock = asyncio.Lock()` guarded by the threading lock), or store it on the FastMCP lifespan/app state; add a conftest fixture that resets module locks between tests.

### F8. MEDIUM — Blocking I/O and 30s subprocesses run directly on the event loop
`server.py`, `run_agentic_loop` → `execute_tool` (sync); `git_tools.py` `_run_git` (`subprocess.run`, `GIT_TIMEOUT=30`).

Every tool execution (file reads up to 10MB, glob walks over 1000 files, git subprocesses up to 30s) blocks the asyncio loop. During that time the MCP server can't answer pings, progress callbacks, or other sessions — clients may declare the server dead.

**Fix:** `result = await asyncio.to_thread(execute_tool, block.name, block.input)`. The threading locks you already use make this safe.

### F9. LOW — Tool-call limit can be exceeded within one response batch
`run_agentic_loop`: the limit is checked only at the top of the `while`; a response containing N parallel `tool_use` blocks executes all of them even if it crosses `max_tool_calls`. Cosmetic given the limit's purpose, but the count reported can exceed the max. Fix: check inside the per-block loop, return synthetic "not executed: limit reached" `tool_result`s for the remainder.

### F10. LOW — `end_turn` with no text blocks returns empty string
`run_agentic_loop` end_turn branch has no `or f"Stopped: ..."` fallback (the unknown-stop branch does). An assistant turn that is, e.g., only `redacted_thinking` yields `""`, which an MCP client may render as nothing. Fix: fall back to a `"[no text content; stop_reason=end_turn]"` marker.

### F11. LOW — `time.time()` for TTL; comment/doc claims unconditional 1h expiry but cleanup only runs when `len(sessions) > 100`
Wall-clock jumps can mass-expire sessions (use `time.monotonic()`), and with ≤100 sessions TTL is never enforced, so README's "expire after 1 hour" is false and memory holds indefinitely. Fix: monotonic clock + periodic cleanup (e.g., also run on every Nth call or a background task).

---

## 2. Anthropic API misuse

### F12. HIGH (medium confidence) — Large manual thinking budgets produce `max_tokens` the request path can't deliver
`server.py`, `run_agentic_loop` ~L560: `kwargs["max_tokens"] = max(16384, thinking_budget + 8000)`; `_consult` allows budgets up to 100000; README *recommends* `thinking_budget=50000`.

Two problems for non-adaptive models (haiku, older opus/sonnet):
1. `budget=100000` → `max_tokens=108000`, which exceeds the max output of essentially every model tier → immediate 400. There is no per-model output-cap clamp.
2. Non-streaming `messages.create` with large `max_tokens` trips the SDK's long-request guard ("streaming is strongly recommended/required for operations that may take longer than 10 minutes") and/or plain network timeouts. With budgets ≥ ~50K you'll hit this on real requests.

**Fix:** Clamp `max_tokens` to a per-model output cap; for large budgets use `client.messages.stream(...)` and accumulate (this also enables incremental `ctx.report_progress`), or at minimum pass an explicit generous `timeout=` and document the failure mode.

### F13. MEDIUM — Thinking budget floor is 1000; the API minimum is 1024
`server.py`, `_consult` ~L700 (`thinking_budget < 1000`), echoed in `resource://config/limits` and both docstrings. Budgets 1000–1023 pass local validation and then 400 at the API with a confusing message. **Fix:** validate `>= 1024` everywhere (and in `count_tokens`/`create_batch`, which currently don't validate the budget at all).

### F14. MEDIUM — `list_batches(limit=N)` auto-paginates past the limit
`server.py`, `list_batches` ~L1010: in the anthropic SDK, `limit` is a *page size*; `async for` on the paginator transparently fetches subsequent pages. `list_batches(limit=20)` can return every batch from the last 29 days. **Fix:** `if len(batches) >= limit: break` inside the loop. (`_fetch_latest_models` uses the same pattern but there "iterate everything" is the intent, so it's fine — worth a comment.)

### F15. MEDIUM — `count_tokens` always counts with thinking enabled and can't mirror a non-thinking consult
`server.py`, `count_tokens` ~L880. The tool's stated purpose is to match the real request, but `consult_claude` may run with `extended_thinking=False` and `count_tokens` has no such parameter — and for non-adaptive models it unconditionally injects `thinking: enabled`. Counts can diverge from the actual request. **Fix:** add `extended_thinking: bool = True` and share a single `build_thinking_kwargs(model, extended_thinking, budget)` helper with `_consult`/`create_batch` (this also fixes the triplicated thinking-config logic).

### F16. MEDIUM — `create_batch` defaults `effort="max"` while `consult_claude` defaults `effort=None`
`server.py`, `create_batch` signature. "Max effort" on adaptive models maximizes output tokens — a silent cost amplifier on bulk jobs, the opposite of the batch API's cost-saving purpose, and inconsistent with the interactive default. **Fix:** default to `None` (model default) and let callers opt in; at minimum document loudly.

### F17. LOW — `effort` silently ignored on non-adaptive models
`run_agentic_loop` ~L575 / `create_batch`: `if effort is not None and _supports_adaptive_thinking(model)` drops the parameter without telling the caller. Philosophy says no silent fallbacks. **Fix:** return/append a notice (`"note: effort ignored for {model}"`) or include it in the result metadata.

### F18. LOW (low confidence) — Model discovery picks "newest by created_at" with substring matching; can select unintended variants
`server.py`, `_fetch_latest_models`. `f"claude-{tier}" in model.id` would also match hypothetical preview/specialty IDs (`claude-opus-4-8-preview`, `-instant`, region variants), and "newest created_at" would then route all `opus` traffic to it. Also, `KNOWN_TIERS` is a set, so tier-check order is nondeterministic (harmless today, fragile if IDs ever match two tiers). **Fix:** match with `_MODEL_VERSION_RE` (you already have it) and prefer the highest parsed version, using created_at only as tiebreaker; iterate a tuple, not a set.

### F19. LOW (low confidence) — 20MB inline media limit exceeds the API's per-image limit
`MAX_INLINE_MEDIA = 20MB`. Anthropic's image limit is ~5MB per image (PDFs are larger); a 19MB PNG passes local validation, gets base64-inflated ~33%, then 400s. **Fix:** per-type limits (image ≈ 5MB, PDF ≈ 20–30MB) and validate decoded size.

### F20. LOW — Discovery failure silently serves stale fallbacks; success is cached forever
`get_model_aliases`: a key-less or offline server returns `FALLBACK_ALIASES` with only a log-warning — `list_models()` output is indistinguishable from real discovery. Conversely, one successful discovery is cached for the process lifetime, so a weeks-long server never sees new releases. **Fix:** include `"source": "discovered" | "fallback"` and a fetch timestamp in `list_models`/`resource://models`; add a TTL (e.g., 24h) on the cache.

*Positive notes:* the adaptive-vs-manual floor logic (`_MODEL_VERSION_RE` with the 2-digit minor cap and lookahead), `display: "summarized"` gating, the beta-endpoint selection for `context_1m`, and required-thinking-block preservation during tool_use are all handled correctly and well-tested.

---

## 3. Security

### F21. MEDIUM-HIGH — Git tool sandbox is the *git toplevel*, which may be an ancestor of the cpal project root
`git_tools.py`, `_get_git_root` (runs `git rev-parse --show-toplevel` with `cwd=None` = process CWD) and `_validate_path(path, root)` validating against that root.

If cpal is started in `repo/subdir`, the advertised sandbox is `repo/subdir`, but the git tool validates paths against `repo/`: `git diff -- ../sibling/secrets.py`, `git log -- <anything in repo>`, and bare `git show HEAD` expose content **outside** the directory cpal claims to sandbox to (README: "All file access is sandboxed to the directory where cpal was started"). Also, using process CWD instead of `server._project_root` means any future `chdir` desynchronizes the two sandboxes.

**Fix:** Pass `server._project_root` as `cwd` to `_get_git_root`, and additionally require validated paths to be within `_project_root` (intersect the two roots). Document that `git show`/`log` without a path inherently reveal whole-repo history, or restrict diff/show output to paths under the project root via pathspec.

### F22. MEDIUM — Secrets in the sandbox are first-class exfiltration targets; `load_dotenv()` compounds it
`server.py` top-level `load_dotenv()`; no denylist in `read_file`/`search_project`/`build_content_blocks`.

The sandbox is "CWD at startup". If that's `$HOME` or a project with `.env`, the remote model can read `~/.config/cpal/api_key`, `.env`, `.git/config` credentials, etc., and the content flows through Anthropic and back to the MCP client. Separately, `load_dotenv()` silently adopts `ANTHROPIC_API_KEY` from whatever project's `.env` happens to be in CWD — surprising key/billing selection, and a silent fallback by philosophy.

**Fix:** (a) deny-list obvious secret paths (`.env*`, `*_key`, `id_rsa*`, `.git/config`, configurable); (b) log which key source is in use at startup (`--key-file` / env / `.env`); (c) consider dropping `load_dotenv()` or restricting it to `~/.config/cpal/.env`.

### F23. LOW — `search_project` glob iterates outside the sandbox before filtering
`server.py`, `execute_tool` search branch. With an absolute pattern (`/etc/*`) `glob.iglob` ignores `root_dir` and enumerates the real filesystem; `..`-relative patterns similarly walk outside. Per-file `_validate_path` correctly discards them, so **content** never leaks, but cpal still stats/walks arbitrary directories, and the `>{MAX_SEARCH_FILES}` error leaks a weak existence/count signal. **Fix:** reject absolute patterns and patterns containing `..` up front.

### F24. LOW — Symlink edge cases in `_validate_path`
`server.py`, `_validate_path` ~L250:
- An absolute symlink whose target *is exactly* `_project_root` is rejected (`str(root)` doesn't start with `str(root) + os.sep`) — false positive.
- The whole secondary symlink check is redundant with `resolve().relative_to(root)` (which already chases symlinks) — dead complexity that invites drift.
- Inherent TOCTOU between validation and the later `open()`/`read_text()` (link can be swapped). Acceptable for this threat model, but worth a comment.

**Fix:** delete the redundant block or replace the prefix test with `link_target.resolve().is_relative_to(_project_root)`.

### F25. LOW — `read_file` has no denylist for `.git` internals; `list_directory` unbounded
A directory with 200K entries produces a multi-MB tool result shipped to the API every subsequent loop iteration. Cap entries (e.g., 2000 + "(truncated)").

*Positive notes:* git ref validation (leading-`-` ban, blocklist, allowlist regex, `--` separators, arg-vector subprocess without shell), output caps, and timeouts are solid. Key handling never logs the key; `--key-file` keeps it out of argv/env.

---

## 4. Error handling

### F26. MEDIUM — `execute_git` is the only tool branch not exception-wrapped; malformed tool input kills the whole turn
`server.py`, `execute_tool`: `elif name == "git": return execute_git(input_data)`. If the model sends `max_count: "20"` (string), `min(max(1, "20"), 100)` raises `TypeError`, which propagates through `run_agentic_loop` into `_consult`'s blanket handler — the entire consult fails and all tool work is discarded, instead of returning an error string the model could correct (the pattern every other tool follows). **Fix:** wrap the git dispatch in try/except returning `f"Error: {e}"`, and/or coerce `max_count` with `int(...)` defensively in `execute_git`.

### F27. MEDIUM — Silent config fallbacks contradict the stated philosophy
- `CPAL_MAX_TOOL_CALLS` parse failure or `<1`: silently ignored (`except ValueError: pass`), ~L60. The operator believes a cap is in place; it isn't.
- `--system-prompt FILE` that fails to read: warning + continue (`_build_system_prompt`). A user who *explicitly* passed a prompt file (possibly policy/safety text) silently runs without it.
- Invalid config.toml: warning + empty config.
- `system_prompt` config value of wrong type: dropped with no warning at all.

**Fix:** For explicit operator inputs (CLI flag, env var), fail fast at startup (`sys.exit` with message). Keep warn-and-continue only for the optional config file, and add the missing warning for non-str `system_prompt`.

### F28. LOW-MEDIUM — `_consult`'s blanket `except Exception` converts cpal bugs into chat strings, without tracebacks
~L770: `logger.error(f"Error in session {session_id}: {e}")` then `return f"Error: {e}"`. A `KeyError` from a cpal bug becomes the string `"Error: 'model'"` shown to the calling LLM, and the log has no stack trace. **Fix:** `logger.exception(...)` (keeps the trace), and consider re-raising non-`anthropic.APIError` exceptions so FastMCP surfaces a proper tool error — that matches "crash > silent fallback".

### F29. LOW — Inconsistent error surfaces
- `get_client()` is called *outside* the try in `_consult` (~L725): a missing API key raises through FastMCP as a tool exception, while every other failure is a string/dict. Tests pass only because path validation happens first.
- `get_batch_results` catches only `anthropic.APIError`; every other batch tool catches `Exception`. Shape-mismatch errors (`entry.result.message` surprises) raise raw.
- `_consult` returns `"Error: ..."` strings; `count_tokens`/batch tools return `{"error": ...}` dicts.

**Fix:** pick one contract per tool family and apply uniformly.

### F30. LOW — File read errors inside `build_content_blocks` become context text instead of failures
A binary file in `file_paths` passes pre-validation (size/path only), then `read_text` fails and the block becomes `"Error reading 'x': ..."` *sent to Claude as context*. Visible but easy to miss; the caller paid for a consult missing the file they asked for. **Fix:** pre-validate decodability (read first KB and try UTF-8) and fail the call, consistent with the other pre-checks.

### F31. LOW — `"... (truncated)"` appended even when exactly MAX_SEARCH_MATCHES were found and nothing was cut
Cosmetic, but it tells the model results were dropped when they weren't.

---

## 5. Test gaps (what could break that the suite won't catch)

- **`run_agentic_loop` has zero coverage.** Every high-severity finding above (F1, F2, F3, F9, F10) lives there. A fake `AsyncAnthropic` returning scripted responses (tool_use → end_turn, tool_use → max_tokens, multi-tool_use parallel blocks) would catch all of them.
- **`git_tools.py` has zero tests** — no coverage of `_validate_ref` blocklist/regex (the security-critical code!), `_validate_path` root behavior, `max_count` clamping/type handling, output truncation, or the toplevel-vs-project-root issue (F21).
- **Pruning** (F1): no test constructs >200-message tool history and validates structural integrity of the pruned head.
- **Thinking-toggle and migration history hygiene** (F3/F4): nothing exercises multi-turn flag changes.
- **Concurrency**: no test of two concurrent `_consult`s on one session, nor the get-lock/cleanup race (F5); the existing lock-survival test covers only the held case.
- **Request construction**: nothing asserts the actual kwargs sent for `count_tokens`/`create_batch`/`run_agentic_loop` (thinking config, max_tokens bump, betas, effort) — the adaptive-floor unit tests check classification but not what's transmitted.
- **`list_batches` pagination** (F14) and batch param building are untested.
- **Weak assertions:** `test_search_project_found` accepts "No matches" as a pass; `test_read_file_nonexistent` passes for *any* error including a path-validation bug. Several tests can't fail meaningfully.
- **Suite infrastructure:** the module-level `asyncio.Lock` cross-loop issue (F7) likely makes the async tests order-dependent/flaky; there's no conftest resetting module globals (`_discovered_models`, `_project_root`, `sessions`, `_session_locks`) — tests mutate live module state and rely on `finally` cleanup. Tests also assume pytest runs from the repo root (first `_validate_path` call pins `_project_root` to CWD).
- `cleanup_old_sessions` is documented "must be called with `_sessions_lock` held" but all three tests call it bare — works under GIL, but the contract is unenforced (assert or take the lock internally).

---

## 6. Design / maintainability

- **F32 (MEDIUM):** `consult_claude` `timeout=600` vs `max_tool_calls=1000` default is incoherent — 1000 tool rounds cannot complete in 10 minutes; the timeout will fire, the session won't be updated, and all spent tokens are lost with no partial state. Either persist progress incrementally (commit history after each tool round under the session lock), lower the default cap, or raise/remove the timeout.
- **F33 (LOW-MEDIUM):** `create_batch` is annotated `READONLY` (`readOnlyHint=True`) but creates remote state and spends money; hosts may auto-approve read-only tools. Give batch creation (and arguably `consult_claude`) honest annotations.
- **F34 (LOW):** Thinking-config construction is triplicated (`run_agentic_loop`, `count_tokens`, `create_batch`) and the text/thinking extraction block is triplicated inside `run_agentic_loop`. Extract `build_thinking_kwargs()` and `extract_text_and_thinking(content)` — this is where the next thinking-related API change will be missed in one copy.
- **F35 (LOW):** Two unrelated `_validate_path` functions (`server.py` raises; `git_tools.py` returns `str | None`) — rename one (`_validate_git_path`).
- **F36 (LOW):** `search_project` has no per-file size cap and reads with `for line in f` — a multi-GB binary with no newlines materializes as one "line" in memory. Skip files > `MAX_FILE_SIZE` (cheap `stat`). Also note glob skips dotfiles by default (`include_hidden=False`), so searches silently miss `.github/`, `.env`, etc. — at least document.
- **F37 (LOW):** `DEFAULT_TOOL_CALLS` comment says "can be overridden per-call" but no per-call parameter exists on `consult_claude`; either add `max_tool_calls` to the tool or fix the comment. Tier names are duplicated across `KNOWN_TIERS`, `DEFAULT_TOOL_CALLS`, `FALLBACK_ALIASES`, `TIER_DESCRIPTIONS` — derive from one table.
- **F38 (LOW):** Memory growth is bounded by message *count* only; 200 messages can hold ~GBs (10MB file reads echoed in tool_results, re-sent every iteration). Consider byte-budget pruning and truncating tool_results stored in history.
- **F39 (LOW):** Docs drift: README comparison table says "Context window 200K" while advertising 1M beta; README claims sessions "expire after 1 hour" (only if >100 sessions, see F11); security section overstates the git sandbox (F21).

---

## Prioritized summary

| # | Sev | Location | Issue | Fix (short) |
|---|-----|----------|-------|-------------|
| F1 | **Critical** | `_consult` pruning | Count-based slice breaks message structure → session permanently 400s | Prune at turn boundaries |
| F2 | **High** | `run_agentic_loop` max_tokens/unknown-stop | Orphaned `tool_use` persisted → session poisoned | Strip/synthesize tool_results on truncation; handle `pause_turn` |
| F3 | **High** | `_filter_thinking_blocks` usage | History not re-filtered when thinking toggled off → 400 | Filter full history at request build |
| F12 | **High** (med conf) | `run_agentic_loop` max_tokens calc | Budgets ≥~50K exceed model output caps / non-streaming 10-min SDK guard | Clamp per model; use streaming |
| F21 | **Med-High** | `git_tools._get_git_root` | Git sandbox = repo toplevel + process CWD, escapes advertised project sandbox | Pin to `_project_root`; intersect roots |
| F5 | Medium | `get_session_lock`/cleanup | Lock get-vs-acquire race → duplicate locks, concurrent history writes | Canonical-lock recheck after acquire |
| F6 | Medium | `get_session`/cleanup | Session reaped mid-consult → turn silently lost | Don't reap held sessions; refresh `last_access` post-loop |
| F7 | Medium | module `_models_lock` | asyncio.Lock bound to first loop; breaks tests/embedding | Lazy per-loop creation; test conftest reset |
| F8 | Medium | `execute_tool` call site | Blocking I/O + 30s subprocess on event loop | `asyncio.to_thread` |
| F13 | Medium | `_consult`/limits | Budget floor 1000 vs API min 1024 | Validate ≥1024 everywhere |
| F14 | Medium | `list_batches` | SDK auto-pagination ignores `limit` | Break at `limit` |
| F15 | Medium | `count_tokens` | Can't mirror non-thinking requests; always thinking-enabled | Add `extended_thinking`; shared kwargs builder |
| F16 | Medium | `create_batch` | Default `effort="max"` = silent cost amplifier, inconsistent with consult | Default `None` |
| F26 | Medium | `execute_tool` git branch | Unwrapped exceptions kill whole turn on bad tool input | try/except → error string; coerce `max_count` |
| F27 | Medium | env/CLI config | Silent fallback on invalid `CPAL_MAX_TOOL_CALLS`, missing `--system-prompt` file | Fail fast on explicit operator input |
| F32 | Medium | `consult_claude` | 600s timeout vs 1000-call default; timeout discards all work | Incremental session commits or coherent limits |
| F4 | Medium (low conf) | `get_session` | Cross-model thinking blocks kept on migration | Strip thinking on migration |
| F22 | Low-Med | server top / tools | Secrets (`.env`, key file) readable in sandbox; `load_dotenv` CWD key adoption | Denylist; log key source |
| F28/F29 | Low-Med | `_consult`, batch tools | Bugs masked as chat strings, no tracebacks; inconsistent error contracts | `logger.exception`; unify; re-raise internal bugs |
| F33 | Low-Med | `create_batch` | `readOnlyHint=True` on a money-spending, state-creating tool | Honest annotations |
| F17–F20, F23–F25, F30, F31, F9–F11, F34–F39 | Low | various | effort silently dropped; discovery substring/staleness; media limit; glob outside-walk; symlink edge; dotfile search miss; doc drift; duplication | per finding |
| Tests | — | `tests/` | No coverage of agentic loop, git_tools, pruning, toggles, concurrency, request shapes; weak assertions; global-state bleed | Mock-client loop tests; git unit tests; conftest resets |

The four fixes I'd land before anything else: **F1, F2, F3** (all are "session permanently broken / request poisoned" bugs in the core loop and trivially reachable with default settings), then **F12** since the README actively recommends a `thinking_budget` that the current request path cannot service.