# HTTP API

`create_app(orchestrator, api_key=None)` returns a FastAPI app. Interactive docs are served at `/docs` (Swagger UI)
and `/redoc`. When `api_key` is set, every endpoint except `/health` requires the header `X-API-Key`.
The optional header `X-User` is recorded as the approver.

| Method | Path | Purpose | Responses |
|---|---|---|---|
| GET | `/health` | Liveness, registered agents and tools | 200 |
| POST | `/runs` | Plan and start a run. Body: `{"task": "..."}` | 201 run · 401 · 422 task could not be planned |
| GET | `/runs` | Runs, newest first | 200 |
| GET | `/runs/{id}` | A run with every step, its status, output, error and attempts | 200 · 404 |
| GET | `/runs/{id}/trace` | Ordered events: `run_started`, `tool_called`, `tool_error`, `step_succeeded`, `approval_requested`, `approval_decided`, `run_finished`, ... | 200 · 404 |
| GET | `/approvals?status=pending` | Approvals (`pending`, `approved`, `rejected`) | 200 |
| POST | `/approvals/{id}/approve` | Approve; the run resumes immediately. Optional body `{"reason": "..."}` | 200 run · 404 · 409 already decided |
| POST | `/approvals/{id}/reject` | Reject (body `{"reason": "..."}` required); dependent steps are skipped | 200 run · 404 · 409 |
| GET | `/metrics` | Counters and per-tool latency (calls, avg, p95, max in ms) | 200 |

## Run and step states

```text
run:   running -> waiting_approval -> running -> succeeded | failed | rejected
step:  pending -> [waiting_approval ->] running -> succeeded | failed      (rejected / skipped are terminal)
```

A step is **skipped** when any step it depends on failed, was rejected or was skipped.

## Example session

```bash
uvicorn examples.procurement:app
curl -s -X POST localhost:8000/runs -H 'Content-Type: application/json' -d '{"task": "restock SKU-1 x 120"}'
# -> "status": "waiting_approval"  (120 x 398.00 = 47,760 is above the 20,000 approval threshold)
curl -s localhost:8000/approvals
curl -s -X POST localhost:8000/approvals/<approval-id>/approve -H 'X-User: priya'
# -> "status": "succeeded", the order step's output contains the purchase order
```
