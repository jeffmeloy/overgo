# Hatchet-derived workflow contracts for Overgo

## Purpose

This record identifies the elements of Hatchet, an orchestration engine for
background tasks, agents and durable workflows, that improve Overgo's workflow
capabilities when re-expressed as contracts on Overgo's existing Go owners. It
states, for each element, the mechanism as observed in the Hatchet source, the
gap it closes in Overgo, the contract to adopt, the owner that adopts it, why
the adoption simplifies or optimizes the current design, and the acceptance
evidence that closes the corresponding plan row. It also states what is not
adopted and why. The executable sequence is `docs/plan.json` in the
`overgo_hatchet` lane.

The review covered the checkout at `C:\Users\jeffm\hatchet`, commit
`086a63f22`, released as v0.105.16 (2026-08-31) under the MIT license:
the documentation under `frontend/docs/content/docs/v1`, the worker and
durable-task protocol under `api-contracts/v1`, the scheduler under
`pkg/scheduling/v1`, the repository queries under `pkg/repository/sqlcv1`,
the engine controllers under `internal/services`, and the load-test and
agent-instruction material. No Hatchet source is copied, translated or
vendored. Every element below is a contract or a mechanism restated in
Overgo's own vocabulary and implemented by the owner that already holds the
neighbouring authority.

## Overgo's workflow owners today

The elements below extend the following owners rather than adding parallel
machinery. Names are the packages as they exist at master `62851924`.

| Owner | Authority today |
| --- | --- |
| `internal/plan` | The campaign document, `ReadyFrontier`, work leases with workspace claims, the locality scheduler, completion authority derived from Git, step-local verification batches. |
| `internal/gate` | The sole commit path: scope from the compiler graph, protection, claims, closure, census and clone ratchets, package evidence reuse, batch acceptance. |
| `internal/loop` | The harness-agnostic driver: plan, implement, verify, commit, with attempt and invocation budgets, closure over admitted proposals, and the recorded operator stop. |
| `internal/workflowruntime` | Typed workflow graphs with bounded ready sets, an `AttemptRecorder`, a `ReconcileCoalescer` for repeated signals, and an `AutomationRuntime` with schedules, webhooks, manual runs and recovery. |
| `internal/agentloop` | Agent sessions: mutation checkpoints, session recovery, attempt closure, delegation, stimulus reconciliation, session policy. |
| `internal/operation` | The common operation lifecycle: state, progress, metrics, workspace claims, re-entrant execution, completion. |
| `internal/capabilityruntime` | Resident model sessions under a `ModelSessionDirector` with `SessionLease`, exact capability placement, measured execution, peers. |
| `internal/modelswap` | One endpoint serving many models through an exclusive child supervisor and a routing proxy. |
| `internal/processcontrol` | Supervised process trees with cancellation, deadlines and terminal receipts. |
| `internal/runrecord` | Attempt records, run outcomes, lease outcomes, webhook deliveries, direction pivots. |
| `internal/overgodb` | Content-addressed claims, journals and rebuildable projections. |

The existing design already holds several of Hatchet's structural decisions.
Hatchet's scheduler owns every slot pool from one run-loop goroutine and
treats a stale pool as a whole; Overgo's sequential-control doctrine makes the
same choice. Hatchet separates transactional state from time-partitioned
read projections; Overgo's claims-versus-projections doctrine is the same
separation. Those parallels are recorded here so that the campaign does not
re-derive them.

## Elements to incorporate

Each element names the plan item that carries it.

### E1. Invocation-counted durable attempts with a keyed memo log

Plan item: `durable-attempt-log`.

Observed mechanism. Every attempt at running a durable task carries a
monotonically increasing invocation count. A side effect is memoised by key:
the worker asks the engine whether a memo exists for the key before executing
and receives the stored result when it does. Child runs and waits are entries
in a per-task event log addressed by branch and node identifiers. A resumed
task replays the log and only executes past its last entry. A replay whose
requests diverge from the log fails with a typed non-determinism error, and an
invocation superseded by a newer one receives an evict notice and cannot
complete.

Gap. Overgo restarts agent sessions, workflow graphs and automations through
three separate recovery paths: `agentloop` reconciles committed interaction
facts, `workflowruntime` recovers completed stages, and the automation runtime
resumes exact persisted plans. Each path re-derives what may run again. A tool
call or model invocation that completed before the interruption has no memo
identity, so a restart re-executes it or refuses to resume.

Contract. One durable-attempt contract in `runrecord`: an attempt identity is
the pair of the durable unit and its invocation count; a memo entry is keyed
by a caller-supplied key bound to the attempt; a wait or a child attempt is a
log entry with an ordinal; completion of a memo, wait or child is recorded
once and read by every later invocation. `workflowruntime` replays a graph
from that log; `agentloop` runs tool calls as memoised child attempts and
refuses a replay whose next request differs from the recorded entry with a
typed divergence failure; an invocation with a stale count is refused at the
first memo request.

Why. Three recovery mechanisms collapse into one log contract with one replay
rule. Restart cost drops from re-executing completed tool and model calls to
reading their recorded results. The agent loop gains the property Hatchet
states for durable tasks: the code between checkpoints must be deterministic
given the log, which is already Overgo's replay doctrine for interaction
records.

Acceptance. Host-only tests in `runrecord`, `workflowruntime` and `agentloop`
that record an attempt, interrupt it after a memoised step, resume under a new
invocation count, prove the memoised step is not re-executed, prove a divergent
replay and a stale invocation are refused, and prove tool calls replay as child
attempts.

### E2. Eviction policy for waiting work and device claim release

Plan item: `eviction-and-release`.

Observed mechanism. A durable task that waits on a sleep, an event or a child
is evicted from its worker slot under a declared policy: a TTL of uninterrupted
waiting, capacity-based eviction when the worker needs the slot for new work,
and a priority that orders eviction candidates. The task resumes by replay
when the awaited condition holds. Separately, a running task may release its
slot after a resource-intensive phase while continuing on the host.

Gap. On Overgo the scarce slot is the device. Which resident session yields
device memory when a new admission needs it is decided by prose rules and by
manual lane choreography, and a row that generates on the device and scores on
the host holds the device through the host phase. The device-memory retention
finding on the E4B ladder is a symptom of the missing release contract.

Contract. A `SessionLease` and a waiting operation declare an eviction policy:
a waiting TTL, whether capacity eviction is permitted, and a priority. The
`ModelSessionDirector` evicts by policy when an admission needs capacity and
records the eviction reason. A running operation may release its device claim
after the device phase; the release is a recorded state transition, the host
phase continues under the same attempt, and the claim cannot be reacquired
without a new admission.

Why. The rule that device timing runs alone, the retention finding and the
manual ordering of device rows become one declared policy that the director
evaluates. Device idle time during host scoring is recoverable capacity.

Acceptance. Host tests with a fake capacity model for policy order, TTL and
capacity eviction, and release semantics; one device test that evicts a
resident session under memory pressure, resumes it by replay, and proves the
retained-memory bound holds.

### E3. Declared capacity: slot cost, labels and sticky assignment

Plan item: `declared-capacity`.

Observed mechanism. A worker declares a slot count; a task declares a slot cost
that it reserves on one worker; workers carry labels; a task declares desired
labels with a comparator, a required flag and a weight, and ties between
equally weighted workers are broken round-robin; sticky assignment keeps a
task's children on the worker that holds its state, with soft and hard modes.

Gap. `plan.Resources` is advisory and never acquires hardware. The locality
scheduler filters workers on artifact locality but has no notion of cost or
weight, so a device row and a host row cannot be co-scheduled by declaration,
and the reuse of one resident session across dependent rows is done by hand.

Contract. A row or operation declares slot cost in device and host units; a
lane declares labels and capacity; assignment requires every required label,
orders candidates by weight and breaks ties round-robin; a row may declare
sticky assignment to the lane holding its resident model, in soft or hard
mode. An exclusive device row is a row whose device cost equals the lane's
device capacity.

Why. The exclusive-device doctrine becomes a declared cost rather than a
prose rule. Host-only rows run beside a device row by construction. The
locality scheduler is extended rather than duplicated.

Acceptance. Host tests in `plan` and `loop` for cost accounting, label
comparison and weighting, round-robin ties, sticky modes, and the admission of
host rows while an exclusive device row runs.

### E4. Keyed admission strategies and status-based idempotency

Plan item: `keyed-admission`.

Observed mechanism. A run belongs to a concurrency group keyed by an expression
over its input. Strategies per group: group round robin with a maximum number
of running instances; cancel in progress, where a new run cancels the running
one; cancel newest, where a new run beyond the limit is cancelled; and cancel
queued except newest or oldest, which keep one queued run per group. An
idempotency key holds one run per key either for a TTL or while the owning run
is not terminal, with a fallback TTL as a bound.

Gap. Server requests, swap requests and automation triggers each carry their
own coalescing and cancellation logic. A new turn in the same conversation does
not supersede the in-flight one by declaration; queued swaps are drained by a
grace period rather than coalesced; a duplicate webhook trigger is not held by
key.

Contract. Admission derives a group key from declared request facts, namely
the conversation, model, workspace or automation identity, and applies a
declared strategy. `modelswap` coalesces queued swaps to the newest request
per endpoint. The automation runtime holds one in-flight run per idempotency
key with a fallback TTL. Every cancellation records a typed reason.

Why. Cancellation and coalescing become one admission decision per key with
recorded reasons, replacing per-owner grace periods and drains. Fairness
across users of one served model is a declared limit rather than an emergent
property.

Acceptance. Host tests in `server`, `modelswap` and `workflowruntime` for
each strategy, the swap coalescing order, and idempotency collisions and
release.

### E5. Two-phase deadlines with progress-refreshed execution timeouts

Plan item: `deadlines-and-requeue`.

Observed mechanism. A schedule timeout bounds how long a task may wait in the
queue before it is cancelled; an execution timeout bounds how long it may run;
a worker extends the execution timeout by reporting progress, and the engine
buffers those refreshes and flushes them at a bounded interval. Requeues use
exponential backoff with jitter and an overflow guard.

Gap. Overgo budgets bound execution only. A row waiting on a contested device
has no recorded deadline, so the wait is invisible until an operator reads a
log, and contested timing records stay provisional by convention.

Contract. Every dispatched operation carries a schedule deadline and an
execution deadline. Progress reports extend the execution deadline; the
extension is buffered and flushed at a bounded interval. A schedule deadline
that elapses cancels the operation with the reason recorded. Requeues back off
exponentially with jitter, bounded by a maximum and guarded against overflow.
Periodic controllers double their interval while they find no work and reset
when they do.

Why. Waiting becomes a measured, cancellable state. Long device rows extend
their own budgets by progress rather than by a generous static bound.

Acceptance. Host tests in `operation` and `loop` for both deadlines, refresh
buffering, requeue backoff bounds and idle-interval backoff.

### E6. Typed conditions on plan rows over recorded outcomes

Plan item: `conditional-plan-rows`.

Observed mechanism. A task in a declared graph may carry skip-if and cancel-if
conditions evaluated against a parent's output, and wait-for conditions over a
sleep or an external event, combined with or-conditions. A skipped task is a
terminal state that dependents can inspect.

Gap. `depends_on` is structural. A conditional row, for example "skip the
27B pass if the 12B guard fails", lives in the rationale text and is evaluated
by whoever reads it.

Contract. A step may declare conditions over the recorded outcome of a parent
row: skip when a typed field comparison holds, cancel when one holds, and
wait until a recorded event exists or an elapsed duration has passed.
Comparisons are typed field, operator and literal triples; no expression
language is embedded. `ReadyFrontier` evaluates conditions, and the completion
authority records skipped and cancelled rows with reasons so that dependents
resolve.

Why. The plan states conditional work as data the frontier evaluates, and a
skipped row leaves the plan through the same completion path as a done row.

Acceptance. Host tests in `plan` and `cmd/plan` for predicate evaluation,
skipped and cancelled completion authority, and frontier reporting of waiting
rows with reasons.

### E7. Keyed batches with declared flush conditions

Plan item: `batch-flush-contracts`.

Observed mechanism. A batch task accumulates inputs per key and flushes when
the batch reaches a maximum size, a maximum interval since the last flush, or a
payload byte bound.

Gap. The verification batches that landed in master declare checkpoints but
not when a batch flushes, and evaluation cases are batched per model by
convention.

Contract. A verification batch and an evaluation batch declare a key, a
maximum size, a maximum interval and a byte bound; flushing is deterministic
from those bounds; keys never mix.

Why. The gate's batch acceptance and benchmark batching share one flush rule,
and the gate cost per accepted checkpoint becomes measurable before and after.

Acceptance. Host tests in `plan` and `gate` for flush declarations and a
recorded gate-cost comparison.

### E8. Pause with queue-or-drop schedule behaviour

Plan item: `schedule-pause`.

Observed mechanism. A paused workflow keeps in-flight runs running, queues new
event and manual runs under a queue TTL, and lets cron and scheduled triggers
either queue or drop while paused; unpausing requeues what has not expired.

Gap. The recorded operator stop pauses the whole plan. A scheduled regression
guard or automation cannot be paused alone with a declared policy for its due
slots.

Contract. One automation may be paused with a declared queue-or-drop behaviour
for due schedule slots and a queue TTL; expired queued runs are cancelled with
the reason recorded; unpausing requeues the rest.

Acceptance. Host tests in `workflowruntime`.

### E9. Queue statistics and cancellation reasons as data

Plan item: `queue-observability`.

Observed mechanism. Every cancellation carries a reason shown per task. A task
statistics endpoint reports queued and running counts, the concurrency-group
distribution and the age of the oldest queued task, which autoscalers consume.

Gap. Cancellation reasons are not uniformly recorded, and the operator
workbench shows operation queues without queue age per key.

Contract. Every cancelled operation and row records a typed reason. The server
publishes queued and running counts with oldest age per key for operations
and lanes.

Acceptance. Host tests in `operation` and `server`.

### E10. A serving load lane with measured floors

Plan item: `serving-load-lane`.

Observed mechanism. A load-test command drives runs at a target rate, collects
per-run timings through a bounded-concurrency collector with a pending TTL, and
gates on thresholds.

Gap. Overgo benchmarks model quality and single-stream rates but never
measures serving under concurrent load. Queued-to-first-token latency and
throughput through the swap proxy at bounded concurrency are unmeasured.

Contract. A lane drives the served model at declared concurrency, collects
timings with bounded fetch concurrency and a pending TTL, publishes evidence,
and ratchets latency and throughput floors from measured runs the way the clone
baseline ratchets.

Acceptance. A host test for the collector and one device test that publishes a
measured run against declared floors.

### E11. State-keyed diagnosis and replay of a plan row

Plan item: `operator-references`.

Observed mechanism. Installable agent references describe, in fixed order, how
to read a run's state, its event log and its logs, followed by a diagnostic
table keyed on the observed state; a replay reference re-runs a workflow with
the same or new input.

Gap. Gate diagnostics now stream bounded failure evidence, but there is no one
command that renders a row's state, its attempt records and evidence tails
with the next action keyed on that state, and no replay of an attempt from its
receipt.

Contract. `cmd/plan` renders a diagnosis of one row from attempt records and
evidence, keyed on the row's state, and the loop replays one attempt from its
receipt with the same or new inputs.

Acceptance. Host tests in `cmd/plan` and `loop`.

## Elements not incorporated

- **PostgreSQL, RabbitMQ and NATS queues.** OvergoDB is the durable store and
  the engine is single-node. The queue semantics above are contracts on
  in-process owners.
- **The gRPC bidirectional worker protocol and language SDKs.** Overgo has no
  polyglot workers; UTCP manuals and native transports are the tool boundary.
- **Multi-tenancy, partitions, roles and the dashboard.** Out of scope for a
  single-operator system.
- **CEL expressions.** Conditions are typed field comparisons owned by the
  plan; an embedded expression language would add an unowned evaluator.
- **Cloud compute, autoscaling integrations, OpenTelemetry and Prometheus
  exporters.** Overgo records evidence in its store; queue statistics are
  published from there.
- **Webhook workers and event ingestion beyond the existing authenticated
  webhook intake.**

## Sequencing

Rows are ranked by their expected effect on automation cost and development
pace. Batch flush declarations and gate cost measurement come first because
gate wall per accepted row is the largest recurring cost of every later row.
Typed conditions follow because they let the frontier dispatch without an
operator reading rationale text. The two-phase deadlines and the durable
attempt log come next because they remove re-executed work after
interruptions; the four are independent and form the first frontier together
with batch flushing. Declared capacity follows because eviction and sticky
assignment are expressed in its units, and row diagnosis and replay follow it
because they consume the attempt log. Eviction and release depend on
capacity and on the deadlines that define waiting. Keyed admission depends on
deadlines for queued runs and precedes schedule pause, which reuses its
idempotency hold. Observability depends on recorded reasons. The load lane
depends on queue statistics. Operator references depend on the attempt log.
The closeout publishes the measurements that justify the campaign.

Every row lands through the gate with one acceptance test in the owning
package; the acceptance names are planned work, not claims that the tests
exist. Device tests are limited to eviction on the real device and the load
lane's measured run. Host rows land first.

## Measurements that decide the campaign

- Restart re-execution: the number of completed tool and model calls
  re-executed after an interruption, before and after the attempt log.
- Gate cost per accepted checkpoint, before and after batch flush contracts.
- Device idle time during host phases of device rows, before and after claim
  release.
- Queued-to-first-token latency and throughput at declared concurrency, as the
  load lane's first floors.

A row whose measurement shows no gain is closed with that outcome recorded;
the campaign does not require every element to win.
