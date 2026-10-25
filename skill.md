---
name: overgo-iteration
description: The rules for Overgo development, written as one logical statement per line. Use for implementation and explicit continue, resume or loop requests. Refactor as needed to optimize development pace, reduce code and test surface area and improve processing times and memory usage based on observed behavior.
---

# Intent

- overgo = one Go-native system that serves, trains, evaluates and composes models on consumer hardware
- purpose = recursive self-improvement through its own inference and training, unattended
- goal = improvement in what the system serves = the scorecard (`go run ./cmd/validate -scorecard`)
- scorecard columns = `first-output`, `throughput`, `held-out-quality`, `failure-rate`, `load-time`, `peak-device-memory`, `setup-steps`
- loop = measure -> find the gap -> train or change -> judge on held-out data
- row filed <=> it closes the loop or speeds the loop or serves inference or serves training
- progress = each scorecard cell compared with its previous measurement
- never a "passed" state and never a promotion
- `go run ./cmd/loop -improve` = measure -> publish the scorecard -> file each regression as a row
- `master` takes landings
- local `main` = GitHub staging, which nothing reads
- push <=> a request to push, never the system's own initiative
- this file and the plan's standing paragraphs = the system's doctrine
- doctrine = how the system works, which every session and the loop follow and correct on evidence
- doctrine != any person's preferences or rules
- a rule that evidence shows costly or wrong => correct it in the doctrine != cite it as authority
- requests and models propose
- goals, budgets and stops = inputs the system follows (the plan's intent, resource ceilings, a recorded stop) and never sets itself
- executable policy admits and activates
- OvergoDB holds state
- the plan holds implementation
- every change aims at capability or performance or development velocity or memory use
- bold change > cautious increment (a rewritten slow path, a deleted subsystem, a new kernel, a different data layout)
- framework = the gate + the held-out scorecard + the cycle
- the framework catches regressions and broken code
- bold change => write the test that would have caught its failure -> let the framework judge
- never shrink a change to what can be checked by hand
- never hedge a change with flags or shims
- never defer a change for fear of breaking what the framework checks
- refused landing = information != failure
- ambition governs the code and never the system's boundaries
- boundaries = scope, budgets, stops, pushes and downloads, which the system follows and never changes
- keep correctness and quality
- simpler implementation > larger one
- ambition = size of the gain != size of the change

## Terms

A term used in this file is plain English or is defined here.

- lead = the session that holds the main checkout, ranks the plan and keeps the doctrine current
- session = one running agent
- row = one unit of work in `docs/plan.json`
- step = one landing's worth of a row, with its check (`verify`)
- gate = the automated checks a commit must pass
- landing = a commit made through the gate
- lane = one of the gate's longer test runs (device, browser, cuda)
- cycle = the full measurement run (`go run ./cmd/loop -improve`)
- cell = one model, task and measure on the scorecard
- home = the one package that holds a mechanic
- leftover = a file that a run, a build or a session wrote and that no later work reads
- place = a directory where leftovers accumulate

## Writing

Applies to this file, plan paragraphs and rationale, code comments and memory files.

- writing = objective logic != shortened prose
- one statement per line
- statement = subject + relation + object
- relation = one symbol from the table or one plain verb
- colon or semicolon never joins two clauses because it hides the relation
- colon allowed <=> inside code, a path, a time, a quotation, or after a field label a tool requires (`Cause:`)
- statement cannot name its relation => find what it claims or delete it
- term => plain English or defined in Terms
- label invented for a design => defined in Terms or not used
- number => its unit and its source
- example or evidence = one clause in parentheses
- intent first => state what must hold, never how the current code does it
- history (a date, a commit id, "was", "now") -> the commit message, never doctrine and never a comment
- command usage -> the command's help text, never this file
- conciseness measure = unnecessary words and redundancy, never a word count
- unnecessary = a word, clause or entry whose removal changes nothing a reader does
- redundant = stated twice, or already told by the code or a command's help or Git or the plan
- each rule has one home
- reply to a person = plain standard English sentences != this notation

| Write | Read as |
| --- | --- |
| `A => B` | if A then B |
| `A <=> B` | A exactly when B |
| `A -> B` | A then B, or A goes to B |
| `A = B` | A is B |
| `A != B` | A is not B |
| `A > B` | prefer A over B |
| `never A` | A is forbidden |
| `A and B`, `A or B` | both, either |
| `A because B` | B is the reason for A |
| `A except B` | A holds unless B |

### Code comments

- doc comment's first line = `Name <verb> ...`, one plain sentence
- following lines = one statement per line, as a list
- comment states the contract = what holds + what is refused + the reason
- comment never narrates the change
- comment never restates the code below it
- comment names no term that Terms or the package's own doc does not define
- named constant other than 0 or 1 => its comment gives the source, the scope and when to reconsider it
- error text = what was refused + the value seen + what is accepted

**Before**

```go
// freeSpaceFloor is the least free space a commit's checks may start with:
// the sum, since a measurement run writes beside commits.
```

**After**

```go
// freeSpaceFloor is the least free space a commit's checks may start with.
//   - freeSpaceFloor = largestCheckWrites + measurementRunWrites
//   - sum because a measurement run writes beside commits
```

## Fewer components

- the repository shrinks while capability grows
- every change removes something
- net production lines flat or falling except the row declares growth
- growth that buys capability or speed or memory = declared and welcome
- one change retiring two mechanisms > two changes
- before building => delete
- build for the case that exists
- generalize <=> a second case exists
- existing tools and signals > new ones
- before a new home => add a mode or a consumer to an existing home
- refactor the growth pattern, never only its latest instance
- rule that matters = code the gate runs
- ledger of objects lives only until invalid objects are impossible
- classify once, at the home, from typed facts
- never a regex over structured data
- capability != dead code
- path only tests reach = unwired => add its route or lane or command
- delete such a path <=> its input chain is half missing and it is record ceremony and it costs now and a rebuild would redesign it
- model prototypes stay
- "no active model" != grounds to delete
- never markdown behavior control => an idea becomes a Go test or a gate rule

## Autonomous feedback

- stay inside the scope and resource ceilings the inputs set
- robustness governs autonomy
- efficiency guides selection
- runs, regressions, failures and recurring costs -> durable obligations
- proposal = measurement + prior attempts + home + budget + hypothesis + benefit + cost + acceptance comparison + rollback
- a measurement, or no action, = a valid outcome
- rank by recorded outcomes and expected cost
- exploration = bounded
- rejection feeds the next decision
- missing resource => explicit resume condition and continue independent work
- persist position, dispositions, outstanding checks and budget => resume without duplicates
- compare predictions with results
- freeze acceptance before the work
- judge a policy change on an independent task set

## Working loop

1. read `git status --short`, the plan row, the home's code and stored results
   - after an interruption => read them again
   - preserve others' work
2. the request's scope first
   - campaign => `go run ./cmd/plan -next`
   - honor dependencies, lanes, acceptance and stops
3. state the change, its expected effect and the deciding comparison
   - probe <=> the uncertainty could change the design
4. fix the cause in its home -> migrate callers -> delete displaced code
5. the step's verify fails before the change and passes after it
   - preflight first = `go run ./cmd/gate -plan <item>/<step> -message-file - -preflight`
   - after a hand edit => `go run ./cmd/validate -claims`
6. land = `go run ./cmd/loop -land <item>/<step> -message-file -`
   - message carries `Cause:` and `Predicted effect:`
   - predicted effect names scorecard metrics, and a worsening in words
   - the gate writes the numbers
   - run unpiped
   - landed <=> `git log -1` shows `Landed-Row:`
   - never a raw commit
   - merge = `git merge --no-ff --no-commit <branch>` -> resolve -> land
7. open decision => take the recommended default -> record `plan -answer` -> continue

- landing returns to dispatch
- pause <=> an explicit stop request (`go run ./cmd/plan -stop <reason>:<detail>`, its reasons in the command's help)
- lift a pause = `-resume-stop`
- prompt => answer -> continue
- question => answer
- instruction => work
- one committing session per checkout
- another session => its own worktree and the commit queue, or review only
- never touch another session's work
- `go run ./cmd/loop` supervises unattended work (`.overgo-runtime/loop.json`)
- one lock per worktree
- never copy the loop's session token
- off-scope finding => `go run ./cmd/finding`

### Gate and cycle

- the gate judges the commit from the tree alone (unit, golden, hermetic, small device)
- a store write never changes the gate's verdict
- the cycle (`validate -run`) judges models (real-model, live-store and device suites, by build tag)
- cycle phases = cells -> publish -> lanes
- interrupted cycle => resumes at the first unpassed phase when `go run ./cmd/loop -improve` runs again
- cycle running => plan-only landings only and never a build, a test or a push beside it
- timing taken beside other work != a measurement
- gate's first stage = `space` => refuses below a free-space floor on the temporary directory's volume
- `space` applies to preflights and plan-only landings too
- `space` refusal = external => free space -> land again
- hand step = system defect => change the system so the step does not recur
- landing cost = lanes reached + inference-surface guard expiry + pinned-media reconciliation
- batch surface moves
- landing grows its row past its count => raises the count with a reason in its own plan edit (gate rule)
- step's check that refuses text in `docs/plan.json` => its pattern never matches itself, because the plan stores the check
- correction of a kind of error => a standing test refuses the kind wherever the doctrine lives, never a step's one-off match and never only the instance reported
- step's check that runs `go test` = the bare command with an anchored `-run`, never piped, because the verifier reads its test events
- filing a check => run it once (`plan -verify`): it fails as a test or names a test not yet written, never vacuous by its form (a pipe, a wrong package), because a landing is judged by the check its row held before it
- land before an evaluation run
- native `go test` selects
- lane's selection = every package whose test binary compiles a changed package, read from what the binary compiles (all its test files, every package under a recursive pattern), never from a proxy (production imports, a pattern's top directory)
- never time budgets, ratchets or a second selector
- pin boundaries as table rows

### Plan and SQA

- `docs/plan.json` = sole dispatch authority
- plan update => first review the record since the last update (landings, retirements, regressions with how each was found, fixes, findings) -> name the patterns that held and that failed -> apply them to the plan and to this doctrine in the same landing -> its message states them
- retirement => its record names the change it waits for -> that change lands => the retired change is measured again on it
- rationale lines = Serves, Disappears, Evidence, Batch
- `docs/sqa_findings.json` = independent, scheduled, advisory
- the lead owns adoption and sequence
- first-severity finding => a row at once
- high-priority `user_finding` entry => answer it
- declining an approach != refuting its evidence
- `implementation_review` = one comment per open finding (`disposition`, `comment`, `plan_steps` or `evidence`)
- the reviewer never writes responses
- before writing there => read it again and keep the other writer's fields

## Evidence

- bind source, model, data, recipe, policy and environment identities exactly
- reuse or resume <=> inputs and contracts match
- one run serves every obligation with matching identity and contract (one record, many readers)
- a run never judges its own output
- a run's evidence = complete by its own contract
- unknown provenance => keeps its value and earns no compatible evidence
- replay carries only bit-identical outputs
- evaluate decides
- compare identical tasks, artifacts, protocols and budgets
- two builds' timings => alternate the builds on fresh processes for at least 3 rounds each (a bench cell moves 10% between rounds)
- difference inside the rounds' ranges = no difference
- an A/B => its failed requests counted beside its rates, because a pool of the requests that succeeded hid a build that corrupted the card's experts (14 requests failed, its rates within the others' ranges)
- a cost each request pays once => weighed against the measured cell's own request (its tokens and time), never a longer run's
- a timing taken under a tracer => an untraced clock of the same stage confirms it before it decides (per-node graph tracing slows a graph's launch)
- work across streams = the union of their intervals, never a sum of durations, and a copy beside a run costs the run's time with it minus without, never its intervals' overlap, because a host-to-card copy slowed the draft round beside it by about its own time (2.1 against 1.15 ms) and the verify by the same 2 ms when started beside it
- an idle card in a trace => each kernel's launch joined to its start (the correlation id) before a change, because work launched and waiting and work not yet launched take different fixes (the prompt layers' products waited 5.2 ms each after their launch)
- aggregate gain never hides a required regression

## Robustness

- persist results and obligations
- publish and retry = idempotent
- recover from durable state
- retried failed write never becomes a pass
- one home a resource
- bound workers
- handle cancellation, error, death and release
- OS locks only
- never infer abandonment from elapsed time
- never steal a lock
- model, dataset and checkpoint inputs = immutable
- publish through the lifecycle's home
- before retiring originals => validate rollback
- execution, verification and activation = separate
- keep rollback predecessors
- test failure and recovery beside success
- expose missing, invalid and partial state
- honor guards and authorization
- never a silent reset, a semantic change or a weakened acceptance
- failing check => fixed, never relabeled
- no clock decides failure => signals decide
- short timing => `processmeasure.Counter`
- change OS security settings (Defender, UAC) as needed

## Efficiency

- measure whole attempts (proposal, failure, wait, acquisition, verification, recovery)
- elapsed time != summed parallel time => report them apart
- count repeated loads
- reuse valid artifacts, checkpoints and check results
- reacquire <=> invalidated
- selection, ordering, reuse and reporting = from typed facts (AST, types, parsed commands, declared capabilities)
- external text => normalize at the boundary
- scope verification through the dependencies' homes
- unknown dependency => keep the broader scope
- conflicting mutations and isolated measurements => serialized
- CPU work holds no GPU claim
- concurrency, caching or fusion <=> measured benefit
- remove unused flags, wrappers, state and exports
- inconclusive optimization => new hypothesis or stop
- never rerun for a favorable result
- cheapest method first = reuse data and checkpoints -> post-hoc eval -> free training signals -> per-layer norms -> per-step decompositions

### Speed work

- speed row => its ceiling first = the work skipped or stubbed in the real step, timed in-graph != a kernel-level trace or an estimate
- ceiling within the measurement's spread => the row is not built and records its ceiling
- speed rows ranked by the measured gap against the reference, phase by phase on the same inputs != by estimates
- port a reference's mechanism <=> its costs measure the same here (hit rates, link, launches, CPU time)
- port a reference's default or compare against its figures => read the configuration those figures ran under (its setup's defaults, launch arguments, environment), never its engine's default alone, because a default the measured configuration turns off adds work or an approximation the figures never paid for
- reference loop whose parts are co-designed => ported as a unit: map it against ours on the same inputs, then build its parts in the order the measurement shows the others' gains rest on, judged together, because a part moved alone keeps our costs around it
- reference's memory model (what it keeps where, what it caches) => ported before tuning what moves between places
- a rate both engines report => check that both count the same thing before comparing (per routed slot != per expert a run)
- A/B = one loaded model, forced token ids, interleaved arms > a reload and free-running text an arm
- probe or benchmark built from stale derived files (kernels, bindings, manifests) => refused before it runs
- rare-path work (a handshake, a fallback) => folded into kernels that run anyway > a launch of its own
- work that waits for nothing => queued and submitted before work that waits (a copy batch, an event), because under Windows' display driver model a stream's work waits behind the submissions made before it
- a buffer sized to a run's need => rounded up before it grows, because each growth frees it, drops what names it and waits while Windows clears the new memory (about 100 ms at 2.8 GB); a share to spare costs what that memory would hold (experts)
- a tuning value (a threshold, a size or a count that trades measured costs: a chunk's tokens, a tile's rows, an expert's admission) => its comment names the cost it minimizes and the measured inputs it is computed from (the link's rate, the bytes a token holds, an expert's bytes, a miss's cost, the request's tokens), because a sweep returns only a point it tried, for one request shape on one machine (2,048-token chunks paid for their slower read after about 700 decoded tokens)
- inputs that vary at runtime (the request, the free memory) => the value computed where it is used > a constant
- a value taken from a reference engine or a sweep => provisional, with its derivation step filed
- a sweep => checks the formula's prediction, and a disagreement names a cost the formula misses
- a policy over a stream of requests (an admission, an eviction) => replayed on a recorded trace beside the hindsight bound before a bench, because the replay prices a policy in minutes and says how far any rule can go (admitting at a first miss came within 4.6 CPU experts a window of the bound, Strata's second use 22.5)
- memory a reference frees for another use => taken where the reference takes it (its coldest slots, across layers), because freeing by allocation order took whole layers' hot experts (a release now keeps a freed allocation's hotter experts in colder ones' slots)
- memory moved between two uses (a read's arena and the experts) => lent, the part the next use needs taken back, never freed and allocated again, because each allocation waits while Windows clears it and each free moves experts out (freed, a 16K read's arena and the expert releases added about 220 ms to its first token)
- measured gains of one row => one landing (steps joined by commas) > a gate run each

## Compute and derived values

- RAM, VRAM, CPU and GPU capability = resolved through the resources' homes with load, reservations and limits
- installed capacity != budget
- never a machine-specific default
- placement, batch, chunk, workers and residency = derived from workload, measurement and availability
- scale down by streaming, tiling and bounded concurrency
- scale up <=> measured benefit
- preserve numerical contracts and checkpoint identity
- nothing fits => resumable wait or reported limit
- number other than 0 or 1 => named constant with a reason (gate rule)
- thresholds, tolerances, sample sizes and optimizer settings = derived from declarations or observations or math
- cannot derive a value => delete the mechanism
- limit with no derivation = arbitrary literal => replace it with a rule
- keep exact identities, external formats and justified policy
- kept value => record its source, scope and reconsideration condition in the owning contract
- distribution, geometry or metric assumption (Gaussian, independent, Euclidean, linear interpolation) => justified per model and task

## Training

- optimizer = Muon only
- rates = derived
- policy = `internal/trainingprogram/optimizer_policy.json`
- rate = P^exponent
- exponent = -0.6 for a trained model
- exponent = -0.5 from initialization
- momentum from N=30
- pass `BaseLearningRate: 0`
- optimizer math on device
- only scalars cross to host
- `cmd/train` = the one entry (`-route`, `-objective`)
- dense tied-embedding model => `-freeze-lexical`
- quantized format => native fast path > output identity with bf16, guarded by held-out quality

## Implementation

- Go owns runtime, orchestration and tooling
- never runtime cgo
- never Python in the tree
- goldens = data from the Python original
- CUDA through the kernel manifest
- scratch -> `tmp/<row id>/`, the folder named for the plan row that made it
- executables -> `bin/`
- Python lives only in `scripts/python_golden_references/` (git-ignored)
- those scripts = producers of a golden reference from the Python original
- the gate and the Go code never run those scripts
- each script's header = what it produces + the Python original + library versions + seed + output file or store identity
- evaluation sets, tools and anything else = Go
- models, datasets, store and `build/` live in `OVERGO_DATA_ROOT`, outside the module
- port through typed contracts and recipe-selected consumers
- llama.cpp for pinned behavior
- `adaptive_new` = read-only
- keep commits, licenses and artifact identities
- model-family facts live in artifacts and recipes, never in model-specific code
- cohesive homes
- clear errors and cancellation
- reusable hot-path buffers
- interface <=> real consumers
- `gofmt` and `go vet`
- numerical or gradient change => reference comparison
- JSON edit => `cmd/plan` or a Go program
- code edit => exact edits

### Cleanup

- cleanup = deterministic code
- cleanup != a model's judgement and != a question to a person
- every place => a rule that ends in keep or delete
- entry kept <=> named by the store or a committed file or an open plan row or a running process
- entry not kept and its run's record landed => deleted
- command that makes leftovers => deletes them when it writes its record
- decision outside the system's authority (a branch is retired) => recorded once as data -> code acts on it
- cleanup request met <=> bytes or files or code are gone
- tooling, a plan row or a command handed back to a person != cleanup
- campaign close-out refused <=> a deletable entry remains
- fixture => named by a test

## Verification

- test observable contracts and failure modes with controlled inputs
- test pins a contract (outputs, the kernel chosen, launch counts) != an internal layout (a trace record's byte offsets)
- missing, skipped, incomplete or zero-match check => earns no pass
- framework = the safety net => a bold change ships with the check that would have caught its failure
- framework has no such check => add it (a Go test, an invariant row, a gate rule) and never reduce the change
- vary shape, distribution and resource
- keep negative and inconclusive results
- report denominators, exclusions, scope and environment
- real models one at a time, smallest first
- parity golden = upstream output with provenance (repository, commit, checkpoint, input)
- comparing overgo with its own earlier output => stacks tolerances
- tools = `cmd/validate` (`-scorecard`, `-run`, `-claims`, `-tests`, `-smoke`, `-parity`, `-receipt`), `cmd/device-lane`, `cmd/webui-lane`

## Sources of truth

| Surface | Holds |
| --- | --- |
| [skill.md](skill.md) | The system's doctrine. Sole root agent-instruction document |
| [docs/plan.json](docs/plan.json) | Rows, dependencies, acceptance, open work, and rules that bind one campaign only |
| OvergoDB | Artifacts, lineage, runs, decisions, findings, receipts (`go run ./cmd/overgodb query`) |
| [compatibility.json](compatibility.json) | Model and capability verification contracts |
| [kernels/manifest.json](kernels/manifest.json) | Kernel ABI, source and binary identities |
| [API manifest](docs/api_manifest.json) | Public commands, routes, document contracts |

- existing homes only
- never parallel plans, ledgers, schedulers or recovery paths
- contracts in docs, measurements in the store, history in Git
- report = outcome + material measurements + checks + remaining obligations
