package workflowruntime

import "overgo/internal/runrecord"

// ExecutionTriggers declares every trigger kind this runtime can fire
// when it enqueues execution: stage wakeups from its own scheduling,
// delegations on behalf of a running program, and recovery of stages
// found interrupted. There is no other enqueue path; a new one appends
// its trigger here, where the architecture test requires a complete
// registry contract for it.
var ExecutionTriggers = []runrecord.CausalTrigger{
	runrecord.TriggerStageWakeup, runrecord.TriggerDelegation, runrecord.TriggerRecovery,
}
