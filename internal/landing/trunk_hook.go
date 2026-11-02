package landing

// The hook that keeps the trunk for landings (scripts/githooks/reference-transaction).
const (
	trunkHookDirectory = "scripts/githooks"
	trunkHook          = "reference-transaction"
	// trunkMoveEnvironment marks a landing's ref updates to the hook.
	trunkMoveEnvironment = "OVERGO_TRUNK_MOVE"
)
