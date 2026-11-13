package testevidence

// CeilingFell is the line a test logs for a count that fell under the ceiling a constant of a test file holds: the
// constant, the file from the repository's root, the ceiling and the count. A ceiling only falls, so the test passes,
// and the landing whose tests said it writes the lower ceiling into its commit (refusal-advice/ceilings-fall).
const CeilingFell = "ceiling fell: %s of %s from %d to %d"
