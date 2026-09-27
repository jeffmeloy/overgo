package runrecord

// The gate's suite-cost record: one document per gate result, a lineage
// child of the result, carrying every test invocation's wall, package
// executions, the two costliest suites and the fixtures the run skipped.
// The plan's optimization review reads it, so its contract lives here.
const (
	// SuiteCostSchema is the suite-cost record's schema.
	SuiteCostSchema = "overgo/gate-suite-cost/v2"
)
