package plan

// LanesDueEnvironment names, for a run of the lanes a runner starts, the file that keeps the commit due next (the
// runner's lanes_due.json): between packages the run yields to a newer commit named there (lanes-first/yield), what it
// ran kept as each test ended (RecordTestResults).
const LanesDueEnvironment = "OVERGO_LANES_DUE"
