package processmeasure

import "time"

// Use is what the process spent over a span: its processor time, user and kernel, and the bytes it read through its
// files. A part's use is its own, as its wall is: what a part begun inside it spent is that part's.
type Use struct {
	CPU       time.Duration `json:"cpu_ns"`
	ReadBytes uint64        `json:"read_bytes"`
}

// minus is what was spent from earlier to u, less inner, which parts begun inside the span spent.
func (u Use) minus(earlier, inner Use) Use {
	return Use{CPU: u.CPU - earlier.CPU - inner.CPU, ReadBytes: u.ReadBytes - earlier.ReadBytes - inner.ReadBytes}
}

// plus is u with more spent.
func (u Use) plus(more Use) Use {
	return Use{CPU: u.CPU + more.CPU, ReadBytes: u.ReadBytes + more.ReadBytes}
}
