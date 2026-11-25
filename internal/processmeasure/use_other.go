//go:build !windows

package processmeasure

// ReadUse reads what this process has spent by now where the platform reads it. Here it reads nothing, so a ledger's
// parts carry no use, as unmeasured, never as zero.
func ReadUse() (use Use, measured bool, err error) { return Use{}, false, nil }
