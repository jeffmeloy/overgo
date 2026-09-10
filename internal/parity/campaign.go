//overgo:runtime-inputs caller

// Package parity records ordered parity-campaign evidence.
package parity

import (
	"fmt"
	"math"
	"os"
	"time"
)

const logFileMode = 0o644

// Verdict identifies a stage evidence class.
type Verdict string

const (
	// Pass marks a stage whose asserted contract passed.
	Pass Verdict = "PASS"
	// Oracle marks a stage whose values match an external oracle.
	Oracle Verdict = "PASS-ORACLE"
	// Derived marks a stage whose structural facts were derived from its artifact.
	Derived Verdict = "PASS-DERIVE"
	// Wired marks a real forward stage without a value oracle.
	Wired Verdict = "PASS-WIRED"
	// Frontier marks an unverified boundary with a named prerequisite.
	Frontier Verdict = "FRONTIER"
	// Failed marks a stage that returned an error.
	Failed Verdict = "FAIL"
)

// Stage returns one parity verdict and its worst observed difference.
type Stage func() (Verdict, float64, string, error)

// Campaign owns ordered stage evidence and stops after the first failure.
type Campaign struct {
	logPath string
	failed  bool
	total   int
	counts  map[Verdict]int
}

// NewCampaign creates an empty campaign that appends evidence to logPath.
func NewCampaign(logPath string) *Campaign {
	return &Campaign{logPath: logPath, counts: map[Verdict]int{}}
}

// Log writes one timestamped campaign message.
func (c *Campaign) Log(line string) {
	full := time.Now().UTC().Format(time.RFC3339) + " " + line + "\n"
	fmt.Print(full)
	if file, err := os.OpenFile(c.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, logFileMode); err == nil {
		_, _ = file.WriteString(full)
		_ = file.Close()
	}
}

// Run executes a stage unless an earlier stage failed.
func (c *Campaign) Run(name string, stage Stage) {
	if c.failed {
		return
	}
	c.total++
	started := time.Now()
	verdict, worst, detail, err := stage()
	wall := time.Since(started).Round(time.Millisecond)
	if err != nil {
		c.failed = true
		c.counts[Failed]++
		c.Log(fmt.Sprintf("STAGE %-26s %-11s worst|d|=%-11s wall=%-9s %v", name, Failed, "-", wall, err))
		return
	}
	c.counts[verdict]++
	worstText := "n/a"
	if !math.IsNaN(worst) {
		worstText = fmt.Sprintf("%.3e", worst)
	}
	c.Log(fmt.Sprintf("STAGE %-26s %-11s worst|d|=%-11s wall=%-9s %s", name, verdict, worstText, wall, detail))
}

// Failed reports whether any executed stage failed.
func (c *Campaign) Failed() bool { return c.failed }

// Count returns the number of stages with verdict v.
func (c *Campaign) Count(v Verdict) int { return c.counts[v] }

// Total returns the number of executed stages.
func (c *Campaign) Total() int { return c.total }
