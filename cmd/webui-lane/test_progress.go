package main

import (
	"bytes"
	"io"
	"slices"
	"strings"
)

// testProgress follows a go test -v stream as it passes through: which tests
// were started and which of them ended. When the run fails it can then say
// which tests failed and which never finished, in the lane's last line,
// where a timeout's own naming of the running test -- printed before a long
// goroutine dump -- is lost to whoever keeps only the end of the output.
type testProgress struct {
	out      io.Writer
	partial  []byte
	running  []string
	failures []string
}

// Write observes the whole lines data completes and passes data on unchanged.
func (p *testProgress) Write(data []byte) (int, error) {
	p.partial = append(p.partial, data...)
	for {
		line, rest, whole := bytes.Cut(p.partial, []byte{'\n'})
		if !whole {
			break
		}
		p.partial = rest
		p.observe(strings.TrimSpace(string(line)))
	}
	return p.out.Write(data)
}

// observe reads one line of the stream. A subtest is its parent's business:
// a parent that fails or hangs is named, and names the rest.
func (p *testProgress) observe(line string) {
	verb, name, _ := strings.Cut(strings.TrimPrefix(line, "--- "), " ")
	name, _, _ = strings.Cut(name, " ")
	if strings.Contains(name, "/") {
		return
	}
	switch verb {
	case "===":
		started, isRun := strings.CutPrefix(line, "=== RUN ")
		if started = strings.TrimSpace(started); isRun && !strings.Contains(started, "/") && !slices.Contains(p.running, started) {
			p.running = append(p.running, started)
		}
	case "PASS:", "SKIP:", "FAIL:":
		p.running = slices.DeleteFunc(p.running, func(test string) bool { return test == name })
		if verb == "FAIL:" {
			p.failures = append(p.failures, name)
		}
	}
}

// summary names the failed and the unfinished tests of a run that failed.
func (p *testProgress) summary() string {
	return "failed=[" + strings.Join(p.failures, ",") + "] unfinished=[" + strings.Join(p.running, ",") + "]"
}
