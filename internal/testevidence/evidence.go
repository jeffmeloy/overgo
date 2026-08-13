// Package testevidence validates that successful test processes ran assertions.
package testevidence

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
)

// GoTestJSON rejects skipped tests and unavailable oracle markers in go test
// -json output. Package rows without tests remain neutral: derived gate scope
// may include importers that intentionally own no tests.
func GoTestJSON(out string) error {
	scanner := bufio.NewScanner(strings.NewReader(out))
	seen := false
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event struct {
			Action  string
			Package string
			Test    string
			Output  string
		}
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			return fmt.Errorf("decode go test event: %w", err)
		}
		seen = true
		if reason := unavailable(event.Output); reason != "" {
			return fmt.Errorf("%s", reason)
		}
		if event.Action == "skip" && event.Test != "" {
			return fmt.Errorf("%s: %s skipped", event.Package, event.Test)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read go test events: %w", err)
	}
	if !seen {
		return fmt.Errorf("go test emitted no events")
	}
	return nil
}

// VerifyOutput rejects successful shell verification that did not prove its
// Go tests ran. Non-Go commands retain ordinary exit-code semantics.
func VerifyOutput(command, out string) error {
	if reason := unavailable(out); reason != "" {
		return fmt.Errorf("%s", reason)
	}
	if strings.Contains(out, "[no tests to run]") {
		return fmt.Errorf("go test filter matched no tests")
	}
	if strings.Contains(out, "[no test files]") {
		return fmt.Errorf("package has no tests")
	}
	if strings.Contains(out, "--- SKIP:") {
		return fmt.Errorf("go test skipped a test")
	}
	if strings.Contains(command, "go test") && !strings.Contains(out, "--- PASS:") {
		return fmt.Errorf("go test output contains no explicit passing test")
	}
	return nil
}

func unavailable(out string) string {
	switch {
	case strings.Contains(out, "UNAVAILABLE"):
		return "output contains UNAVAILABLE"
	case strings.Contains(out, "parity NOT verified"):
		return "output reports parity NOT verified"
	default:
		return ""
	}
}
