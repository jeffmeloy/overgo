package main

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

// TestLaneJudgesTypedLegOutcomes holds the lane to judging a run by its
// typed report and leg records, not by the words its tests log: a required
// leg counts only when the test that recorded it passed in the package that
// recorded it, so a same-named test of another package lends it nothing in
// either event order; a failed, unfinished or skipped named test, or a run
// without a pass, proves nothing.
func TestLaneJudgesTypedLegOutcomes(t *testing.T) {
	t.Parallel()
	const server, lane, shared = "overgo/internal/server", "overgo/internal/webuilane", "TestWebUIBrowserShared"
	// ran is one package's go test -json events: its start, each test's run
	// and terminal action (none when it never ended), then its verdict.
	ran := func(path string, outcomes ...[2]string) []string {
		event := func(action, test string) string {
			encoded, err := json.Marshal(map[string]string{"Action": action, "Package": path, "Test": test})
			if err != nil {
				t.Fatal(err)
			}
			return string(encoded)
		}
		events, verdict := []string{event("start", "")}, "pass"
		for _, outcome := range outcomes {
			events = append(events, event("run", outcome[0]))
			if outcome[1] != "" {
				events = append(events, event(outcome[1], outcome[0]))
			}
			if outcome[1] == "fail" {
				verdict = "fail"
			}
		}
		return append(events, event(verdict, ""))
	}
	judge := func(records []webuilane.LegRecord, required []string, events ...[]string) error {
		report, err := testevidence.GoTestJSONReport(strings.Join(slices.Concat(events...), "\n"))
		if err != nil {
			t.Fatal(err)
		}
		return webuilane.LaneVerdict(report, records, required)
	}
	passes := ran(server, [2]string{"TestWebUIBrowserPasses", "pass"})
	remembered := []string{"remembered settings leg"}
	recorded := []webuilane.LegRecord{{Package: server, Test: "TestWebUIBrowserPasses", Leg: remembered[0]}}
	if err := judge(recorded, remembered, passes); err != nil {
		t.Fatalf("a leg a passing test recorded was refused: %v", err)
	}
	if err := judge(nil, remembered, passes); err == nil {
		t.Fatal("a leg only logged, never recorded, was credited")
	}
	if err := judge(recorded, remembered, ran(lane, [2]string{"TestWebUIBrowserPasses", "pass"})); err == nil {
		t.Fatal("another package's same-named pass credited a leg")
	}
	if err := judge(recorded, remembered, passes, ran(lane, [2]string{"TestWebUIBrowserFails", "fail"})); err == nil || !strings.Contains(err.Error(), "failed=[overgo/internal/webuilane: TestWebUIBrowserFails") {
		t.Fatalf("a failed run was accepted: %v", err)
	}
	if err := judge(recorded, remembered, passes, ran(lane, [2]string{"TestWebUIBrowserHangs", ""})); err == nil || !strings.Contains(err.Error(), "unfinished=[overgo/internal/webuilane: TestWebUIBrowserHangs]") {
		t.Fatalf("an unfinished run was accepted: %v", err)
	}
	if err := judge(nil, nil, passes, ran(lane, [2]string{"TestWebUIBrowserSkips", "skip"})); err == nil || !strings.Contains(err.Error(), "TestWebUIBrowserSkips was skipped") {
		t.Fatalf("a skipped test was accepted: %v", err)
	}
	if err := judge(nil, nil, ran(lane)); err == nil || !strings.Contains(err.Error(), "no browser test passed") {
		t.Fatalf("a run without a pass was accepted: %v", err)
	}

	interleave := func(first, second []string) []string {
		var merged []string
		for index := range max(len(first), len(second)) {
			if index < len(first) {
				merged = append(merged, first[index])
			}
			if index < len(second) {
				merged = append(merged, second[index])
			}
		}
		return merged
	}
	leg := []string{"shared leg"}
	sharedRecord := []webuilane.LegRecord{{Package: server, Test: shared + "/leg", Leg: leg[0]}}
	outcome := func(path, action string) []string {
		return ran(path, [2]string{shared + "/leg", action}, [2]string{shared, "pass"})
	}
	for _, order := range [][2][]string{{outcome(server, "skip"), outcome(lane, "pass")}, {outcome(lane, "pass"), outcome(server, "skip")}} {
		if err := judge(sharedRecord, leg, interleave(order[0], order[1])); err == nil {
			t.Fatal("a same-named pass in another package credited a leg its own skipped test recorded")
		}
	}
	for _, order := range [][2][]string{{outcome(server, "pass"), outcome(lane, "skip")}, {outcome(lane, "skip"), outcome(server, "pass")}} {
		if err := judge(sharedRecord, leg, interleave(order[0], order[1])); err != nil {
			t.Fatalf("a same-named skip in another package withheld a leg its own passing test recorded: %v", err)
		}
	}
}
