package testevidence

import (
	"encoding/json"
	"strings"
)

// Keep sequential invocations distinct without another execution path. The
// existing parser validates both the complete stream and each package run.
func goTestInvocationReports(out string, short, auxiliary bool) ([]GoTestReport, error) {
	var groups [][]goTestEvent
	active := map[string]int{}
	closed := map[string]bool{}
	_, err := readGoTestJSON(strings.NewReader(out), short, auxiliary, 0, nil, func(event goTestEvent) {
		if event.Package == "" {
			return
		}
		index, exists := active[event.Package]
		if !exists || event.Action == "start" && closed[event.Package] {
			index = len(groups)
			groups = append(groups, nil)
			active[event.Package] = index
			closed[event.Package] = false
		}
		groups[index] = append(groups[index], event)
		if event.Test == "" && (event.Action == "pass" || event.Action == "fail" || event.Action == "skip") {
			closed[event.Package] = true
		}
	})
	if err != nil {
		return nil, err
	}
	var reports []GoTestReport
	for _, events := range groups {
		var stream strings.Builder
		encoder := json.NewEncoder(&stream)
		for _, event := range events {
			if err := encoder.Encode(event); err != nil {
				return nil, err
			}
		}
		report, err := goTestJSONReport(stream.String(), short, false)
		if err != nil {
			return nil, err
		}
		reports = append(reports, report)
	}
	return reports, nil
}
