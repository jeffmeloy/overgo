package main

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/testevidence"
)

func TestCIRequiredEvidenceCommand(t *testing.T) {
	events := fmt.Sprintf(
		"{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestX\"}\n"+
			"{\"Action\":\"output\",\"Package\":\"x\",\"Test\":\"TestSlow\",\"Output\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestSlow\"}\n"+
			"{\"Action\":\"pass\",\"Package\":\"x\"}\n",
		testevidence.ShortIntegrationSkip+"\n",
	)
	var stdout, stderr bytes.Buffer
	code := run([]string{"./..."}, &stdout, &stderr, func([]string) (testevidence.GoTestReport, error) {
		return testevidence.GoTestJSONShortReport(events)
	})
	if code != 0 || stderr.Len() != 0 || !strings.Contains(stdout.String(), "tests=1 classified_short_skips=1") {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run(nil, &stdout, &stderr, func([]string) (testevidence.GoTestReport, error) {
		report, err := testevidence.GoTestJSONShortReport("{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestMystery\"}\n")
		return report, errors.Join(err, errors.New("exit status 1"))
	})
	if code == 0 || !strings.Contains(stderr.String(), "evidence rejected") {
		t.Fatalf("incomplete evidence was accepted: code=%d stderr=%q", code, stderr.String())
	}
}
