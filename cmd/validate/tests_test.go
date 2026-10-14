package main

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
)

func TestCIRequiredEvidenceCommand(t *testing.T) {
	t.Parallel()
	events := fmt.Sprintf(
		"{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestX\"}\n"+
			"{\"Action\":\"attr\",\"Package\":\"x\",\"Test\":\"TestSlow\",\"Key\":%q,\"Value\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestSlow\"}\n"+
			"{\"Action\":\"pass\",\"Package\":\"x\"}\n",
		testskip.Key, testskip.KindShort,
	)
	var output bytes.Buffer
	err := runTests([]string{"./..."}, &output, func([]string) (testevidence.GoTestReport, error) {
		return testevidence.GoTestJSONReader(strings.NewReader(events), true, clioptions.DiagnosticTailBytes, nil)
	})
	if err != nil || !strings.Contains(output.String(), "tests=1 classified_short_skips=1") {
		t.Fatalf("err=%v output=%q", err, output.String())
	}

	output.Reset()
	err = runTests(nil, &output, func([]string) (testevidence.GoTestReport, error) {
		report, err := testevidence.GoTestJSONReader(strings.NewReader("{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestMystery\"}\n"), true, clioptions.DiagnosticTailBytes, nil)
		return report, errors.Join(err, errors.New("exit status 1"))
	})
	if err == nil || !strings.Contains(output.String(), "evidence rejected") {
		t.Fatalf("incomplete evidence was accepted: err=%v output=%q", err, output.String())
	}
}
