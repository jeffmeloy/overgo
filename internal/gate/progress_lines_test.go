package gate

import (
	"bytes"
	"testing"
	"time"

	"overgo/internal/testevidence"
)

// TestTestStepProgressLines pins the line each package prints as its test
// step reports it: the step running it, the package, go test's result and
// the elapsed it measured, written as the stream delivers them; the test
// runner's options carry the printer, and the stream is standard error
// unless bound.
func TestTestStepProgressLines(t *testing.T) {
	t.Parallel()
	var output bytes.Buffer
	g := &gateContext{progress: &output}
	options := g.goTestOptions(true, nil)
	if options.Progress == nil || !options.Short {
		t.Fatalf("test options = %+v, want the progress printer on a short run", options)
	}
	g.setTestStep(testOwnersCheckName)
	options.Progress(testevidence.PackageProgress{Package: "overgo/cmd/plan", Action: "pass", Elapsed: 41230 * time.Millisecond})
	g.setTestStep(testRestCheckName)
	options.Progress(testevidence.PackageProgress{Package: "overgo/internal/loop", Action: "fail", Elapsed: 1500 * time.Millisecond})
	options.Progress(testevidence.PackageProgress{Package: "overgo/cmd/sbom", Action: "skip"})
	want := "gate: step=test-owners package=overgo/cmd/plan result=pass elapsed=41.23s\n" +
		"gate: step=test package=overgo/internal/loop result=fail elapsed=1.5s\n" +
		"gate: step=test package=overgo/cmd/sbom result=skip elapsed=0s\n"
	if output.String() != want {
		t.Fatalf("progress lines:\n%s\nwant:\n%s", output.String(), want)
	}
	if (&gateContext{}).progressWriter() == nil {
		t.Fatal("an unbound gate prints progress nowhere")
	}
}
