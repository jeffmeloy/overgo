package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
)

// TestReceiptKeepsTheStreamThatProvesAPass holds the receipt to publishing a
// complete passing go test -json stream byte for byte, so a reconciliation
// that binds it can re-read the tests it names, and to refusing a stream that
// proves no pass: a failed test, or a package that never finished, is not
// written at all.
func TestReceiptKeepsTheStreamThatProvesAPass(t *testing.T) {
	t.Parallel()
	passing := strings.Join([]string{
		`{"Action":"start","Package":"example/pkg"}`,
		`{"Action":"run","Package":"example/pkg","Test":"TestOne"}`,
		`{"Action":"pass","Package":"example/pkg","Test":"TestOne","Elapsed":0.01}`,
		`{"Action":"pass","Package":"example/pkg","Elapsed":0.02}`,
	}, "\n") + "\n"
	store := t.TempDir()
	id, err := publishReceipt(t.Context(), store, []byte(passing))
	testcheck.NoError(t, err)
	opened, err := overgodb.OpenReadOnly(store)
	testcheck.NoError(t, err)
	defer opened.Close()
	content, found, err := artifact.ReadContent(t.Context(), opened, id)
	if err != nil || !found || string(content.Data) != passing {
		t.Fatalf("published receipt differs from the stream: found=%v err=%v", found, err)
	}
	report, err := testevidence.GoTestJSONReport(string(content.Data))
	if err != nil || !report.PackagePassed("example/pkg") {
		t.Fatalf("the receipt does not read back as a pass: %v", err)
	}
	for name, stream := range map[string]string{
		"failed": strings.Join([]string{
			`{"Action":"start","Package":"example/pkg"}`,
			`{"Action":"run","Package":"example/pkg","Test":"TestOne"}`,
			`{"Action":"fail","Package":"example/pkg","Test":"TestOne","Elapsed":0.01}`,
			`{"Action":"fail","Package":"example/pkg","Elapsed":0.02}`,
		}, "\n") + "\n",
		"unfinished": `{"Action":"run","Package":"example/pkg","Test":"TestOne"}` + "\n",
	} {
		if _, err := publishReceipt(t.Context(), t.TempDir(), []byte(stream)); err == nil {
			t.Errorf("a %s stream was published as a receipt", name)
		}
	}
}
