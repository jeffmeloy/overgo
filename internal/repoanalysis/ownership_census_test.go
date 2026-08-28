package repoanalysis

import (
	"path/filepath"
	"testing"
)

// TestRSIOwnershipCensusContract runs the census on the real repository
// and holds it to its contract: all ten declared mechanic families are
// reported, ownership arithmetic is internally consistent, the result
// is deterministic, and the census is computed -- an overlaid file that
// re-implements a mechanic moves the numbers.
func TestRSIOwnershipCensusContract(t *testing.T) {
	root := filepath.Join("..", "..")
	snapshot, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	findings, err := RSIOwnershipCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	families := map[string]CensusFinding{}
	for _, finding := range findings {
		families[finding.Family] = finding
	}
	for _, name := range []string{
		"storage", "publication", "process", "outcome", "failure",
		"retry", "capability", "trigger", "resource", "query",
	} {
		finding, present := families[name]
		if !present {
			t.Fatalf("family %q missing from census", name)
		}
		if finding.Sites != finding.OwnerSites+finding.Recurrence {
			t.Fatalf("family %q arithmetic: sites=%d owner=%d recurrence=%d",
				name, finding.Sites, finding.OwnerSites, finding.Recurrence)
		}
		if finding.Sites > 0 && finding.Owner == "" {
			t.Fatalf("family %q has %d sites and no owner", name, finding.Sites)
		}
		if len(finding.Consumers) > 0 && finding.Recurrence == 0 {
			t.Fatalf("family %q names consumers without recurrence", name)
		}
	}
	if families["process"].Sites == 0 || families["storage"].Sites == 0 {
		t.Fatalf("census blind to known mechanics: process=%d storage=%d",
			families["process"].Sites, families["storage"].Sites)
	}

	again, err := RSIOwnershipCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for index := range findings {
		if findings[index].Family != again[index].Family || findings[index].Sites != again[index].Sites ||
			findings[index].Owner != again[index].Owner || findings[index].Recurrence != again[index].Recurrence {
			t.Fatalf("census is not deterministic at %q", findings[index].Family)
		}
	}

	mutated, err := snapshot.Overlay(map[string][]byte{
		"internal/censusprobe/probe.go": []byte(
			"package censusprobe\n\nimport \"os/exec\"\n\nfunc probe() error {\n\treturn exec.Command(\"probe\").Run()\n}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	moved, err := RSIOwnershipCensus(mutated)
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range moved {
		if finding.Family != "process" {
			continue
		}
		if finding.Sites != families["process"].Sites+1 {
			t.Fatalf("overlaid exec site not counted: %d -> %d", families["process"].Sites, finding.Sites)
		}
		return
	}
	t.Fatal("process family missing after overlay")
}
