package codeprofile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCloneBaselineRatchet(t *testing.T) {
	if _, present, err := LoadCloneBaseline(filepath.Join(t.TempDir(), "absent.json")); err != nil || present {
		t.Fatalf("absent baseline: present=%v err=%v", present, err)
	}
	path := filepath.Join(t.TempDir(), "clone_baseline.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"duplicate_excess_nodes":10}`), 0o644); err != nil {
		t.Fatal(err)
	}
	baseline, present, err := LoadCloneBaseline(path)
	if err != nil || !present || baseline.DuplicateExcessNodes != 10 {
		t.Fatalf("baseline: %+v present=%v err=%v", baseline, present, err)
	}
	if err := AdmitCloneBaseline(baseline, 10); err != nil {
		t.Fatalf("at ceiling: %v", err)
	}
	if err := AdmitCloneBaseline(baseline, 11); err == nil {
		t.Fatal("above ceiling admitted")
	}
	if err := os.WriteFile(path, []byte(`{"version":2,"duplicate_excess_nodes":10}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadCloneBaseline(path); err == nil {
		t.Fatal("unsupported version admitted")
	}
}
