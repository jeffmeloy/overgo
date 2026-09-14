package codeprofile

import (
	"testing"

	"overgo/internal/repoanalysis"
)

// The imported function and a shadowing receiver must retain distinct owners.
// Deferred calls, callbacks and unresolved receivers remain conservative uses.
func TestSurfaceOwnerFlowWitness(t *testing.T) {
	const dependency = "internal/dependency/source.go"
	const consumer = "internal/consumer/source.go"
	snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{
		dependency: []byte(`package dependency
func Open(string) error { return nil }
`),
		consumer: []byte(`package consumer
import dep "example/internal/dependency"
type Reader struct{}
func (Reader) Open(string) error { return nil }
type Other struct{}
func (Other) Open(string) error { return nil }
func Imported(path string) error { return dep.Open(path) }
func Shadowed(dep Reader, root string) error {
 path := root + "/input"
 if err := dep.Open(path); err != nil { return err }
 defer dep.Open(path)
 return nil
}
func Callback(run func(string) error, path string) error { return run(path) }
func Bound(dep Reader, path string) error { return Callback(dep.Open, path) }
func Async(dep Reader, path string) { go dep.Open(path) }
func Unknown(dep interface{ Open(string) error }, path string) error { return dep.Open(path) }
`),
	})
	if err != nil {
		t.Fatal(err)
	}
	selection := repoanalysis.BuildSelection{
		Context:  "windows/amd64",
		Files:    map[string]bool{dependency: true, consumer: true},
		Packages: map[string]string{dependency: "example/internal/dependency", consumer: "example/internal/consumer"},
	}
	_, edges, _, err := ProductionConsumerGraph(snapshot, selection)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]map[string]int{
		"Imported": {"/call": 1},
		"Shadowed": {"Reader/interface": 1, "Other/interface": 1, "Reader/use": 1, "Other/use": 1},
		"Bound":    {"Reader/use": 1, "Other/use": 1},
		"Async":    {"Reader/use": 1, "Other/use": 1},
		"Unknown":  {"Reader/interface": 1, "Other/interface": 1},
	}
	for _, edge := range edges {
		if edge.To.Name != "Open" {
			continue
		}
		key := edge.To.Receiver + "/" + edge.Kind
		if want[edge.From.Name][key] == 0 {
			t.Fatalf("incorrect owner or relation: %+v", edge)
		}
		want[edge.From.Name][key]--
		if edge.To.Receiver == "" && edge.To.File != dependency || edge.To.Receiver != "" && edge.To.File != consumer {
			t.Fatalf("owner provenance mismatch: %+v", edge)
		}
		if edge.Line <= 0 || edge.Offset <= 0 {
			t.Fatalf("missing source site: %+v", edge)
		}
	}
	for caller, relations := range want {
		for relation, missing := range relations {
			if missing != 0 {
				t.Errorf("%s missing %d %s references", caller, missing, relation)
			}
		}
	}
}
