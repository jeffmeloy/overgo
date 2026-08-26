package agenttool

import (
	"testing"

	"overgo/internal/artifact"
)

func TestCatalogSnapshotSearchIsCanonicalAndBounded(t *testing.T) {
	weather := catalogSearchManual(t, "weather.forecast", "Forecast weather by place.", "place")
	repository := catalogSearchManual(t, "repo.search", "Search source code.", "pattern")
	first, err := NewCatalogSnapshot([]Manual{weather, repository})
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewCatalogSnapshot([]Manual{repository, weather})
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || first.Entries[0].Name != "repo.search" {
		t.Fatalf("catalogs differ: %+v %+v", first, second)
	}
	results, err := first.Search("weather place", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Manual != weather.ID || results[0].Score != catalogNameTermWeight+catalogFieldTermWeight {
		t.Fatalf("results = %+v", results)
	}
	exact, err := first.Search("weather.forecast", 1)
	if err != nil || len(exact) != 1 || exact[0].Score != catalogExactNameWeight {
		t.Fatalf("exact results = %+v, %v", exact, err)
	}
	if _, err := first.Search("weather", 0); err == nil {
		t.Fatal("zero result bound accepted")
	}
}

func TestCatalogSnapshotSearchTieOrderIsStable(t *testing.T) {
	alpha := catalogSearchManual(t, "alpha.probe", "Inspect target state.", "target")
	beta := catalogSearchManual(t, "beta.probe", "Inspect target state.", "target")
	snapshot, err := NewCatalogSnapshot([]Manual{beta, alpha})
	if err != nil {
		t.Fatal(err)
	}
	results, err := snapshot.Search("target", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].Name != "alpha.probe" || results[1].Name != "beta.probe" {
		t.Fatalf("tie order = %+v", results)
	}
	if results[0].Score != catalogFieldTermWeight {
		t.Fatalf("field score = %d", results[0].Score)
	}
	description, err := snapshot.Search("inspect", 2)
	if err != nil || len(description) != 2 || description[0].Score != catalogDescriptionWeight {
		t.Fatalf("description results = %+v, %v", description, err)
	}
	content, err := snapshot.ArtifactContent()
	if err != nil || content.Descriptor.ID.Kind() != artifact.KindProfile {
		t.Fatalf("catalog content = %+v, %v", content.Descriptor, err)
	}
}

func catalogSearchManual(t *testing.T, name, description, field string) Manual {
	t.Helper()
	manual, err := NewManual(Manual{
		Name: name, Description: description, Effect: EffectInspection,
		Arguments: []Field{{Name: field, Kind: FieldString, Required: true}},
		Transport: Transport{Kind: TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	return manual
}
