package repoanalysis

import (
	"testing"

	"overgo/internal/gosource"
)

func TestPackageNamesPrefersProductionDeclaration(t *testing.T) {
	snapshot, err := (SourceSnapshot{}).Overlay(map[string][]byte{
		"internal/sample/sample.go":           []byte("package sample\n"),
		"internal/sample/zz_external_test.go": []byte("package sample_test\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{
		Root: ".",
		Files: map[string]bool{
			"internal/sample/sample.go":           true,
			"internal/sample/zz_external_test.go": true,
		},
		Packages: map[string]string{
			"internal/sample/sample.go":           "overgo/internal/sample",
			"internal/sample/zz_external_test.go": "overgo/internal/sample",
		},
	}
	names, err := PackageNames(snapshot, selection)
	if err != nil {
		t.Fatal(err)
	}
	if got := names["overgo/internal/sample"]; got != "sample" {
		t.Fatalf("package name = %q, want production declaration", got)
	}
}
