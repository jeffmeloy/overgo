package testevidence

import (
	"overgo/internal/testskip"
	"slices"
	"strings"
	"testing"
)

func TestShortProfilePackageEvidence(t *testing.T) {
	start := packageEvent("start", "example", "", "")
	passed := packageEvent("run", "example", "TestUnit", "") + packageEvent("pass", "example", "TestUnit", "")
	excluded := packageEvent("run", "example", "TestIntegration", "") +
		packageEvent("output", "example", "TestIntegration", testskip.ShortIntegration) + packageEvent("skip", "example", "TestIntegration", "")
	terminal := packageEvent("pass", "example", "", "")
	for _, tc := range []struct {
		name, stream string
		short, want  bool
	}{
		{"short profile complete", start + passed + excluded + terminal, true, true},
		{"full profile incomplete", start + passed + excluded + terminal, false, false},
		{"no executed assertions", start + excluded + terminal, true, false},
		{"unclassified skip", start + passed + strings.ReplaceAll(excluded, testskip.ShortIntegration, "fixture missing") + terminal, true, false},
		{"missing start", start + passed + strings.Replace(excluded, packageEvent("run", "example", "TestIntegration", ""), "", 1) + terminal, true, false},
		{"missing resource", start + passed + excluded + packageEvent("output", "example", "TestIntegration", "UNAVAILABLE: missing fixture") + terminal, true, false},
		{"duplicate exclusion", start + passed + excluded + packageEvent("skip", "example", "TestIntegration", "") + terminal, true, false},
		{"late classification", start + passed + packageEvent("run", "example", "TestIntegration", "") + packageEvent("skip", "example", "TestIntegration", "") + packageEvent("output", "example", "TestIntegration", testskip.ShortIntegration) + terminal, true, false},
		{"failure before exclusion", start + passed + packageEvent("run", "example", "TestIntegration", "") + packageEvent("fail", "example", "TestIntegration", "") + excluded + terminal, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var updates []bool
			report, err := GoTestJSONReader(strings.NewReader(tc.stream), tc.short, len(tc.stream), func(pkg string, passed bool) error {
				if pkg != "example" {
					t.Fatalf("unexpected package %q", pkg)
				}
				updates = append(updates, passed)
				return nil
			})
			if err != nil || report.PackagePassed("example") != tc.want {
				t.Fatalf("profile credit=%t want=%t error=%v report=%+v", report.PackagePassed("example"), tc.want, err, report)
			}
			var wantUpdates []bool
			if tc.want {
				wantUpdates = []bool{true}
			}
			if !slices.Equal(updates, wantUpdates) {
				t.Fatalf("stream receipts=%v want=%v", updates, wantUpdates)
			}
			if tc.want && (report.PassedTests != 1 || !slices.Equal(report.ClassifiedSkipped, []string{"example: TestIntegration"})) {
				t.Fatalf("excluded test counted or lost: %+v", report)
			}
		})
	}
}
