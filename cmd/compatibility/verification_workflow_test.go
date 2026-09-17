package main

import (
	"maps"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
)

func TestMediaReceiptAssertions(t *testing.T) {
	const pkg = "overgo/fixture"
	const assertion = "TestGeneration"
	valid := "{\"Action\":\"start\",\"Package\":\"overgo/fixture\"}\n" +
		"{\"Action\":\"run\",\"Package\":\"overgo/fixture\",\"Test\":\"TestGeneration\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/fixture\",\"Test\":\"TestGeneration\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/fixture\"}\n"
	for _, test := range []struct {
		name, output string
		required     map[string][]string
		passed       bool
	}{
		{"complete", valid, map[string][]string{pkg: {assertion}}, true},
		{"empty denominator", valid, nil, false},
		{"empty package denominator", valid, map[string][]string{pkg: {}}, false},
		{"unrelated pass", valid, map[string][]string{pkg: {"TestOther"}}, false},
		{"same name wrong package", strings.ReplaceAll(valid, pkg, pkg+"other"), map[string][]string{pkg: {assertion}}, false},
		{"omitted assertion", valid, map[string][]string{pkg: {assertion, "TestOther"}}, false},
		{"renamed assertion", strings.ReplaceAll(valid, assertion, assertion+"Renamed"), map[string][]string{pkg: {assertion}}, false},
		{"skipped assertion", strings.ReplaceAll(valid, `"pass"`, `"skip"`), map[string][]string{pkg: {assertion}}, false},
		{"failed assertion", strings.ReplaceAll(valid, `"pass"`, `"fail"`), map[string][]string{pkg: {assertion}}, false},
		{"malformed stream", valid + "{bad json}\n", map[string][]string{pkg: {assertion}}, false},
		{"unfinished package", valid[:strings.LastIndex(strings.TrimSuffix(valid, "\n"), "\n")+1], map[string][]string{pkg: {assertion}}, false},
		{"terminal without acquisition", strings.Join(strings.Split(valid, "\n")[2:], "\n"), map[string][]string{pkg: {assertion}}, false},
		{"repeated lifecycle", valid + valid, map[string][]string{pkg: {assertion}}, false},
		{"duplicate assertion", valid, map[string][]string{pkg: {assertion, assertion}}, false},
		{"two packages same name", valid + strings.ReplaceAll(valid, pkg, pkg+"other"), map[string][]string{pkg: {assertion}, pkg + "other": {assertion}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := checkMediaTestReceipt(test.name, []byte(test.output), test.required); (err == nil) != test.passed {
				t.Fatalf("accepted=%v want=%v: %v", err == nil, test.passed, err)
			}
			if got := strings.HasPrefix(mediaValidationVerdict([]byte(test.output), test.required), "Passed"); got != test.passed {
				t.Fatal("report and retained acceptance disagree")
			}
		})
	}
	t.Run("retained contracts", func(t *testing.T) {
		if testing.Short() {
			t.Skip(testskip.ShortIntegration)
		}
		root, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			t.Fatal(err)
		}
		roots, err := dataroot.Resolve(root)
		if err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		var bindings map[string]map[string][]string
		if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &bindings); err != nil {
			t.Fatal(err)
		}
		seen := map[string]bool{}
		accepted, rejected := 0, 0
		for _, path := range []string{mediaValidationPath, "docs/image_video_lifecycle.json", "docs/image_video_capacity.json"} {
			var index struct {
				Checks []struct {
					Name     string
					Evidence artifact.ID
					Required []string `json:"required_tests"`
				}
			}
			if err := jsonfile.Decode(filepath.Join(root, path), &index); err != nil {
				t.Fatal(err)
			}
			for _, check := range index.Checks {
				if seen[check.Name] || len(bindings[check.Name]) == 0 {
					t.Fatal("duplicate check or missing assertion contract", check.Name)
				}
				seen[check.Name] = true
				content, found, err := artifact.ReadContent(t.Context(), store, check.Evidence)
				if err != nil || !found {
					t.Fatal("retained acquisition absent", check.Name, err)
				}
				report, parseErr := testevidence.GoTestJSONReport(string(content.Data))
				err = checkMediaTestReceipt(check.Name, content.Data, bindings[check.Name])
				if parseErr != nil || testevidence.RequireComplete(report) != nil {
					if err == nil {
						t.Fatal("historical failure gained assertion credit", check.Name)
					}
					rejected++
					continue
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(check.Required) != 0 {
					requireMediaTestReceipt(t, check.Name, content.Data, check.Required, bindings)
				}
				accepted++
				substituted := maps.Clone(bindings[check.Name])
				for pkg, names := range substituted {
					delete(substituted, pkg)
					substituted[pkg+"-unrelated"] = names
					break
				}
				if checkMediaTestReceipt(check.Name, content.Data, substituted) == nil {
					t.Fatal("retained names credited to an unrelated package", check.Name)
				}
			}
		}
		if len(seen) != len(bindings) || accepted == 0 || rejected == 0 {
			t.Fatalf("contracts=%d checks=%d accepted=%d retained failures=%d", len(bindings), len(seen), accepted, rejected)
		}
		t.Logf("%d exact contracts: %d accepted, %d historical failures retained; wrong-package substitutions refused; no model execution or source promotion", len(seen), accepted, rejected)
	})
}
