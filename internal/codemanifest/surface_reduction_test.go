package codemanifest

import (
	"encoding/json"
	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestSurfaceManifestRoundTrip(t *testing.T) {
	for _, large := range []bool{false, true} {
		t.Run(strconv.FormatBool(large), func(t *testing.T) {
			value := fixtureManifest()
			if large {
				boundary := Uncertainty{Kind: UncertaintyAnalysis, Path: "internal/fixture/0.go", Reason: strings.Repeat("x", maxPathBytes)}
				encoded, err := json.Marshal(boundary)
				if err != nil {
					t.Fatal(err)
				}
				value.Uncertainty = make([]Uncertainty, artifact.MaxContentBytes/len(encoded)+1)
				for index := range value.Uncertainty {
					value.Uncertainty[index] = boundary
					value.Uncertainty[index].Path = "internal/fixture/" + strconv.Itoa(index) + ".go"
				}
			}
			manifest, err := identifyManifest(value)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			if (len(encoded) > artifact.MaxContentBytes) != large {
				t.Fatalf("graph bytes=%d, storage limit=%d, large=%t", len(encoded), artifact.MaxContentBytes, large)
			}
			replayed, err := Parse(encoded)
			if err != nil || replayed.ID != manifest.ID {
				t.Fatalf("analysis round-trip ID=%s, want %s: %v", replayed.ID, manifest.ID, err)
			}
			if _, err := Parse(append(encoded, '\n')); err == nil {
				t.Fatal("noncanonical analysis bytes accepted")
			}
			if _, err := manifest.Content(); (err != nil) != large {
				t.Fatalf("storage admission large=%t: %v", large, err)
			}
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			if _, err := PublishDigest(t.Context(), store, manifest); err != nil {
				t.Fatal(err)
			}
			if _, err := PublishDigest(t.Context(), store, manifest); err != nil {
				t.Fatalf("repeat digest publication: %v", err)
			}
			digest, err := LoadDigest(t.Context(), store, manifest.ID)
			if err != nil || digest.ID != manifest.ID {
				t.Fatalf("digest round-trip: %+v, %v", digest, err)
			}
			if _, err := Load(t.Context(), store, manifest.ID); err == nil {
				t.Fatal("digest publication stored the full graph")
			}
			manifest.SourceIdentity = strings.Repeat("f", len(fixtureDigest))
			if err := manifest.Validate(); err == nil {
				t.Fatal("changed source retained its old identity")
			}
		})
	}
}

func TestSurfaceAnalysisReuse(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/example.go"
	testName := "internal/example/example_test.go"
	writeGeneratorFixture(t, root, name, "package example\nfunc A(x int) int { if x > 0 { return x + 1 }; return 0 }; func B(x int) int { if x > 0 { return x + 1 }; return 0 }\n")
	writeGeneratorFixture(t, root, testName, "package example\nfunc C(x int) int { if x > 0 { return x + 1 }; return 0 }; func D(x int) int { if x > 0 { return x + 1 }; return 0 }\n")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	selection := repoanalysis.BuildSelection{Context: "linux/amd64", Root: root, Files: map[string]bool{name: true, testName: true}, Packages: map[string]string{name: "overgo/internal/example", testName: "overgo/internal/example"}}
	cache, err := NewCache(1)
	if err != nil {
		t.Fatal(err)
	}
	if _, found := cache.Profile(snapshot); found {
		t.Fatal("uncomputed profile reused")
	}
	manifest, reused, err := cache.Generate(snapshot, []repoanalysis.BuildSelection{selection}, nil)
	if err != nil || reused {
		t.Fatalf("initial generation: reused=%t err=%v", reused, err)
	}
	want, err := codeprofile.Build(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Clones) != 2 || want.Clones[0].AdvisoryClass != "" || want.Clones[1].AdvisoryClass != "test" {
		t.Fatal("equal clone fingerprints lack canonical class ordering")
	}
	assertProfile := func() {
		t.Helper()
		got, found := cache.Profile(snapshot)
		if !found || !reflect.DeepEqual(got, want) {
			t.Fatalf("profile changed or missing: found=%t", found)
		}
	}
	assertProfile()
	got, _ := cache.Profile(snapshot)
	if len(got.Functions) == 0 || len(got.Clones) == 0 {
		t.Fatal("fixture lacks mutable profile collections")
	}
	got.Functions[0].Name = "caller mutation"
	got.Clones[0].Functions[0] = "caller mutation"
	got.Impact.Identity = "caller policy"
	assertProfile()
	again, reused, err := cache.Generate(snapshot, []repoanalysis.BuildSelection{selection}, nil)
	if err != nil || !reused || again.ID != manifest.ID {
		t.Fatalf("exact manifest reuse: %t %v", reused, err)
	}
	// These inputs change manifest authority, not the syntax-only profile.
	selection.Context = "windows/amd64"
	if changed, reused, err := cache.Generate(snapshot, []repoanalysis.BuildSelection{selection}, nil); err != nil || reused || changed.ID == manifest.ID {
		t.Fatalf("context binding: %t %v", reused, err)
	}
	assertProfile()
	inputs := []ExternalInput{{Path: "architecture_profiles.json", ContentID: fixtureDigest, Kind: "architecture-profiles", Owner: "internal/modelrecipe"}}
	if _, reused, err := cache.Generate(snapshot, []repoanalysis.BuildSelection{selection}, inputs); err != nil || reused {
		t.Fatalf("external input binding: %t %v", reused, err)
	}
	assertProfile()
	classified := repoanalysis.SourceSnapshot{Files: slices.Clone(snapshot.Files)}
	classified.Files[0].Test = true
	if _, found := cache.Profile(classified); found {
		t.Fatal("changed file classification reused profile")
	}
	changed, err := snapshot.Overlay(map[string][]byte{name: []byte("package example\nfunc A() int { return 2 }\n")})
	if err != nil {
		t.Fatal(err)
	}
	if _, found := cache.Profile(changed); found {
		t.Fatal("changed source reused profile")
	}
	broken, err := snapshot.Overlay(map[string][]byte{name: []byte("package example\nfunc")})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := cache.Generate(broken, []repoanalysis.BuildSelection{selection}, nil); err == nil {
		t.Fatal("invalid source accepted")
	}
	if _, found := cache.Profile(broken); found {
		t.Fatal("failed computation retained profile")
	}
	assertProfile()
	if _, _, err := cache.Generate(changed, []repoanalysis.BuildSelection{selection}, nil); err != nil {
		t.Fatal(err)
	}
	if _, found := cache.Profile(snapshot); found {
		t.Fatal("profile outlived evicted manifest")
	}
	if _, found := cache.Profile(changed); !found {
		t.Fatal("new profile missing")
	}
}
