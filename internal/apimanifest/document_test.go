package apimanifest

import (
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

const fixtureDigest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

var fixtureContract = ContractRef{
	Kind: artifact.KindOutput, MediaType: artifact.JSONMediaType, Schema: "overgo/fixture/v1",
}

func fixtureManifest(t *testing.T) Manifest {
	t.Helper()
	value, err := New(Manifest{
		Version: artifact.InitialDocumentVersion, Release: "fixture", SourceIdentity: fixtureDigest,
		BuildContexts: []BuildContext{{ID: "windows-amd64", GOOS: "windows", GOARCH: "amd64"}},
		Documents: []Document{{
			Name: "fixture", Owner: "internal/fixture", Kind: fixtureContract.Kind,
			MediaType: fixtureContract.MediaType, Schema: fixtureContract.Schema,
		}},
		Binaries:    []Binary{{Name: "overgo", Package: "cmd/overgo", BuildContexts: []string{"windows-amd64"}, Output: []ContractRef{fixtureContract}}},
		Routes:      []Route{{Path: "/fixture", Method: "GET", Authentication: "public", Responses: []ContractRef{fixtureContract}}},
		Modules:     []Module{{ID: "fixture.module", Owner: "internal/fixture", Tasks: []string{"inference"}, Outputs: []ContractRef{fixtureContract}}},
		Protocols:   []Protocol{{Name: "fixture", Owner: "internal/fixture", Transport: "http", Contracts: []ContractRef{fixtureContract}}},
		Workspaces:  []Workspace{{Name: "fixture", Path: "internal/server/workspace_manifest.json", ContentID: fixtureDigest}},
		Authorities: []Authority{{Name: "kernel", Path: "kernels/manifest.json", ContentID: fixtureDigest}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestManifestCanonicalRoundTrip(t *testing.T) {
	manifest := fixtureManifest(t)
	content, err := manifest.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(content.Data)
	if err != nil || parsed.ID != manifest.ID || !reflect.DeepEqual(parsed.Routes, manifest.Routes) {
		t.Fatalf("round trip = %+v/%v", parsed, err)
	}
}

func TestManifestRejectsUnknownContract(t *testing.T) {
	manifest := fixtureManifest(t)
	manifest.ID = artifact.ID{}
	manifest.Routes[0].Responses[0].Schema = "overgo/absent/v1"
	if _, err := New(manifest); err == nil || !strings.Contains(err.Error(), "route") {
		t.Fatalf("unknown contract error = %v", err)
	}
}

func TestCompareReportsCurrentSurfaceChangesWithoutTombstones(t *testing.T) {
	previous := fixtureManifest(t)
	current := fixtureManifest(t)
	current.ID = artifact.ID{}
	current.Routes = append(current.Routes, Route{
		Path: "/new", Method: "POST", Authentication: "key", Request: &fixtureContract,
	})
	current, err := New(current)
	if err != nil {
		t.Fatal(err)
	}
	changes, err := Compare(previous, current)
	if err != nil || len(changes) != 1 || changes[0] != (Change{Class: ChangeAdd, Kind: "route", Key: "POST /new"}) {
		t.Fatalf("changes = %+v/%v", changes, err)
	}
	if len(current.Routes) != len(previous.Routes)+1 {
		t.Fatal("current manifest retained hidden history")
	}
}
