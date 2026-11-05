package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

const declaredManuals = `{"manuals":[{"version":1,"name":"repo.search",` +
	`"description":"Search the repository for a pattern.","effect":"inspection",` +
	`"arguments":[{"name":"pattern","kind":"string","required":true}],` +
	`"transport":{"kind":"builtin"}},{"version":1,"name":"go.os",` +
	`"description":"Report the Go host operating system.","effect":"inspection",` +
	`"transport":{"kind":"argv","program":"go","args":["env","GOOS"]}}]}`

func TestAgentToolPublishInspectResolve(t *testing.T) {
	repository := t.TempDir()
	manuals := filepath.Join(t.TempDir(), "manuals.json")
	testcheck.NoError(t, os.WriteFile(manuals, []byte(declaredManuals), 0o600))
	var published bytes.Buffer
	if err := run([]string{"-repo", repository, "-manuals", manuals}, &published); err == nil {
		t.Fatal("argv manual published without an allowlist entry")
	}
	testcheck.NoError(t, run([]string{"-repo", repository, "-manuals", manuals, "-argv-allow", "go"}, &published))
	if !strings.Contains(published.String(), "registered=4 published=4 changed=true") {
		t.Fatalf("publish output = %q", published.String())
	}
	var inspected bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-manuals", manuals, "-argv-allow", "go", "-inspect"}, &inspected))
	if !strings.Contains(inspected.String(), "complete=true") {
		t.Fatalf("inspect output = %q", inspected.String())
	}
	var resolved bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-resolve", "repo.search"}, &resolved))
	if !strings.Contains(resolved.String(), `"repo.search"`) {
		t.Fatalf("resolve output = %q", resolved.String())
	}
	if err := run([]string{"-repo", repository, "-resolve", "repo.absent"}, &resolved); err == nil {
		t.Fatal("unregistered tool resolved")
	}
}

func TestAgentToolInvokesStandardBuiltin(t *testing.T) {
	repository := t.TempDir()
	var published bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-manuals", ""}, &published))
	var invoked bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-invoke", "store.head"}, &invoked))
	if !strings.Contains(invoked.String(), `"sequence"`) {
		t.Fatalf("invoke output = %q", invoked.String())
	}
	testcheck.NoError(t, run([]string{"-repo", repository, "-invoke", "store.alias", "-arguments", `{"name":"tool.registered.store.head"}`}, &invoked))
}

func TestAgentToolCompilesOpenAPICandidatesWithoutRepository(t *testing.T) {
	document := filepath.Join(t.TempDir(), "openapi.json")
	source := `{"openapi":"3.1.0","info":{"title":"Probe","version":"1"},"paths":{"/probe":{"post":{"operationId":"remote.probe","summary":"Probe remote state.","x-overgo-effect":"inspection","responses":{"200":{"description":"ok"}},"requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}}}}}}`
	testcheck.NoError(t, os.WriteFile(document, []byte(source), 0o600))
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-openapi", document, "-base-url", "https://api.example.test/v1"}, &output))
	if !strings.Contains(output.String(), `"source": "file:sha256:`) ||
		!strings.Contains(output.String(), `"name": "remote.probe"`) {
		t.Fatalf("candidate output = %q", output.String())
	}
}

func TestAgentToolStagesVerifiesAndActivatesOpenAPI(t *testing.T) {
	repository := t.TempDir()
	document := filepath.Join(t.TempDir(), "openapi.json")
	source := `{"openapi":"3.1.0","info":{"title":"Probe","version":"1"},"paths":{"/probe":{"post":{"operationId":"remote.probe","summary":"Probe remote state.","x-overgo-effect":"inspection","responses":{"200":{"description":"ok"}},"requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}}}}}}`
	testcheck.NoError(t, os.WriteFile(document, []byte(source), 0o600))
	var staged bytes.Buffer
	if err := run([]string{
		"-repo", repository, "-openapi", document, "-base-url", "https://api.example.test/v1", "-stage",
	}, &staged); err != nil {
		t.Fatal(err)
	}
	var staging struct {
		CandidateID string `json:"candidate_id"`
	}
	if err := json.Unmarshal(staged.Bytes(), &staging); err != nil || staging.CandidateID == "" {
		t.Fatalf("staging = %q, %v", staged.String(), err)
	}
	var verified bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-verify-candidate", staging.CandidateID}, &verified))
	var verification struct {
		EvidenceID string `json:"evidence_id"`
	}
	if err := json.Unmarshal(verified.Bytes(), &verification); err != nil || verification.EvidenceID == "" {
		t.Fatalf("verification = %q, %v", verified.String(), err)
	}
	var activated bytes.Buffer
	if err := run([]string{
		"-repo", repository, "-activate-candidate", staging.CandidateID, "-verification", verification.EvidenceID,
	}, &activated); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(activated.String(), `"complete": true`) {
		t.Fatalf("activation = %q", activated.String())
	}
	var resolved bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-resolve", "remote.probe"}, &resolved))
}
