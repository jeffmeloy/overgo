package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/build/constraint"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/model"
	"overgo/internal/plan"
)

// compatibility.json holds the feature claims, each naming the sources it
// binds and the test that establishes it, and the model prototypes;
// docs/COMPATIBILITY.md is published from it, and a claim's standing is where
// its test last passed, as the trunk's record keeps it (claimStandings).
const (
	manifestPath = "compatibility.json"
	matrixPath   = "docs/COMPATIBILITY.md"
	// manifestSchema is the manifest version this reader decodes.
	manifestSchema = 3
	// commitLength is the length of a full hexadecimal commit name.
	commitLength = 40
	// deviceVerify prefixes the command of a claim whose test runs on the
	// device, which the test's caller enables.
	deviceVerify = "OVERGO_CUDA_TEST=1 "
	// writeCommand writes the matrices again from the claims.
	writeCommand = "go run ./cmd/validate -claims -write"
)

type manifest struct {
	Schema   int                   `json:"schema"`
	Upstream upstream              `json:"upstream"`
	Host     host                  `json:"host"`
	Go       goRuntime             `json:"go"`
	Claims   []claim               `json:"claims"`
	Models   map[string]modelClaim `json:"models"`
}

type upstream struct {
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
}

type host struct {
	OS   string `json:"os"`
	Arch string `json:"arch"`
}

type goRuntime struct {
	Minimum string `json:"minimum"`
	CGO     bool   `json:"cgo"`
}

type claim struct {
	ID               string       `json:"id"`
	EvidenceTier     evidenceTier `json:"evidence_tier"`
	SourceCommit     string       `json:"source_commit,omitzero"`
	ArtifactIdentity string       `json:"artifact_identity,omitzero"`
	Summary          string       `json:"summary"`
	Evidence         []evidence   `json:"evidence"`
}

type evidence struct {
	Path     string       `json:"path"`
	Contains string       `json:"contains"`
	Role     evidenceRole `json:"role"`
}

type evidenceTier string

const (
	tierContract evidenceTier = "contract-tested"
	tierFixture  evidenceTier = "fixture-gated"
	tierOracle   evidenceTier = "pinned-oracle"
	tierDevice   evidenceTier = "device-gated"
)

type evidenceRole string

const (
	roleSource   evidenceRole = "source"
	roleArtifact evidenceRole = "artifact"
)

// modelClaim is what the manifest says of one architecture of the registry that the registry does not: how it runs,
// what it lacks and which real models validate it. What the architecture is, its policies, the registry's profile holds
// (internal/model), and the manifest copies none of it.
type modelClaim struct {
	Execution                  []string `json:"execution"`
	Limitations                []string `json:"limitations"`
	RealModelValidation        any      `json:"real_model_validation"`
	ValidatedFixture           string   `json:"validated_fixture"`
	AdditionalValidatedFixture string   `json:"additional_validated_fixture"`
	MultimodalValidatedFixture string   `json:"multimodal_validated_fixture"`
	VideoValidatedFixture      string   `json:"video_validated_fixture"`
}

// publishedClaims checks the published matrices against the claims they are
// generated from, or with write writes them again. It runs no claim's test:
// a claim's standing is its test's last pass, wherever that ran
// (claimStandings), where a claim whose source moved had been run again in a
// landing's fix stage, under an environment of its own, to rebind it.
func publishedClaims(output io.Writer, root string, write bool) error {
	for path, generator := range map[string]func(string) ([]byte, error){matrixPath: generate, trainingMatrixPath: generateTraining} {
		data, err := generator(root)
		if err != nil {
			return err
		}
		if err := clioptions.OutputGenerated(data, filepath.Join(root, filepath.FromSlash(path)), !write, write,
			path+" is stale; regenerate with: "+writeCommand, output); err != nil {
			return err
		}
	}
	return nil
}

// claimStandings says of each claim where its test last passed, as the trunk
// checkout's record keeps it (plan.ReadTestResults), and which file the claim
// names moved since; a claim whose test has no pass kept says that. A claim
// refuses nothing: its standing is read when a session asks for it.
func claimStandings(ctx context.Context, output io.Writer, root string) error {
	document, err := loadManifest(root)
	if err != nil {
		return err
	}
	passes := plan.ReadTestResults(cmp.Or(trunkCheckout(ctx, root), root))
	for _, item := range slices.SortedFunc(slices.Values(document.Claims), func(left, right claim) int { return strings.Compare(left.ID, right.ID) }) {
		directory, test, ok := item.test()
		if !ok {
			return fmt.Errorf("compatibility manifest: claim %q has no artifact naming a test", item.ID)
		}
		pass, kept := lastPass(passes, directory, test)
		if !kept {
			fmt.Fprintf(output, "claim %s: %s has no pass kept\n", item.ID, test)
			continue
		}
		paths := make([]string, 0, len(item.Evidence))
		for _, proof := range item.Evidence {
			paths = append(paths, proof.Path)
		}
		line := fmt.Sprintf("claim %s: %s passed at %.12s", item.ID, test, pass.Commit)
		moved, err := gitauthority.Query(ctx, root, append([]string{"diff", "--name-only", pass.Commit, "HEAD", "--"}, paths...)...)
		switch {
		case err != nil:
			line += ", a commit this checkout does not hold"
		case len(strings.Fields(string(moved))) > 0:
			line += ", and " + strings.Join(strings.Fields(string(moved)), ", ") + " moved since"
		}
		fmt.Fprintln(output, line)
	}
	return nil
}

// lastPass is the pass kept for test in the package at directory, which the
// record names by its import path (plan.PassedTest); a kept failure is no pass.
func lastPass(results map[string]plan.TestResult, directory, test string) (plan.TestResult, bool) {
	for name, result := range results {
		pkg, named, found := strings.Cut(name, " ")
		if found && named == test && !result.Failed && samePackage("./"+directory, pkg) {
			return result, true
		}
	}
	return plan.TestResult{}, false
}

func generate(root string) ([]byte, error) {
	document, err := loadManifest(root)
	if err != nil {
		return nil, err
	}
	if err := validateManifest(root, document); err != nil {
		return nil, err
	}
	var output strings.Builder
	output.WriteString("# Compatibility matrix\n\n")
	output.WriteString("Generated from [`compatibility.json`](../compatibility.json). Do not edit this file directly.\n\n")
	fmt.Fprintf(&output, "Baseline: `%s` at `%s`; host `%s/%s`; Go `%s`; cgo `%t`.\n\n",
		document.Upstream.Repository, document.Upstream.Commit, document.Host.OS,
		document.Host.Arch, document.Go.Minimum, document.Go.CGO)
	output.WriteString("## Verified feature claims\n\n")
	output.WriteString("Tier describes evidence strength; each claim's command is the one its test derives, and `go run ./cmd/validate -claims` says where each test last passed.\n\n")
	output.WriteString("| ID | Tier | Claim | Evidence |\n| --- | --- | --- | --- |\n")
	claims := slices.SortedFunc(slices.Values(document.Claims), func(left, right claim) int { return strings.Compare(left.ID, right.ID) })
	for _, item := range claims {
		links := make([]string, 0, len(item.Evidence)+1)
		for _, proof := range item.Evidence {
			links = append(links, fmt.Sprintf("%s: [`%s`](../%s)", proof.Role, proof.Contains, filepath.ToSlash(proof.Path)))
		}
		command, err := item.command(root)
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(&output, "| `%s` | %s | %s | %s |\n",
			item.ID, item.EvidenceTier, escapeCell(item.Summary), strings.Join(append(links, "verify: `"+item.shown(command)+"`"), "<br>"))
	}
	output.WriteString("\n## Model prototypes\n\n")
	output.WriteString("A prototype is structural architecture support guarded by strict metadata/catalog validation. Validation is a property of specific models, never of the prototype: the real-model validation column names the exact model artifacts whose oracle or fixture evidence stands, and the typed specifications under `docs/verification` carry each specific model's inference and training verification.\n\n")
	output.WriteString("| Model | Execution | Real-model validation | Limitations |\n| --- | --- | --- | --- |\n")
	for _, name := range slices.Sorted(maps.Keys(document.Models)) {
		item := document.Models[name]
		execution := cmp.Or(strings.Join(item.Execution, ", "), "model-dependent")
		limitations := cmp.Or(strings.Join(item.Limitations, ", "), "-")
		fmt.Fprintf(&output, "| `%s` | %s | %s | %s |\n",
			name, escapeCell(execution), escapeCell(modelValidation(item)), escapeCell(limitations))
	}
	return []byte(output.String()), nil
}

func loadManifest(root string) (manifest, error) {
	data, err := os.ReadFile(filepath.Join(root, manifestPath))
	if err != nil {
		return manifest{}, err
	}
	return decodeManifest(data)
}

func decodeManifest(data []byte) (manifest, error) {
	var document manifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(&document); err != nil {
		return manifest{}, fmt.Errorf("compatibility manifest: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return manifest{}, errors.New("compatibility manifest: multiple JSON documents")
	} else if !errors.Is(err, io.EOF) {
		return manifest{}, fmt.Errorf("compatibility manifest: trailing data: %w", err)
	}
	return document, nil
}

func validateManifest(root string, document manifest) error {
	if document.Schema != manifestSchema || document.Upstream.Repository == "" || len(document.Upstream.Commit) != commitLength ||
		document.Host.OS == "" || document.Host.Arch == "" || document.Go.Minimum == "" {
		return errors.New("compatibility manifest: baseline is incomplete")
	}
	if len(document.Claims) == 0 || len(document.Models) == 0 {
		return errors.New("compatibility manifest: claims or models are empty")
	}
	seen := make(map[string]struct{}, len(document.Claims))
	for index, item := range document.Claims {
		if item.ID == "" || item.Summary == "" || !item.EvidenceTier.valid() {
			return fmt.Errorf("compatibility manifest: claim %d is incomplete", index)
		}
		if _, exists := seen[item.ID]; exists {
			return fmt.Errorf("compatibility manifest: duplicate claim %q", item.ID)
		}
		seen[item.ID] = struct{}{}
		hasSource, hasArtifact := false, false
		for _, proof := range item.Evidence {
			if err := evidenceHolds(root, item.ID, proof); err != nil {
				return err
			}
			hasSource = hasSource || proof.Role == roleSource
			hasArtifact = hasArtifact || proof.Role == roleArtifact
		}
		if !hasSource || !hasArtifact {
			return fmt.Errorf("compatibility manifest: claim %q requires source and artifact evidence", item.ID)
		}
		if err := validateClaimEvidenceTier(item); err != nil {
			return fmt.Errorf("compatibility manifest: claim %q: %w", item.ID, err)
		}
	}
	for name := range document.Models {
		if name == "" {
			return errors.New("compatibility manifest: a model has no name")
		}
	}
	if _, err := os.Stat(filepath.Join(root, "internal", "model", "architecture.go")); err == nil {
		if err := validateModelCoverage(document.Models, model.SupportedArchitectures()); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("compatibility manifest: inspect architecture registry: %w", err)
	}
	return nil
}

func validateModelCoverage(models map[string]modelClaim, supported []string) error {
	missing := make([]string, 0)
	for _, name := range supported {
		if _, ok := models[name]; !ok {
			missing = append(missing, name)
		}
	}
	extra := make([]string, 0)
	for name := range models {
		if !slices.Contains(supported, name) {
			extra = append(extra, name)
		}
	}
	slices.Sort(missing)
	slices.Sort(extra)
	if len(missing) > 0 || len(extra) > 0 {
		return fmt.Errorf("compatibility manifest: model coverage differs: missing=%v extra=%v", missing, extra)
	}
	return nil
}

// evidenceHolds checks one piece of a claim's evidence: a file of the
// checkout that holds its text, and for the test artifact one that names an
// exact, failable go test.
func evidenceHolds(root, claimID string, proof evidence) error {
	clean := filepath.Clean(filepath.FromSlash(proof.Path))
	if proof.Path == "" || proof.Contains == "" || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return fmt.Errorf("compatibility manifest: claim %q has invalid evidence", claimID)
	}
	data, err := os.ReadFile(filepath.Join(root, clean))
	if err != nil {
		return fmt.Errorf("compatibility manifest: claim %q evidence: %w", claimID, err)
	}
	if !bytes.Contains(data, []byte(proof.Contains)) {
		return fmt.Errorf("compatibility manifest: claim %q evidence %q lacks %q", claimID, proof.Path, proof.Contains)
	}
	switch proof.Role {
	case roleSource:
	case roleArtifact:
		if _, ok := evidenceCommand(proof, data); !ok {
			return fmt.Errorf("compatibility manifest: claim %q artifact %q lacks an exact failable Go test command", claimID, proof.Path)
		}
	default:
		return fmt.Errorf("compatibility manifest: claim %q evidence %q has invalid role %q", claimID, proof.Path, proof.Role)
	}
	return nil
}

func (tier evidenceTier) valid() bool {
	return tier == tierContract || tier == tierFixture || tier == tierOracle || tier == tierDevice
}

func validateClaimEvidenceTier(item claim) error {
	if item.EvidenceTier == tierOracle && len(item.SourceCommit) != commitLength {
		return errors.New("pinned-oracle evidence needs a source_commit")
	}
	if (item.EvidenceTier == tierFixture || item.EvidenceTier == tierDevice) && item.ArtifactIdentity == "" {
		return fmt.Errorf("%s evidence needs an artifact_identity", item.EvidenceTier)
	}
	return nil
}

// command is the test command a claim's first artifact derives (evidenceCommand); the claim records none of its own,
// so no copy of it can name another test.
func (item claim) command(root string) (string, error) {
	for _, proof := range item.Evidence {
		if proof.Role != roleArtifact {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(proof.Path)))
		if err != nil {
			return "", err
		}
		if command, ok := evidenceCommand(proof, data); ok {
			return command, nil
		}
	}
	return "", fmt.Errorf("compatibility manifest: claim %q has no artifact naming an uncached, verbose test", item.ID)
}

// test is the directory and name of the test a claim's first artifact names.
func (item claim) test() (directory, name string, ok bool) {
	for _, proof := range item.Evidence {
		if directory, name, ok := artifactTest(proof); ok {
			return directory, name, true
		}
	}
	return "", "", false
}

// shown is command as a person runs it: a device claim's with the switch that enables the device before it.
func (item claim) shown(command string) string {
	if item.EvidenceTier == tierDevice {
		return deviceVerify + command
	}
	return command
}

// artifactTest is the directory, slash-separated, and the name of the test an artifact's text declares.
func artifactTest(proof evidence) (directory, name string, ok bool) {
	if proof.Role != roleArtifact || !strings.HasSuffix(proof.Path, "_test.go") {
		return "", "", false
	}
	name, ok = strings.CutPrefix(proof.Contains, "func ")
	if !ok || !strings.HasPrefix(name, "Test") {
		return "", "", false
	}
	name, _, ok = strings.Cut(name, "(")
	if !ok || name == "Test" || strings.ContainsAny(name, " \\/") {
		return "", "", false
	}
	return filepath.ToSlash(filepath.Dir(proof.Path)), name, true
}

func evidenceCommand(proof evidence, data []byte) (string, bool) {
	directory, name, ok := artifactTest(proof)
	if !ok {
		return "", false
	}
	target := "./" + directory
	if directory == "." {
		target = "."
	}
	tags := ""
	if named := cycleTagsOf(data); len(named) > 0 {
		tags = " -tags " + strings.Join(named, ",")
	}
	return fmt.Sprintf("go test%s %s -run '^%s$' -count=1 -v", tags, target, name), true
}

// cycleTagsOf are the improvement cycle's build tags, integration and modeltest, that a test file's //go:build line names
// wherever it names them ("windows && modeltest" among them): a test behind one runs only when its command names it.
func cycleTagsOf(data []byte) []string {
	for line := range strings.Lines(string(data)) {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "package ") {
			return nil
		}
		if !constraint.IsGoBuild(line) {
			continue
		}
		expression, err := constraint.Parse(line)
		if err != nil {
			return nil
		}
		named := map[string]bool{}
		expression.Eval(func(tag string) bool {
			named[tag] = true
			return true
		})
		return slices.DeleteFunc([]string{"integration", "modeltest"}, func(tag string) bool { return !named[tag] })
	}
	return nil
}

func modelValidation(item modelClaim) string {
	modelFixtures := slices.DeleteFunc([]string{item.ValidatedFixture, item.AdditionalValidatedFixture}, func(value string) bool { return value == "" })
	var parts []string
	if len(modelFixtures) != 0 {
		parts = append(parts, "validated: "+strings.Join(modelFixtures, ", "))
	}
	if item.MultimodalValidatedFixture != "" {
		parts = append(parts, "multimodal: "+item.MultimodalValidatedFixture)
	}
	if item.VideoValidatedFixture != "" {
		parts = append(parts, "video: "+item.VideoValidatedFixture)
	}
	if len(modelFixtures) == 0 {
		validation := "pending-fixture"
		if item.RealModelValidation != nil {
			validation = fmt.Sprint(item.RealModelValidation)
		}
		parts = append(parts, validation)
	}
	return strings.Join(parts, "; ")
}

func escapeCell(value string) string {
	value = strings.ReplaceAll(value, "|", "\\|")
	value = strings.ReplaceAll(value, "\n", " ")
	return value
}
