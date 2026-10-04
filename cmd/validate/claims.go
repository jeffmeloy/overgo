package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/model"
	"overgo/internal/testevidence"
)

// compatibility.json holds the feature claims, each bound by identity to the
// sources and the test artifact that establish it, and the model prototypes;
// docs/COMPATIBILITY.md is published from it.
const (
	manifestPath = "compatibility.json"
	matrixPath   = "docs/COMPATIBILITY.md"
	// manifestSchema is the manifest version this reader decodes.
	manifestSchema = 2
	// commitLength is the length of a full hexadecimal commit name.
	commitLength = 40
	// deviceVerify prefixes the verify of a claim whose test runs on the
	// device, which the test's caller enables.
	deviceVerify = "OVERGO_CUDA_TEST=1 "
	// reverifyCommand is how a moved claim is re-verified.
	reverifyCommand = "go run ./cmd/validate -claims -run"
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
	Status           string       `json:"status"`
	EvidenceTier     evidenceTier `json:"evidence_tier"`
	Verify           string       `json:"verify"`
	SourceCommit     string       `json:"source_commit,omitzero"`
	ArtifactIdentity string       `json:"artifact_identity,omitzero"`
	Summary          string       `json:"summary"`
	Evidence         []evidence   `json:"evidence"`
}

type evidence struct {
	Path     string       `json:"path"`
	Contains string       `json:"contains"`
	Symbol   string       `json:"symbol,omitzero"`
	Role     evidenceRole `json:"role"`
	Identity string       `json:"identity"`
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

type modelClaim struct {
	Execution                  []string `json:"execution"`
	Features                   []string `json:"features"`
	Limitations                []string `json:"limitations"`
	RealModelValidation        any      `json:"real_model_validation"`
	ValidatedFixture           string   `json:"validated_fixture"`
	AdditionalValidatedFixture string   `json:"additional_validated_fixture"`
	MultimodalValidatedFixture string   `json:"multimodal_validated_fixture"`
	VideoValidatedFixture      string   `json:"video_validated_fixture"`
}

// publishedClaims checks the published matrices against the claims they are
// generated from. With execute it first re-verifies every claim whose bound
// source moved, then writes the matrices again: a claim is rebound only by
// its own verify passing, never by hand.
func publishedClaims(ctx context.Context, output io.Writer, root string, execute bool) error {
	if execute {
		if err := reverifyClaims(ctx, output, root); err != nil {
			return err
		}
	}
	for path, generator := range map[string]func(string) ([]byte, error){matrixPath: generate, trainingMatrixPath: generateTraining} {
		data, err := generator(root)
		if err != nil {
			return err
		}
		if err := clioptions.OutputGenerated(data, filepath.Join(root, filepath.FromSlash(path)), !execute, execute,
			path+" is stale; regenerate with: "+reverifyCommand, output); err != nil {
			return err
		}
	}
	return nil
}

// reverifyClaims re-runs the verify test of every claim one of whose bound
// sources moved, at this tree, and rebinds each claim whose test passed to
// the identities its inputs had when the test started. The pass is the typed
// result testevidence reads from the test's events in its package, never its
// log text. Claims whose verify is the same command share one run of it and
// its result: each one's inputs are read before that run and must still have
// those identities when it is rebound. A claim whose test failed or skipped,
// or whose inputs or manifest entry moved before its rebinding, keeps its old
// binding and its reason, so the matrix stays refused until it passes; a
// device claim passes only when the caller enables the device its verify
// names.
func reverifyClaims(ctx context.Context, output io.Writer, root string) error {
	moved, err := movedClaims(root)
	if err != nil {
		return err
	}
	var commands []string
	sharing := map[string][]claim{}
	for _, item := range moved {
		command := strings.TrimPrefix(item.Verify, deviceVerify)
		if _, seen := sharing[command]; !seen {
			commands = append(commands, command)
		}
		sharing[command] = append(sharing[command], item)
	}
	var refused []string
	for _, command := range commands {
		items := sharing[command]
		tested := make([][]string, len(items))
		for at, item := range items {
			tested[at] = make([]string, len(item.Evidence))
			for index, proof := range item.Evidence {
				if tested[at][index], err = evidenceIdentity(root, item.ID, proof); err != nil {
					return err
				}
			}
			fmt.Fprintf(output, "validate -claims -run: re-verifying %s: %s\n", item.ID, item.Verify)
		}
		fields := strings.Fields(command)
		for index, field := range fields {
			fields[index] = strings.Trim(field, "'")
		}
		var events bytes.Buffer
		ran := runCell(ctx, &events, root, fields[0], append(fields[1:], "-json"))
		verdict := cmp.Or(testevidence.VerifyGoTestEvidence(command, events.String()), ran)
		for at, item := range items {
			err := verdict
			if err == nil {
				err = rebindClaim(root, item, tested[at])
			}
			if err != nil {
				refused = append(refused, fmt.Sprintf("%s (%v)", item.ID, err))
			}
		}
	}
	if len(refused) != 0 {
		return fmt.Errorf("validate -claims -run: %d claim(s) keep their old binding: %s", len(refused), strings.Join(refused, "; "))
	}
	return nil
}

// movedClaims are the claims one of whose bound sources no longer has the
// identity the claim records, each with a verify that exactly names its test.
func movedClaims(root string) ([]claim, error) {
	document, err := loadManifest(root)
	if err != nil {
		return nil, err
	}
	var moved []claim
	for _, item := range document.Claims {
		for _, proof := range item.Evidence {
			current, err := evidenceIdentity(root, item.ID, proof)
			if err != nil {
				return nil, err
			}
			if current != proof.Identity {
				if err := validateClaimEvidenceTier(root, item); err != nil {
					return nil, fmt.Errorf("compatibility manifest: claim %q: %w", item.ID, err)
				}
				moved = append(moved, item)
				break
			}
		}
	}
	return moved, nil
}

// rebindClaim writes a passing claim's tested identities, the ones its
// inputs had when its verify started, into the manifest as it stands now:
// only that claim's identities change, so edits made meanwhile elsewhere in
// the manifest survive. It refuses when the claim's manifest entry or an
// input it binds moved since its verify started, or when the manifest moves
// between this read and the write.
func rebindClaim(root string, tested claim, identities []string) error {
	manifestFile := filepath.Join(root, manifestPath)
	raw, err := os.ReadFile(manifestFile)
	if err != nil {
		return err
	}
	document, err := decodeManifest(raw)
	if err != nil {
		return err
	}
	if index := slices.IndexFunc(document.Claims, func(item claim) bool { return item.ID == tested.ID }); index < 0 || !reflect.DeepEqual(document.Claims[index], tested) {
		return errors.New("its manifest entry changed during its verify")
	}
	rebound := slices.Clone(raw)
	cursor := 0
	for _, item := range document.Claims {
		for index, proof := range item.Evidence {
			current := proof.Identity
			if item.ID == tested.ID {
				if current, err = evidenceIdentity(root, item.ID, proof); err != nil || current != identities[index] {
					return cmp.Or(err, fmt.Errorf("%s changed during its verify", proof.Path))
				}
			}
			before, after := []byte(proof.Identity), []byte(current)
			offset := bytes.Index(rebound[cursor:], before)
			if offset < 0 {
				return fmt.Errorf("compatibility manifest: identity %q not found", proof.Identity)
			}
			start := cursor + offset
			if len(before) != len(after) {
				return errors.New("compatibility manifest: identity length changed")
			}
			copy(rebound[start:], after)
			cursor = start + len(after)
			// A rebound identity's line ends as the rest of the file does:
			// the carriage return after its closing quote goes.
			if !bytes.Equal(before, after) && bytes.HasPrefix(rebound[cursor:], []byte("\"\r\n")) {
				rebound = append(rebound[:cursor+len(`"`)], rebound[cursor+len("\"\r"):]...)
			}
		}
	}
	if now, err := os.ReadFile(manifestFile); err != nil || !bytes.Equal(now, raw) {
		return cmp.Or(err, errors.New("the manifest changed while the claim was rebound"))
	}
	return clioptions.WriteOutputFile(manifestFile, rebound)
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
	output.WriteString("State describes implementation completeness; tier describes evidence strength. `implemented` does not imply `oracle`.\n\n")
	output.WriteString("| ID | State | Tier | Claim | Evidence |\n| --- | --- | --- | --- | --- |\n")
	claims := slices.SortedFunc(slices.Values(document.Claims), func(left, right claim) int { return strings.Compare(left.ID, right.ID) })
	for _, item := range claims {
		links := make([]string, 0, len(item.Evidence)+1)
		for _, proof := range item.Evidence {
			label := proof.Contains
			if proof.Symbol != "" {
				label = "symbol " + proof.Symbol
			}
			links = append(links, fmt.Sprintf("%s: [`%s`](../%s) `%s`", proof.Role, label, filepath.ToSlash(proof.Path), proof.Identity))
		}
		fmt.Fprintf(&output, "| `%s` | %s | %s | %s | %s |\n",
			item.ID, item.Status, item.EvidenceTier, escapeCell(item.Summary), strings.Join(append(links, "verify: `"+item.Verify+"`"), "<br>"))
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
		if item.ID == "" || item.Summary == "" || !item.EvidenceTier.valid() || (item.Status != "implemented" && item.Status != "partial" && item.Status != "blocked") {
			return fmt.Errorf("compatibility manifest: claim %d is incomplete", index)
		}
		if _, exists := seen[item.ID]; exists {
			return fmt.Errorf("compatibility manifest: duplicate claim %q", item.ID)
		}
		seen[item.ID] = struct{}{}
		hasSource, hasArtifact := false, false
		for _, proof := range item.Evidence {
			current, err := evidenceIdentity(root, item.ID, proof)
			if err != nil {
				return err
			}
			if proof.Identity != current {
				return fmt.Errorf("compatibility manifest: claim %q evidence %q is stale: have %q want %q; re-verify it with: %s", item.ID, proof.Path, proof.Identity, current, reverifyCommand)
			}
			hasSource = hasSource || proof.Role == roleSource
			hasArtifact = hasArtifact || proof.Role == roleArtifact
		}
		if !hasSource || !hasArtifact {
			return fmt.Errorf("compatibility manifest: claim %q requires source and artifact identities", item.ID)
		}
		if err := validateClaimEvidenceTier(root, item); err != nil {
			return fmt.Errorf("compatibility manifest: claim %q: %w", item.ID, err)
		}
	}
	for name, item := range document.Models {
		if name == "" || len(item.Features) == 0 {
			return fmt.Errorf("compatibility manifest: model %q is incomplete", name)
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

// evidenceIdentity checks one piece of a claim's evidence and returns the
// identity its bound bytes have now.
func evidenceIdentity(root, claimID string, proof evidence) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(proof.Path))
	if proof.Path == "" || proof.Contains == "" || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("compatibility manifest: claim %q has invalid evidence", claimID)
	}
	data, err := os.ReadFile(filepath.Join(root, clean))
	if err != nil {
		return "", fmt.Errorf("compatibility manifest: claim %q evidence: %w", claimID, err)
	}
	if !bytes.Contains(data, []byte(proof.Contains)) {
		return "", fmt.Errorf("compatibility manifest: claim %q evidence %q lacks %q", claimID, proof.Path, proof.Contains)
	}
	kind := artifact.KindFile
	switch proof.Role {
	case roleSource:
	case roleArtifact:
		kind = artifact.KindEvidence
		if _, ok := evidenceCommand(proof, data); !ok {
			return "", fmt.Errorf("compatibility manifest: claim %q artifact %q lacks an exact failable Go test command", claimID, proof.Path)
		}
	default:
		return "", fmt.Errorf("compatibility manifest: claim %q evidence %q has invalid role %q", claimID, proof.Path, proof.Role)
	}
	payload, err := evidencePayload(proof, data)
	if err != nil {
		return "", fmt.Errorf("compatibility manifest: claim %q evidence %q: %w", claimID, proof.Path, err)
	}
	current, err := artifact.IdentifyBytes(kind, payload)
	if err != nil {
		return "", err
	}
	return current.String(), nil
}

func canonicalEvidence(data []byte) []byte {
	return bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
}

func evidencePayload(proof evidence, data []byte) ([]byte, error) {
	if proof.Symbol == "" {
		return canonicalEvidence(data), nil
	}
	if !strings.HasSuffix(proof.Path, ".go") {
		return nil, errors.New("symbol evidence requires a Go file")
	}
	files := token.NewFileSet()
	parsed, err := parser.ParseFile(files, proof.Path, data, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	var matched *ast.FuncDecl
	for _, declaration := range parsed.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Name.Name != proof.Symbol {
			continue
		}
		start := files.Position(function.Pos()).Offset
		end := files.Position(function.End()).Offset
		if start < 0 || end > len(data) || !bytes.Contains(data[start:end], []byte(proof.Contains)) {
			continue
		}
		if matched != nil {
			return nil, fmt.Errorf("symbol %q is ambiguous", proof.Symbol)
		}
		matched = function
	}
	if matched == nil {
		return nil, fmt.Errorf("symbol %q containing %q not found", proof.Symbol, proof.Contains)
	}
	var normalized bytes.Buffer
	if err := format.Node(&normalized, files, matched); err != nil {
		return nil, err
	}
	return canonicalEvidence(normalized.Bytes()), nil
}

func (tier evidenceTier) valid() bool {
	return tier == tierContract || tier == tierFixture || tier == tierOracle || tier == tierDevice
}

func validateClaimEvidenceTier(root string, item claim) error {
	if item.EvidenceTier == tierOracle && len(item.SourceCommit) != commitLength {
		return errors.New("pinned-oracle evidence needs a source_commit")
	}
	if (item.EvidenceTier == tierFixture || item.EvidenceTier == tierDevice) && item.ArtifactIdentity == "" {
		return fmt.Errorf("%s evidence needs an artifact_identity", item.EvidenceTier)
	}
	for _, proof := range item.Evidence {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(proof.Path)))
		if err != nil {
			return err
		}
		if command, ok := evidenceCommand(proof, data); ok {
			if item.Verify == command || item.EvidenceTier == tierDevice && item.Verify == deviceVerify+command {
				return nil
			}
		}
	}
	return errors.New("verify must exactly name an uncached, verbose test artifact")
}

func evidenceCommand(proof evidence, data []byte) (string, bool) {
	if proof.Role != roleArtifact || !strings.HasSuffix(proof.Path, "_test.go") {
		return "", false
	}
	name, ok := strings.CutPrefix(proof.Contains, "func Test")
	if !ok {
		return "", false
	}
	name, _, ok = strings.Cut(name, "(")
	if !ok || name == "" || strings.ContainsAny(name, " \\/") {
		return "", false
	}
	directory := filepath.ToSlash(filepath.Dir(proof.Path))
	target := "./" + directory
	if directory == "." {
		target = "."
	}
	tags := ""
	if bytes.Contains(data, []byte("//go:build integration")) {
		tags = " -tags integration"
	}
	return fmt.Sprintf("go test%s %s -run '^Test%s$' -count=1 -v", tags, target, name), true
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
