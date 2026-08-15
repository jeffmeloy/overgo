package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/model"
)

const (
	manifestPath = "compatibility.json"
	matrixPath   = "docs/COMPATIBILITY.md"
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
	SourceCommit     string       `json:"source_commit,omitempty"`
	ArtifactIdentity string       `json:"artifact_identity,omitempty"`
	Summary          string       `json:"summary"`
	Evidence         []evidence   `json:"evidence"`
}

type evidence struct {
	Path     string       `json:"path"`
	Contains string       `json:"contains"`
	Symbol   string       `json:"symbol,omitempty"`
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
	Status                     string   `json:"status"`
	Execution                  []string `json:"execution"`
	Features                   []string `json:"features"`
	Limitations                []string `json:"limitations"`
	RealModelValidation        any      `json:"real_model_validation"`
	ValidatedFixture           string   `json:"validated_fixture"`
	AdditionalValidatedFixture string   `json:"additional_validated_fixture"`
	MultimodalValidatedFixture string   `json:"multimodal_validated_fixture"`
	VideoValidatedFixture      string   `json:"video_validated_fixture"`
}

func main() {
	clioptions.Main(run)
}

func run() error {
	check := flag.Bool("check", false, "verify compatibility claims and generated matrix")
	update := flag.Bool("update", false, "write generated compatibility matrix")
	refresh := flag.Bool("refresh-identities", false, "refresh evidence identities and generated matrix")
	flag.Parse()
	if flag.NArg() != 0 || *check && (*update || *refresh) {
		return errors.New("usage: compatibility [-check|-update|-refresh-identities]")
	}
	if *refresh {
		if err := refreshEvidenceIdentities("."); err != nil {
			return err
		}
	}
	data, err := generate(".")
	if err != nil {
		return err
	}
	return clioptions.OutputGenerated(
		data, matrixPath, *check, *update || *refresh,
		"docs/COMPATIBILITY.md is stale; regenerate with: go run ./cmd/compatibility -update",
		os.Stdout,
	)
}

func refreshEvidenceIdentities(root string) error {
	document, err := loadManifest(root)
	if err != nil {
		return err
	}
	manifestFile := filepath.Join(root, manifestPath)
	raw, err := os.ReadFile(manifestFile)
	if err != nil {
		return err
	}
	cursor := 0
	for i := range document.Claims {
		for j := range document.Claims[i].Evidence {
			proof := &document.Claims[i].Evidence[j]
			data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(proof.Path)))
			if err != nil {
				return err
			}
			payload, err := evidencePayload(*proof, data)
			if err != nil {
				return err
			}
			kind := artifact.KindFile
			if proof.Role == roleArtifact {
				kind = artifact.KindEvidence
			}
			id, err := artifact.IdentifyBytes(kind, payload)
			if err != nil {
				return err
			}
			before, after := []byte(proof.Identity), []byte(id.String())
			offset := bytes.Index(raw[cursor:], before)
			if offset < 0 {
				return fmt.Errorf("compatibility manifest: identity %q not found", proof.Identity)
			}
			start := cursor + offset
			if len(before) != len(after) {
				return fmt.Errorf("compatibility manifest: identity length changed")
			}
			copy(raw[start:], after)
			cursor = start + len(after)
			if !bytes.Equal(before, after) && bytes.HasPrefix(raw[cursor:], []byte("\"\r\n")) {
				raw = append(raw[:cursor+1], raw[cursor+2:]...)
			}
		}
	}
	return os.WriteFile(manifestFile, raw, 0o644)
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
	claims := append([]claim(nil), document.Claims...)
	sort.Slice(claims, func(i, j int) bool { return claims[i].ID < claims[j].ID })
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
	output.WriteString("\n## Model families\n\n")
	output.WriteString("`experimental` means the implementation is guarded by strict metadata/catalog validation but may still lack a local real-model oracle.\n\n")
	output.WriteString("| Model | State | Execution | Real-model validation | Limitations |\n| --- | --- | --- | --- | --- |\n")
	names := make([]string, 0, len(document.Models))
	for name := range document.Models {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		item := document.Models[name]
		execution := strings.Join(item.Execution, ", ")
		if execution == "" {
			execution = "model-dependent"
		}
		validation := modelValidation(item)
		limitations := strings.Join(item.Limitations, ", ")
		if limitations == "" {
			limitations = "-"
		}
		fmt.Fprintf(&output, "| `%s` | %s | %s | %s | %s |\n",
			name, item.Status, escapeCell(execution), escapeCell(validation), escapeCell(limitations))
	}
	return []byte(output.String()), nil
}

func loadManifest(root string) (manifest, error) {
	data, err := os.ReadFile(filepath.Join(root, manifestPath))
	if err != nil {
		return manifest{}, err
	}
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
	if document.Schema != 2 || document.Upstream.Repository == "" || len(document.Upstream.Commit) != 40 ||
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
			if err := validateEvidence(root, item.ID, proof); err != nil {
				return err
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
		if name == "" || item.Status == "" || len(item.Features) == 0 {
			return fmt.Errorf("compatibility manifest: model %q is incomplete", name)
		}
		// Evidence tier: "implemented" claims oracle-backed behavior, not
		// structural coverage. A model without real-model validation or a
		// pinned fixture is honestly "experimental" -- breadth must not
		// inherit evidence it does not carry.
		if item.Status == "implemented" && !modelHasValidation(item) {
			return fmt.Errorf("compatibility manifest: model %q claims implemented without real_model_validation or a validated fixture; use experimental until oracle evidence exists", name)
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

func modelHasValidation(item modelClaim) bool {
	if item.ValidatedFixture != "" || item.AdditionalValidatedFixture != "" ||
		item.MultimodalValidatedFixture != "" || item.VideoValidatedFixture != "" {
		return true
	}
	switch v := item.RealModelValidation.(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(v) != "" && !strings.EqualFold(strings.TrimSpace(v), "none")
	case bool:
		return v
	default:
		return true
	}
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
	sort.Strings(missing)
	sort.Strings(extra)
	if len(missing) > 0 || len(extra) > 0 {
		return fmt.Errorf("compatibility manifest: model coverage differs: missing=%v extra=%v", missing, extra)
	}
	return nil
}

func validateEvidence(root, claimID string, proof evidence) error {
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
	kind := artifact.KindFile
	switch proof.Role {
	case roleSource:
	case roleArtifact:
		kind = artifact.KindEvidence
		if _, ok := evidenceCommand(proof, data); !ok {
			return fmt.Errorf("compatibility manifest: claim %q artifact %q lacks an exact failable Go test command", claimID, proof.Path)
		}
	default:
		return fmt.Errorf("compatibility manifest: claim %q evidence %q has invalid role %q", claimID, proof.Path, proof.Role)
	}
	payload, err := evidencePayload(proof, data)
	if err != nil {
		return fmt.Errorf("compatibility manifest: claim %q evidence %q: %w", claimID, proof.Path, err)
	}
	want, err := artifact.IdentifyBytes(kind, payload)
	if err != nil {
		return err
	}
	if proof.Identity != want.String() {
		return fmt.Errorf("compatibility manifest: claim %q evidence %q is stale: have %q want %q", claimID, proof.Path, proof.Identity, want)
	}
	return nil
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
	if item.EvidenceTier == tierOracle && len(item.SourceCommit) != 40 {
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
			if item.Verify == command || item.EvidenceTier == tierDevice && item.Verify == "OVERGO_CUDA_TEST=1 "+command {
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

func scalarText(value any) string {
	if value == nil {
		return "pending-fixture"
	}
	return fmt.Sprint(value)
}

func modelValidation(item modelClaim) string {
	modelFixtures := []string{
		item.ValidatedFixture,
		item.AdditionalValidatedFixture,
	}
	modelFixtures = slices.DeleteFunc(modelFixtures, func(value string) bool { return value == "" })
	parts := make([]string, 0, 4)
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
		parts = append(parts, scalarText(item.RealModelValidation))
	}
	return strings.Join(parts, "; ")
}

func escapeCell(value string) string {
	value = strings.ReplaceAll(value, "|", "\\|")
	value = strings.ReplaceAll(value, "\n", " ")
	return value
}
