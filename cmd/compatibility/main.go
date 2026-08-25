package main

import (
	"bytes"
	"context"
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
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/model"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
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
	checkTraining := flag.Bool("check-training", false, "verify training specifications and generated matrix")
	updateTraining := flag.Bool("update-training", false, "write generated training matrix")
	refresh := flag.Bool("refresh-identities", false, "refresh evidence identities and generated matrix")
	recordVerification := flag.String("record-verification", "", "commit a typed model-verification record from a JSON spec (model, name, evidenced capability claims)")
	recordStore := flag.String("record", "", "OvergoDB root for -record-verification/-claim")
	claimFlag := flag.Bool("claim", false, "build and commit one verification claim from flags: -claim -model-file <weights> -name <n> -capability <c> -tier <t> -evidence-file <doc> [-wall <dur>] [-context <tokens>] [-peak <bytes>] -record <overgodb>")
	claimModelFile := flag.String("model-file", "", "claim: weights file; its digest is the model identity")
	claimName := flag.String("name", "", "claim: human model name")
	claimCapability := flag.String("capability", "inference", "claim: capability (lowercase kebab)")
	claimTier := flag.String("tier", "real-artifact-smoke", "claim: verification tier")
	claimEvidenceFile := flag.String("evidence-file", "", "claim: evidence document; committed by content and referenced by the claim")
	claimWall := flag.Duration("wall", 0, "claim: measured wall")
	claimContext := flag.Uint64("context", 0, "claim: context tokens (requires -wall)")
	claimPeak := flag.Uint64("peak", 0, "claim: peak device bytes (requires -wall)")
	flag.Parse()
	if *checkTraining || *updateTraining {
		if len(flag.Args()) != len([]string(nil)) || *checkTraining && *updateTraining || *check || *update || *refresh || *claimFlag || *recordVerification != "" {
			return errors.New("usage: compatibility [-check-training|-update-training]")
		}
		data, err := generateTraining(".")
		if err != nil {
			return err
		}
		return clioptions.OutputGenerated(
			data, trainingMatrixPath, *checkTraining, *updateTraining,
			"docs/TRAINING_COMPATIBILITY.md is stale; regenerate with: go run ./cmd/compatibility -update-training",
			os.Stdout,
		)
	}
	if *claimFlag {
		if flag.NArg() != 0 || *claimModelFile == "" || *claimName == "" || *claimEvidenceFile == "" || *recordStore == "" {
			return errors.New("usage: compatibility -claim -model-file <weights> -name <n> -evidence-file <doc> -record <overgodb> [-capability c] [-tier t] [-wall d] [-context n] [-peak b]")
		}
		return runClaim(claimInput{
			modelFile: *claimModelFile, name: *claimName, capability: *claimCapability,
			tier: *claimTier, evidenceFile: *claimEvidenceFile,
			wall: *claimWall, context: *claimContext, peak: *claimPeak,
		}, *recordStore, os.Stdout)
	}
	if *recordVerification != "" {
		if flag.NArg() != 0 || *check || *update || *refresh || *recordStore == "" {
			return errors.New("usage: compatibility -record-verification <spec.json> -record <overgodb>")
		}
		return runRecordVerification(*recordVerification, *recordStore, os.Stdout)
	}
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
	if err := clioptions.OutputGenerated(
		data, matrixPath, *check, *update || *refresh,
		"docs/COMPATIBILITY.md is stale; regenerate with: go run ./cmd/compatibility -update",
		os.Stdout,
	); err != nil {
		return err
	}
	training, err := generateTraining(".")
	if err != nil {
		return err
	}
	return clioptions.OutputGenerated(
		training, trainingMatrixPath, *check, *update || *refresh,
		"docs/TRAINING_COMPATIBILITY.md is stale; regenerate with: go run ./cmd/compatibility -update-training",
		os.Stdout,
	)
}

type claimInput struct {
	modelFile, name, capability, tier, evidenceFile string
	wall                                            time.Duration
	context, peak                                   uint64
}

// runClaim builds and commits one verification claim entirely from flags:
// the model identity is the weights-file digest, the evidence document
// commits by content, and the record lands with lineage -- no external
// scripting anywhere in the path.
func runClaim(input claimInput, recordStore string, output io.Writer) error {
	commit, err := verifyingCommit(".")
	if err != nil {
		return err
	}
	weights, err := os.Open(input.modelFile)
	if err != nil {
		return err
	}
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	closeErr := weights.Close()
	if err != nil || closeErr != nil {
		return errors.Join(err, closeErr)
	}
	evidenceData, err := os.ReadFile(input.evidenceFile)
	if err != nil {
		return err
	}
	evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(evidenceData))
	if err != nil {
		return err
	}
	record, err := runrecord.NewModelVerification(model, input.name, []runrecord.CapabilityClaim{{
		Capability: input.capability, Tier: runrecord.VerificationTier(input.tier), Commit: commit,
		ContextTokens: input.context, WallNS: uint64(input.wall.Nanoseconds()), PeakDeviceBytes: input.peak,
		Evidence: []artifact.ID{evidence},
	}})
	if err != nil {
		return err
	}
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	batch, err := record.Batch("verification/" + record.ID.String())
	if err != nil {
		return err
	}
	if _, ok, err := artifact.ReadContent(ctx, store, evidence); err != nil {
		return err
	} else if !ok {
		batch.Contents = append(batch.Contents, artifact.Content{
			Descriptor: artifact.Descriptor{ID: evidence, Size: uint64(len(evidenceData))}, Data: evidenceData,
		})
	}
	if _, ok, err := store.Artifact(ctx, model); err != nil {
		return err
	} else if !ok {
		absolute, err := filepath.Abs(input.modelFile)
		if err != nil {
			return err
		}
		info, err := os.Stat(input.modelFile)
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: model, Size: uint64(info.Size())})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{
			Location: artifact.Location{Artifact: model, Kind: artifact.LocationFile, Value: absolute},
			Action:   artifact.LocationAdd,
		})
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Fprintf(output, "claim committed: %s model=%s name=%s %s=%s commit=%.12s\n",
		record.ID, model, input.name, input.capability, input.tier, commit)
	fmt.Fprintln(output, "honesty: verifier source matched a clean HEAD; the model identity is the weights digest; the claim is grounded in the committed evidence document")
	return nil
}

// verifyingCommit returns HEAD only when tracked and untracked worktree state
// is empty, so a claim cannot identify a commit that differs from executed
// source or fixture bytes.
func verifyingCommit(root string) (string, error) {
	status, err := exec.Command("git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all").Output()
	if err != nil {
		return "", fmt.Errorf("inspect verifying worktree: %w", err)
	}
	if len(status) != 0 {
		return "", errors.New("verifying worktree is dirty; commit or remove every change before recording a claim")
	}
	head, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("resolve verifying commit: %w", err)
	}
	commit := strings.TrimSpace(string(head))
	if !validGitCommit(commit) {
		return "", errors.New("verifying HEAD is not a full Git commit identity")
	}
	return commit, nil
}

func validGitCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	for _, char := range value {
		if char < '0' || char > '9' && char < 'a' || char > 'f' {
			return false
		}
	}
	return true
}

// runRecordVerification commits one typed model-verification record: the
// claims discipline as a store artifact rather than a document -- every
// capability tier grounded in named evidence, with lineage to all of it.
// The spec may ground its references: evidence files commit by content
// (their identity must match a claimed evidence ID) and the model file
// registers by descriptor and location -- both verified against the named
// identities, never invented.
func runRecordVerification(specPath, recordStore string, output io.Writer) error {
	var specification struct {
		Model         artifact.ID                 `json:"model"`
		Name          string                      `json:"name"`
		Supersedes    []artifact.ID               `json:"supersedes,omitempty"`
		ModelFile     string                      `json:"model_file,omitempty"`
		EvidenceFiles []string                    `json:"evidence_files,omitempty"`
		DatasetFiles  []string                    `json:"dataset_files,omitempty"`
		Claims        []runrecord.CapabilityClaim `json:"claims"`
	}
	if err := jsonfile.Decode(specPath, &specification); err != nil {
		return err
	}
	record, err := runrecord.NewModelVerificationCorrection(
		specification.Model, specification.Name, specification.Claims, specification.Supersedes,
	)
	if err != nil {
		return err
	}
	claimed := map[artifact.ID]bool{}
	for _, claim := range record.Claims {
		for _, evidence := range claim.Evidence {
			claimed[evidence] = true
		}
	}
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	for _, superseded := range record.Supersedes {
		content, ok, err := artifact.ReadContent(ctx, store, superseded)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("superseded model verification %s is absent", superseded)
		}
		prior, err := runrecord.ParseModelVerification(content.Data)
		if err != nil {
			return fmt.Errorf("parse superseded model verification %s: %w", superseded, err)
		}
		if prior.Model != record.Model {
			return fmt.Errorf("superseded model verification %s names model %s, correction names %s", superseded, prior.Model, record.Model)
		}
	}
	batch, err := record.Batch("verification/" + record.ID.String())
	if err != nil {
		return err
	}
	datasets := map[artifact.ID]bool{}
	for _, claim := range record.Claims {
		if claim.Dataset.Valid() {
			datasets[claim.Dataset] = true
		}
	}
	ground := func(paths []string, kind artifact.Kind, referenced map[artifact.ID]bool, role string) error {
		for _, path := range paths {
			data, err := os.ReadFile(path)
			if err != nil {
				return fmt.Errorf("%s file %s: %w", role, path, err)
			}
			identity, _, err := artifact.Identify(kind, bytes.NewReader(data))
			if err != nil {
				return err
			}
			if !referenced[identity] {
				return fmt.Errorf("%s file %s identifies as %s, which no claim references", role, path, identity)
			}
			if _, ok, err := artifact.ReadContent(ctx, store, identity); err != nil {
				return err
			} else if !ok {
				batch.Contents = append(batch.Contents, artifact.Content{
					Descriptor: artifact.Descriptor{ID: identity, Size: uint64(len(data))}, Data: data,
				})
			}
		}
		return nil
	}
	if err := ground(specification.EvidenceFiles, artifact.KindEvidence, claimed, "evidence"); err != nil {
		return err
	}
	if err := ground(specification.DatasetFiles, artifact.KindDatasetShard, datasets, "dataset"); err != nil {
		return err
	}
	if _, ok, err := store.Artifact(ctx, record.Model); err != nil {
		return err
	} else if !ok {
		if specification.ModelFile == "" {
			return fmt.Errorf("model %s is not in the store; the spec must name model_file to register it", record.Model)
		}
		file, err := os.Open(specification.ModelFile)
		if err != nil {
			return err
		}
		identity, size, err := artifact.Identify(artifact.KindModel, file)
		closeErr := file.Close()
		if err != nil || closeErr != nil {
			return errors.Join(err, closeErr)
		}
		if identity != record.Model {
			return fmt.Errorf("model file %s identifies as %s, spec claims %s", specification.ModelFile, identity, record.Model)
		}
		absolute, err := filepath.Abs(specification.ModelFile)
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: identity, Size: size})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{
			Location: artifact.Location{Artifact: identity, Kind: artifact.LocationFile, Value: absolute},
			Action:   artifact.LocationAdd,
		})
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	committed, external := 0, 0
	for evidence := range claimed {
		if _, ok, err := artifact.ReadContent(ctx, store, evidence); err != nil {
			return err
		} else if ok {
			committed++
		} else {
			external++
		}
	}
	fmt.Fprintf(output, "model verification committed: %s model=%s name=%s claims=%d\n",
		record.ID, record.Model, record.Name, len(record.Claims))
	for _, claim := range record.Claims {
		provenance := "commit=" + claim.Commit
		if claim.Dataset.Valid() {
			provenance += fmt.Sprintf(" dataset=%s span_steps=%d span_tokens=%d", claim.Dataset, claim.SpanSteps, claim.SpanTokens)
		}
		if claim.WallNS > 0 {
			provenance += fmt.Sprintf(" wall=%s context_tokens=%d peak_device_bytes=%d",
				time.Duration(claim.WallNS).Round(time.Millisecond), claim.ContextTokens, claim.PeakDeviceBytes)
		}
		fmt.Fprintf(output, "claim %s: %s (%d evidence) %s\n", claim.Capability, claim.Tier, len(claim.Evidence), provenance)
	}
	fmt.Fprintf(output, "honesty: %d evidence artifact(s) committed in this store, %d identified externally; tiers claim only what their evidence grounds\n",
		committed, external)
	return nil
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
