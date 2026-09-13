// Package codemanifest defines the canonical structural authority used to
// plan verification for an exact source snapshot.
package codemanifest

import (
	"cmp"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// Version is the current canonical code-manifest document version.
	Version uint16 = 1
	// MediaType identifies encoded code-manifest documents.
	MediaType = "application/vnd.overgo.code-manifest+json"
	// Schema identifies the exact stored code-manifest schema.
	Schema = "overgo/code-manifest/v1"

	maxTextBytes        = 2048
	maxPathBytes        = 8192
	maxBuildContexts    = 256
	maxFiles            = 1 << 20
	maxSymbols          = 1 << 22
	maxReferences       = 1 << 24
	maxExternalInputs   = 1 << 20
	maxUncertaintyItems = 1 << 20
)

// Analyzer identifies the implementation whose structural conclusions are
// encoded in a manifest. Version must change when analysis semantics change.
type Analyzer struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// BuildContext identifies one complete source-selection context.
type BuildContext struct {
	ID     string   `json:"id"`
	GOOS   string   `json:"goos"`
	GOARCH string   `json:"goarch"`
	Tags   []string `json:"tags,omitempty"`
	Cgo    bool     `json:"cgo,omitzero"`
}

// File records one parsed source file and the contexts that select it.
type File struct {
	Path             string   `json:"path"`
	ContentID        string   `json:"content_id"`
	Package          string   `json:"package"`
	SelectedContexts []string `json:"selected_contexts,omitempty"`
	BuildExpression  string   `json:"build_expression,omitzero"`
	Generated        bool     `json:"generated,omitzero"`
	Test             bool     `json:"test,omitzero"`
}

// SymbolKind distinguishes structural declaration classes without embedding
// language-specific AST values in stored documents.
type SymbolKind string

const (
	// SymbolFunction identifies a package function.
	SymbolFunction SymbolKind = "function"
	// SymbolMethod identifies a receiver method.
	SymbolMethod SymbolKind = "method"
	// SymbolType identifies a named type declaration.
	SymbolType SymbolKind = "type"
	// SymbolField identifies a named struct or interface field.
	SymbolField SymbolKind = "field"
	// SymbolVariable identifies a variable declaration.
	SymbolVariable SymbolKind = "variable"
	// SymbolConstant identifies a constant declaration.
	SymbolConstant SymbolKind = "constant"
)

// SymbolID is the stable declaration key used by edges and ownership joins.
type SymbolID struct {
	Package  string     `json:"package"`
	Context  string     `json:"context"`
	Receiver string     `json:"receiver,omitzero"`
	Name     string     `json:"name"`
	Kind     SymbolKind `json:"kind"`
}

// Symbol binds a declaration to source and normalized signature/body hashes.
type Symbol struct {
	ID              SymbolID `json:"id"`
	File            string   `json:"file"`
	SignatureSHA256 string   `json:"signature_sha256"`
	BodySHA256      string   `json:"body_sha256,omitzero"`
	Exported        bool     `json:"exported,omitzero"`
}

// ReferenceKind identifies the syntactic relationship represented by an edge.
type ReferenceKind string

const (
	// ReferenceCall identifies direct invocation syntax, not observed execution.
	ReferenceCall ReferenceKind = "call"
	// ReferenceType identifies use of a named type.
	ReferenceType ReferenceKind = "type"
	// ReferenceField identifies use of a named field.
	ReferenceField ReferenceKind = "field"
	// ReferenceInterface identifies conservative unresolved receiver reach.
	ReferenceInterface ReferenceKind = "interface"
	// ReferenceUse identifies a resolved declaration use whose syntax does not
	// prove a narrower relationship.
	ReferenceUse ReferenceKind = "use"
)

// Reference is one resolved directed edge between declarations.
type Reference struct {
	From SymbolID      `json:"from"`
	To   SymbolID      `json:"to"`
	Kind ReferenceKind `json:"kind"`
	// A zero line denotes a historical edge without source-site provenance.
	Line   int `json:"line,omitzero"`
	Offset int `json:"offset,omitzero"`
}

// ExternalInput binds non-Go authority bytes to their structural owner.
type ExternalInput struct {
	Path      string `json:"path"`
	ContentID string `json:"content_id"`
	Kind      string `json:"kind"`
	Owner     string `json:"owner"`
}

// UncertaintyKind identifies an analysis boundary that prevents proof of
// verification independence.
type UncertaintyKind string

const (
	// UncertaintyReflection marks dynamically resolved references.
	UncertaintyReflection UncertaintyKind = "reflection"
	// UncertaintyCgo marks declarations exported across the cgo boundary.
	UncertaintyCgo UncertaintyKind = "cgo"
	// UncertaintyInterface marks unresolved dynamic receiver dispatch.
	UncertaintyInterface UncertaintyKind = "interface"
	// UncertaintyGenerated marks generated source or an unavailable generator.
	UncertaintyGenerated UncertaintyKind = "generated"
	// UncertaintyBuildSelection marks source omitted by an analyzed build context.
	UncertaintyBuildSelection UncertaintyKind = "build-selection"
	// UncertaintyNonGo marks an owned non-Go input without a structural adapter.
	UncertaintyNonGo UncertaintyKind = "non-go"
	// UncertaintyOutsideSnapshot marks a changed path absent from both snapshots.
	UncertaintyOutsideSnapshot UncertaintyKind = "outside-snapshot"
	// UncertaintyAnalysis marks an analyzer failure or incomplete result.
	UncertaintyAnalysis UncertaintyKind = "analysis"
	// UncertaintyExternal marks a declaration observable outside the snapshot.
	UncertaintyExternal UncertaintyKind = "external"
)

// Uncertainty records a boundary that cannot prove verification independence.
// Its presence is affirmative evidence to broaden, never narrow, selection.
type Uncertainty struct {
	Kind    UncertaintyKind `json:"kind"`
	Path    string          `json:"path,omitzero"`
	Context string          `json:"context,omitzero"`
	Symbol  *SymbolID       `json:"symbol,omitempty"`
	Reason  string          `json:"reason"`
}

// Manifest is the complete canonical structural view of one source snapshot.
type Manifest struct {
	Version        uint16          `json:"version"`
	SourceIdentity string          `json:"source_identity"`
	Analyzer       Analyzer        `json:"analyzer"`
	BuildContexts  []BuildContext  `json:"build_contexts"`
	Files          []File          `json:"files"`
	Symbols        []Symbol        `json:"symbols,omitempty"`
	References     []Reference     `json:"references,omitempty"`
	ExternalInputs []ExternalInput `json:"external_inputs,omitempty"`
	Uncertainty    []Uncertainty   `json:"uncertainty,omitempty"`
	ID             artifact.ID     `json:"-"`
}

var codec = artifact.JSONDocumentCodec(
	"code manifest", artifact.KindProfile, MediaType, Schema,
	canonicalize,
	func(value Manifest) artifact.ID { return value.ID },
	func(value *Manifest, id artifact.ID) { value.ID = id },
	clone,
)

// Parse strictly decodes canonical code-manifest bytes.
func Parse(data []byte) (Manifest, error) {
	return codec.Parse(data)
}

// Validate verifies canonical ordering and content identity.
func (m Manifest) Validate() error {
	identified, err := identifyManifest(m)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(m, identified) {
		return errors.New("code manifest: document is not canonical or has an invalid identity")
	}
	return nil
}

// ExclusionAuthority succeeds only for a valid canonical manifest whose
// analysis contains no unresolved boundary. Planners must obtain this verdict
// before converting absence of impact into an exclusion.
func (m Manifest) ExclusionAuthority() error {
	if err := m.Validate(); err != nil {
		return fmt.Errorf("code manifest: exclusion authority: %w", err)
	}
	if len(m.Uncertainty) != 0 {
		return fmt.Errorf("code manifest: exclusion authority: %d unresolved boundaries", len(m.Uncertainty))
	}
	return nil
}

// Content returns the canonical stored representation.
func (m Manifest) Content() (artifact.Content, error) {
	return codec.Content(m)
}

// identifyManifest keeps the rebuildable analysis graph out of the durable
// document-size policy. Large repositories can exceed one artifact document,
// but PublishDigest stores only a bounded summary; identity remains the exact
// SHA-256 of the same canonical JSON bytes used by the legacy full document.
func identifyManifest(value Manifest) (Manifest, error) {
	value.ID = artifact.ID{}
	if err := canonicalize(&value); err != nil {
		return Manifest{}, err
	}
	data, err := json.Marshal(value)
	if err != nil {
		return Manifest{}, err
	}
	value.ID, err = artifact.IdentifyBytes(artifact.KindProfile, data)
	return value, err
}

func clone(value Manifest) Manifest {
	value.BuildContexts = slices.Clone(value.BuildContexts)
	for index := range value.BuildContexts {
		value.BuildContexts[index].Tags = slices.Clone(value.BuildContexts[index].Tags)
	}
	value.Files = slices.Clone(value.Files)
	for index := range value.Files {
		value.Files[index].SelectedContexts = slices.Clone(value.Files[index].SelectedContexts)
	}
	value.Symbols = slices.Clone(value.Symbols)
	value.References = slices.Clone(value.References)
	value.ExternalInputs = slices.Clone(value.ExternalInputs)
	value.Uncertainty = slices.Clone(value.Uncertainty)
	for index := range value.Uncertainty {
		if value.Uncertainty[index].Symbol != nil {
			symbol := *value.Uncertainty[index].Symbol
			value.Uncertainty[index].Symbol = &symbol
		}
	}
	return value
}

func canonicalize(value *Manifest) error {
	if value == nil {
		return errors.New("code manifest: nil document")
	}
	for index := range value.BuildContexts {
		slices.Sort(value.BuildContexts[index].Tags)
		value.BuildContexts[index].Tags = slices.Compact(value.BuildContexts[index].Tags)
	}
	for index := range value.Files {
		slices.Sort(value.Files[index].SelectedContexts)
		value.Files[index].SelectedContexts = slices.Compact(value.Files[index].SelectedContexts)
	}
	slices.SortFunc(value.BuildContexts, func(left, right BuildContext) int { return cmp.Compare(left.ID, right.ID) })
	slices.SortFunc(value.Files, func(left, right File) int { return cmp.Compare(left.Path, right.Path) })
	slices.SortFunc(value.Symbols, func(left, right Symbol) int { return cmp.Compare(symbolKey(left.ID), symbolKey(right.ID)) })
	slices.SortFunc(value.References, func(left, right Reference) int {
		return cmp.Or(cmp.Compare(symbolKey(left.From), symbolKey(right.From)), cmp.Compare(symbolKey(left.To), symbolKey(right.To)), cmp.Compare(left.Kind, right.Kind), cmp.Compare(left.Line, right.Line), cmp.Compare(left.Offset, right.Offset))
	})
	slices.SortFunc(value.ExternalInputs, func(left, right ExternalInput) int {
		return cmp.Or(cmp.Compare(left.Path, right.Path), cmp.Compare(left.Kind, right.Kind), cmp.Compare(left.Owner, right.Owner))
	})
	slices.SortFunc(value.Uncertainty, func(left, right Uncertainty) int {
		return cmp.Or(cmp.Compare(left.Kind, right.Kind), cmp.Compare(left.Path, right.Path), cmp.Compare(left.Context, right.Context), cmp.Compare(optionalSymbolKey(left.Symbol), optionalSymbolKey(right.Symbol)), cmp.Compare(left.Reason, right.Reason))
	})
	return validate(*value)
}

func validate(value Manifest) error {
	if value.Version != Version || !validDigest(value.SourceIdentity) {
		return errors.New("code manifest: invalid version or source identity")
	}
	if !validText(value.Analyzer.Name, maxTextBytes) || !validText(value.Analyzer.Version, maxTextBytes) {
		return errors.New("code manifest: invalid analyzer provenance")
	}
	if len(value.BuildContexts) == 0 || len(value.BuildContexts) > maxBuildContexts || len(value.Files) == 0 || len(value.Files) > maxFiles || len(value.Symbols) > maxSymbols || len(value.References) > maxReferences || len(value.ExternalInputs) > maxExternalInputs || len(value.Uncertainty) > maxUncertaintyItems {
		return errors.New("code manifest: invalid collection count")
	}
	contexts := make(map[string]bool, len(value.BuildContexts))
	for _, context := range value.BuildContexts {
		if !validText(context.ID, maxTextBytes) || !validText(context.GOOS, maxTextBytes) || !validText(context.GOARCH, maxTextBytes) || contexts[context.ID] {
			return errors.New("code manifest: invalid or duplicate build context")
		}
		contexts[context.ID] = true
		for _, tag := range context.Tags {
			if !validText(tag, maxTextBytes) {
				return errors.New("code manifest: invalid build tag")
			}
		}
	}
	files := make(map[string]bool, len(value.Files))
	for _, file := range value.Files {
		if !validPath(file.Path) || !validDigest(file.ContentID) || !validText(file.Package, maxPathBytes) || files[file.Path] || len(file.BuildExpression) > maxTextBytes || strings.ContainsAny(file.BuildExpression, "\x00\r\n") {
			return errors.New("code manifest: invalid or duplicate file")
		}
		files[file.Path] = true
		for _, context := range file.SelectedContexts {
			if !contexts[context] {
				return fmt.Errorf("code manifest: file %q names unknown build context %q", file.Path, context)
			}
		}
	}
	symbols := make(map[string]bool, len(value.Symbols))
	for _, symbol := range value.Symbols {
		key := symbolKey(symbol.ID)
		if err := validateSymbolID(symbol.ID); err != nil || !files[symbol.File] || !validDigest(symbol.SignatureSHA256) || symbol.BodySHA256 != "" && !validDigest(symbol.BodySHA256) || symbols[key] {
			return errors.New("code manifest: invalid or duplicate symbol")
		}
		symbols[key] = true
	}
	references := map[Reference]bool{}
	for _, reference := range value.References {
		if !symbols[symbolKey(reference.From)] || !symbols[symbolKey(reference.To)] || !validReferenceKind(reference.Kind) || references[reference] || reference.Line < 0 || reference.Offset < 0 || reference.Line == 0 && reference.Offset != 0 {
			return errors.New("code manifest: invalid or duplicate reference")
		}
		references[reference] = true
	}
	inputs := map[string]bool{}
	for _, input := range value.ExternalInputs {
		key := input.Path + "\x00" + input.Kind + "\x00" + input.Owner
		if !validPath(input.Path) || !validDigest(input.ContentID) || !validText(input.Kind, maxTextBytes) || !validText(input.Owner, maxPathBytes) || inputs[key] {
			return errors.New("code manifest: invalid or duplicate external input")
		}
		inputs[key] = true
	}
	uncertainty := map[string]bool{}
	for _, item := range value.Uncertainty {
		key := string(item.Kind) + "\x00" + item.Path + "\x00" + item.Context + "\x00" + optionalSymbolKey(item.Symbol) + "\x00" + item.Reason
		if !validUncertaintyKind(item.Kind) || item.Path != "" && !validPath(item.Path) || item.Context != "" && !contexts[item.Context] || !validText(item.Reason, maxPathBytes) || uncertainty[key] {
			return errors.New("code manifest: invalid or duplicate uncertainty")
		}
		if item.Symbol != nil {
			if err := validateSymbolID(*item.Symbol); err != nil {
				return errors.New("code manifest: invalid uncertainty symbol")
			}
		}
		uncertainty[key] = true
	}
	return nil
}

func validateSymbolID(id SymbolID) error {
	if !validText(id.Package, maxPathBytes) || !validText(id.Context, maxTextBytes) || !validText(id.Name, maxTextBytes) || id.Receiver != "" && !validText(id.Receiver, maxTextBytes) {
		return errors.New("invalid symbol text")
	}
	switch id.Kind {
	case SymbolFunction, SymbolMethod, SymbolType, SymbolField, SymbolVariable, SymbolConstant:
		return nil
	default:
		return errors.New("invalid symbol kind")
	}
}

func validReferenceKind(kind ReferenceKind) bool {
	switch kind {
	case ReferenceCall, ReferenceType, ReferenceField, ReferenceInterface, ReferenceUse:
		return true
	default:
		return false
	}
}

func validUncertaintyKind(kind UncertaintyKind) bool {
	switch kind {
	case UncertaintyReflection, UncertaintyCgo, UncertaintyInterface, UncertaintyGenerated,
		UncertaintyBuildSelection, UncertaintyNonGo, UncertaintyOutsideSnapshot, UncertaintyAnalysis,
		UncertaintyExternal:
		return true
	default:
		return false
	}
}

func symbolKey(id SymbolID) string {
	return id.Package + "\x00" + id.Context + "\x00" + id.Receiver + "\x00" + id.Name + "\x00" + string(id.Kind)
}

func optionalSymbolKey(id *SymbolID) string {
	if id == nil {
		return ""
	}
	return symbolKey(*id)
}

func validText(value string, limit int) bool {
	return value != "" && len(value) <= limit && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n")
}

func validPath(value string) bool {
	return validText(value, maxPathBytes) && !strings.Contains(value, "\\") && !strings.HasPrefix(value, "/") && value != "." && value != ".." && !strings.HasPrefix(value, "../") && path.Clean(value) == value
}

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == 32 && strings.ToLower(value) == value
}
