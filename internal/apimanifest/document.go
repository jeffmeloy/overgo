// Package apimanifest owns the generated release-interface projection.
package apimanifest

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// MediaType identifies API manifests.
	MediaType = "application/vnd.overgo.api-manifest+json"
	// Schema identifies the API manifest schema.
	Schema = "overgo/api-manifest/v1"
)

// BuildContext declares one exact build configuration a release ships for.
type BuildContext struct {
	ID     string   `json:"id"`
	GOOS   string   `json:"goos"`
	GOARCH string   `json:"goarch"`
	Tags   []string `json:"tags,omitempty"`
	Cgo    bool     `json:"cgo,omitempty"`
}

// ContractRef names one document contract by kind, media type, and schema.
type ContractRef struct {
	Kind      artifact.Kind `json:"kind"`
	MediaType string        `json:"media_type"`
	Schema    string        `json:"schema"`
}

// Parameter declares one named flag or environment input of a binary.
type Parameter struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Required bool   `json:"required,omitempty"`
	Default  string `json:"default,omitempty"`
}

// Binary describes one shipped executable: its package, build contexts,
// parameters, exit codes, and input and output document contracts.
type Binary struct {
	Name          string        `json:"name"`
	Package       string        `json:"package"`
	BuildContexts []string      `json:"build_contexts"`
	Flags         []Parameter   `json:"flags,omitempty"`
	Environment   []Parameter   `json:"environment,omitempty"`
	Input         []ContractRef `json:"input,omitempty"`
	Output        []ContractRef `json:"output,omitempty"`
	ExitCodes     []int         `json:"exit_codes,omitempty"`
}

// Route describes one HTTP endpoint contract: method, path, authentication,
// and its request, response, and stream documents.
type Route struct {
	Path           string        `json:"path"`
	Method         string        `json:"method"`
	Authentication string        `json:"authentication"`
	RequestBound   uint64        `json:"request_bound,omitempty"`
	Request        *ContractRef  `json:"request,omitempty"`
	Responses      []ContractRef `json:"responses,omitempty"`
	Stream         *ContractRef  `json:"stream,omitempty"`
	Capability     string        `json:"capability,omitempty"`
}

// Document declares one owned document contract and the source that defines it.
type Document struct {
	Name           string        `json:"name"`
	Owner          string        `json:"owner"`
	VersionOwner   string        `json:"version_owner"`
	Source         string        `json:"source"`
	SourceIdentity string        `json:"source_identity"`
	Kind           artifact.Kind `json:"kind"`
	MediaType      string        `json:"media_type"`
	Schema         string        `json:"schema"`
	BuildContexts  []string      `json:"build_contexts,omitempty"`
}

// Module describes one task-owning module and its document contracts.
type Module struct {
	ID      string        `json:"id"`
	Owner   string        `json:"owner"`
	Tasks   []string      `json:"tasks"`
	Inputs  []ContractRef `json:"inputs,omitempty"`
	Outputs []ContractRef `json:"outputs,omitempty"`
}

// Protocol describes one non-HTTP interface and the contracts it carries.
type Protocol struct {
	Name      string        `json:"name"`
	Owner     string        `json:"owner"`
	Transport string        `json:"transport"`
	Contracts []ContractRef `json:"contracts"`
}

// Workspace names one release workspace file by path and content digest.
type Workspace struct {
	Name      string `json:"name"`
	Path      string `json:"path"`
	ContentID string `json:"content_id"`
}

// Authority names one authority document by path and content digest,
// optionally bound to a committed artifact.
type Authority struct {
	Name      string       `json:"name"`
	Path      string       `json:"path"`
	ContentID string       `json:"content_id"`
	Artifact  *artifact.ID `json:"artifact,omitempty"`
}

// Manifest is the canonical release-interface projection: build contexts,
// binaries, routes, documents, modules, protocols, workspaces, and authorities.
type Manifest struct {
	Version        uint16         `json:"version"`
	Release        string         `json:"release"`
	SourceIdentity string         `json:"source_identity"`
	BuildContexts  []BuildContext `json:"build_contexts"`
	Binaries       []Binary       `json:"binaries,omitempty"`
	Routes         []Route        `json:"routes,omitempty"`
	Documents      []Document     `json:"documents,omitempty"`
	Modules        []Module       `json:"modules,omitempty"`
	Protocols      []Protocol     `json:"protocols,omitempty"`
	Workspaces     []Workspace    `json:"workspaces,omitempty"`
	Authorities    []Authority    `json:"authorities"`
	ID             artifact.ID    `json:"-"`
}

var codec = artifact.JSONDocumentCodec(
	"API manifest", artifact.KindProfile, MediaType, Schema, canonicalize,
	func(value Manifest) artifact.ID { return value.ID },
	func(value *Manifest, id artifact.ID) { value.ID = id }, clone,
)

// New canonicalizes and identifies one manifest.
func New(value Manifest) (Manifest, error) { return codec.New(value) }

// Parse decodes one canonical manifest.
func Parse(data []byte) (Manifest, error) { return codec.Parse(data) }

// Validate checks the manifest's canonical form and content-addressed identity.
func (m Manifest) Validate() error { return codec.ValidateIdentity(m) }

// Content returns the canonical committed bytes of the manifest.
func (m Manifest) Content() (artifact.Content, error) { return codec.Content(m) }

func clone(value Manifest) Manifest {
	value.BuildContexts = slices.Clone(value.BuildContexts)
	for index := range value.BuildContexts {
		value.BuildContexts[index].Tags = slices.Clone(value.BuildContexts[index].Tags)
	}
	value.Binaries = slices.Clone(value.Binaries)
	for index := range value.Binaries {
		value.Binaries[index].BuildContexts = slices.Clone(value.Binaries[index].BuildContexts)
		value.Binaries[index].Flags = slices.Clone(value.Binaries[index].Flags)
		value.Binaries[index].Environment = slices.Clone(value.Binaries[index].Environment)
		value.Binaries[index].Input = slices.Clone(value.Binaries[index].Input)
		value.Binaries[index].Output = slices.Clone(value.Binaries[index].Output)
		value.Binaries[index].ExitCodes = slices.Clone(value.Binaries[index].ExitCodes)
	}
	value.Routes = slices.Clone(value.Routes)
	for index := range value.Routes {
		value.Routes[index].Request = cloneRef(value.Routes[index].Request)
		value.Routes[index].Responses = slices.Clone(value.Routes[index].Responses)
		value.Routes[index].Stream = cloneRef(value.Routes[index].Stream)
	}
	value.Documents = slices.Clone(value.Documents)
	for index := range value.Documents {
		value.Documents[index].BuildContexts = slices.Clone(value.Documents[index].BuildContexts)
	}
	value.Modules = slices.Clone(value.Modules)
	for index := range value.Modules {
		value.Modules[index].Tasks = slices.Clone(value.Modules[index].Tasks)
		value.Modules[index].Inputs = slices.Clone(value.Modules[index].Inputs)
		value.Modules[index].Outputs = slices.Clone(value.Modules[index].Outputs)
	}
	value.Protocols = slices.Clone(value.Protocols)
	for index := range value.Protocols {
		value.Protocols[index].Contracts = slices.Clone(value.Protocols[index].Contracts)
	}
	value.Workspaces = slices.Clone(value.Workspaces)
	value.Authorities = slices.Clone(value.Authorities)
	for index := range value.Authorities {
		if value.Authorities[index].Artifact != nil {
			id := *value.Authorities[index].Artifact
			value.Authorities[index].Artifact = &id
		}
	}
	return value
}

func cloneRef(value *ContractRef) *ContractRef {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func canonicalize(value *Manifest) error {
	if value == nil {
		return errors.New("API manifest is nil")
	}
	for index := range value.BuildContexts {
		slices.Sort(value.BuildContexts[index].Tags)
	}
	for index := range value.Binaries {
		slices.Sort(value.Binaries[index].BuildContexts)
		slices.SortFunc(value.Binaries[index].Flags, compareParameter)
		slices.SortFunc(value.Binaries[index].Environment, compareParameter)
		slices.SortFunc(value.Binaries[index].Input, compareContract)
		slices.SortFunc(value.Binaries[index].Output, compareContract)
		slices.Sort(value.Binaries[index].ExitCodes)
	}
	for index := range value.Routes {
		slices.SortFunc(value.Routes[index].Responses, compareContract)
	}
	for index := range value.Documents {
		slices.Sort(value.Documents[index].BuildContexts)
	}
	for index := range value.Modules {
		slices.Sort(value.Modules[index].Tasks)
		slices.SortFunc(value.Modules[index].Inputs, compareContract)
		slices.SortFunc(value.Modules[index].Outputs, compareContract)
	}
	for index := range value.Protocols {
		slices.SortFunc(value.Protocols[index].Contracts, compareContract)
	}
	slices.SortFunc(value.BuildContexts, func(left, right BuildContext) int { return cmp.Compare(left.ID, right.ID) })
	slices.SortFunc(value.Binaries, func(left, right Binary) int { return cmp.Compare(left.Name, right.Name) })
	slices.SortFunc(value.Routes, func(left, right Route) int {
		return cmp.Or(cmp.Compare(left.Path, right.Path), cmp.Compare(left.Method, right.Method))
	})
	slices.SortFunc(value.Documents, func(left, right Document) int {
		return cmp.Compare(documentKey(left.Kind, left.MediaType, left.Schema), documentKey(right.Kind, right.MediaType, right.Schema))
	})
	slices.SortFunc(value.Modules, func(left, right Module) int { return cmp.Compare(left.ID, right.ID) })
	slices.SortFunc(value.Protocols, func(left, right Protocol) int { return cmp.Compare(left.Name, right.Name) })
	slices.SortFunc(value.Workspaces, func(left, right Workspace) int { return cmp.Compare(left.Name, right.Name) })
	slices.SortFunc(value.Authorities, func(left, right Authority) int { return cmp.Compare(left.Name, right.Name) })
	return validate(*value)
}

func validate(value Manifest) error {
	if value.Version != artifact.InitialDocumentVersion || !validText(value.Release) || !validDigest(value.SourceIdentity) || len(value.BuildContexts) == 0 || len(value.Authorities) == 0 {
		return errors.New("API manifest header is invalid")
	}
	contexts := map[string]bool{}
	for _, context := range value.BuildContexts {
		if !validText(context.ID) || !validText(context.GOOS) || !validText(context.GOARCH) || contexts[context.ID] || duplicateStrings(context.Tags) {
			return errors.New("API manifest build context is invalid")
		}
		contexts[context.ID] = true
	}
	documents := map[string]bool{}
	for _, document := range value.Documents {
		key := documentKey(document.Kind, document.MediaType, document.Schema)
		if !validText(document.Name) || !validText(document.Owner) || !validText(document.VersionOwner) || !validText(document.Source) || !validDigest(document.SourceIdentity) || !validContract(ContractRef{Kind: document.Kind, MediaType: document.MediaType, Schema: document.Schema}) || documents[key] || !knownContexts(document.BuildContexts, contexts) {
			return errors.New("API manifest document contract is invalid")
		}
		documents[key] = true
	}
	if err := validateBinaries(value.Binaries, contexts, documents); err != nil {
		return err
	}
	if err := validateRoutes(value.Routes, documents); err != nil {
		return err
	}
	if err := validateModules(value.Modules, documents); err != nil {
		return err
	}
	if err := validateProtocols(value.Protocols, documents); err != nil {
		return err
	}
	if err := validateNamedContent(value.Workspaces, func(item Workspace) (string, string, string) { return item.Name, item.Path, item.ContentID }); err != nil {
		return fmt.Errorf("API manifest workspace: %w", err)
	}
	if err := validateNamedContent(value.Authorities, func(item Authority) (string, string, string) { return item.Name, item.Path, item.ContentID }); err != nil {
		return fmt.Errorf("API manifest authority: %w", err)
	}
	return nil
}

func validateBinaries(values []Binary, contexts, documents map[string]bool) error {
	seen := map[string]bool{}
	for _, value := range values {
		if !validText(value.Name) || !validText(value.Package) || seen[value.Name] || len(value.BuildContexts) == 0 || !knownContexts(value.BuildContexts, contexts) || duplicateParameters(value.Flags) || duplicateParameters(value.Environment) || duplicateInts(value.ExitCodes) || !knownContracts(value.Input, documents) || !knownContracts(value.Output, documents) {
			return errors.New("API manifest binary is invalid")
		}
		seen[value.Name] = true
	}
	return nil
}

func validateRoutes(values []Route, documents map[string]bool) error {
	seen := map[string]bool{}
	for _, value := range values {
		key := value.Method + " " + value.Path
		refs := slices.Clone(value.Responses)
		if value.Request != nil {
			refs = append(refs, *value.Request)
		}
		if value.Stream != nil {
			refs = append(refs, *value.Stream)
		}
		if !strings.HasPrefix(value.Path, "/") || !validText(value.Method) || !validText(value.Authentication) || seen[key] || !knownContracts(refs, documents) {
			return errors.New("API manifest route is invalid")
		}
		seen[key] = true
	}
	return nil
}

func validateModules(values []Module, documents map[string]bool) error {
	seen := map[string]bool{}
	for _, value := range values {
		if !validText(value.ID) || !validText(value.Owner) || seen[value.ID] || len(value.Tasks) == 0 || duplicateStrings(value.Tasks) || !knownContracts(value.Inputs, documents) || !knownContracts(value.Outputs, documents) {
			return errors.New("API manifest module is invalid")
		}
		seen[value.ID] = true
	}
	return nil
}

func validateProtocols(values []Protocol, documents map[string]bool) error {
	seen := map[string]bool{}
	for _, value := range values {
		if !validText(value.Name) || !validText(value.Owner) || !validText(value.Transport) || seen[value.Name] || len(value.Contracts) == 0 || !knownContracts(value.Contracts, documents) {
			return errors.New("API manifest protocol is invalid")
		}
		seen[value.Name] = true
	}
	return nil
}

func validateNamedContent[T any](values []T, fields func(T) (string, string, string)) error {
	seen := map[string]bool{}
	for _, value := range values {
		name, path, identity := fields(value)
		if !validText(name) || !validText(path) || !validDigest(identity) || seen[name] {
			return errors.New("invalid or duplicate entry")
		}
		seen[name] = true
	}
	return nil
}

func knownContracts(values []ContractRef, documents map[string]bool) bool {
	for _, value := range values {
		if !validContract(value) || !documents[documentKey(value.Kind, value.MediaType, value.Schema)] {
			return false
		}
	}
	return !duplicateContracts(values)
}

func validContract(value ContractRef) bool {
	return value.Kind != artifact.KindInvalid && validText(value.MediaType) && validText(value.Schema)
}

func knownContexts(values []string, contexts map[string]bool) bool {
	if duplicateStrings(values) {
		return false
	}
	for _, value := range values {
		if !contexts[value] {
			return false
		}
	}
	return true
}

func compareParameter(left, right Parameter) int { return cmp.Compare(left.Name, right.Name) }

func compareContract(left, right ContractRef) int {
	return cmp.Compare(documentKey(left.Kind, left.MediaType, left.Schema), documentKey(right.Kind, right.MediaType, right.Schema))
}

func documentKey(kind artifact.Kind, mediaType, schema string) string {
	return kind.String() + "\x00" + mediaType + "\x00" + schema
}

func duplicateParameters(values []Parameter) bool {
	var previous Parameter
	var havePrevious bool
	for _, value := range values {
		if !validText(value.Name) || !validText(value.Type) || havePrevious && previous.Name == value.Name {
			return true
		}
		previous = value
		havePrevious = true
	}
	return false
}

func duplicateContracts(values []ContractRef) bool {
	var previous ContractRef
	var havePrevious bool
	for _, value := range values {
		if havePrevious && compareContract(previous, value) == 0 {
			return true
		}
		previous = value
		havePrevious = true
	}
	return false
}

func duplicateStrings(values []string) bool {
	var previous string
	var havePrevious bool
	for _, value := range values {
		if !validText(value) || havePrevious && previous == value {
			return true
		}
		previous = value
		havePrevious = true
	}
	return false
}

func duplicateInts(values []int) bool {
	var previous int
	var havePrevious bool
	for _, value := range values {
		if havePrevious && previous == value {
			return true
		}
		previous = value
		havePrevious = true
	}
	return false
}

func validText(value string) bool {
	return value != "" && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n")
}

func validDigest(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}
