package agenttool

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	openAPIMaxBytes      = 4 << 20
	openAPIMaxOperations = 1024
	openAPIMaxFields     = 256
)

// OpenAPICompilation is a deterministic, inactive set of candidate manuals.
// Source binds the exact external description bytes; no registered alias is
// changed by compilation.
type OpenAPICompilation struct {
	Source  artifact.ID `json:"source"`
	Manuals []Manual    `json:"manuals"`
	source  []byte
}

// SourceContent returns the exact bounded provider description as a durable file.
func (compilation OpenAPICompilation) SourceContent() (artifact.Content, error) {
	contract := artifact.DocumentContract{
		Kind: artifact.KindFile, MediaType: artifact.JSONMediaType, Schema: "overgo/openapi-source/v1",
	}
	content, err := contract.ContentBytes(compilation.source)
	if err != nil || content.Descriptor.ID != compilation.Source {
		return artifact.Content{}, errors.Join(errors.New("agent tool: openapi source identity differs"), err)
	}
	return content, nil
}

type openAPIDocument struct {
	OpenAPI           string                     `json:"openapi"`
	Info              json.RawMessage            `json:"info"`
	Servers           json.RawMessage            `json:"servers,omitempty"`
	Paths             map[string]openAPIPathItem `json:"paths"`
	Components        json.RawMessage            `json:"components,omitempty"`
	Security          json.RawMessage            `json:"security,omitempty"`
	Tags              json.RawMessage            `json:"tags,omitempty"`
	ExternalDocs      json.RawMessage            `json:"externalDocs,omitempty"`
	JSONSchemaDialect string                     `json:"jsonSchemaDialect,omitzero"`
	Webhooks          json.RawMessage            `json:"webhooks,omitempty"`
}

type openAPIPathItem struct {
	Ref         string            `json:"$ref,omitzero"`
	Summary     string            `json:"summary,omitzero"`
	Description string            `json:"description,omitzero"`
	Get         *openAPIOperation `json:"get,omitempty"`
	Put         *openAPIOperation `json:"put,omitempty"`
	Post        *openAPIOperation `json:"post,omitempty"`
	Delete      *openAPIOperation `json:"delete,omitempty"`
	Options     *openAPIOperation `json:"options,omitempty"`
	Head        *openAPIOperation `json:"head,omitempty"`
	Patch       *openAPIOperation `json:"patch,omitempty"`
	Trace       *openAPIOperation `json:"trace,omitempty"`
	Servers     json.RawMessage   `json:"servers,omitempty"`
	Parameters  json.RawMessage   `json:"parameters,omitempty"`
}

type openAPIOperation struct {
	Tags         json.RawMessage     `json:"tags,omitempty"`
	Summary      string              `json:"summary,omitzero"`
	Description  string              `json:"description,omitzero"`
	ExternalDocs json.RawMessage     `json:"externalDocs,omitempty"`
	OperationID  string              `json:"operationId"`
	Parameters   json.RawMessage     `json:"parameters,omitempty"`
	RequestBody  *openAPIRequestBody `json:"requestBody,omitempty"`
	Responses    json.RawMessage     `json:"responses"`
	Callbacks    json.RawMessage     `json:"callbacks,omitempty"`
	Deprecated   bool                `json:"deprecated,omitzero"`
	Security     json.RawMessage     `json:"security,omitempty"`
	Servers      json.RawMessage     `json:"servers,omitempty"`
	Effect       Effect              `json:"x-overgo-effect"`
}

type openAPIRequestBody struct {
	Description string                      `json:"description,omitzero"`
	Content     map[string]openAPIMediaType `json:"content"`
	Required    bool                        `json:"required,omitzero"`
}

type openAPIMediaType struct {
	Schema   openAPISchema   `json:"schema"`
	Example  json.RawMessage `json:"example,omitempty"`
	Examples json.RawMessage `json:"examples,omitempty"`
	Encoding json.RawMessage `json:"encoding,omitempty"`
}

type openAPISchema struct {
	Ref                  string                   `json:"$ref,omitzero"`
	Type                 string                   `json:"type"`
	Description          string                   `json:"description,omitzero"`
	Format               string                   `json:"format,omitzero"`
	Properties           map[string]openAPISchema `json:"properties,omitempty"`
	Required             []string                 `json:"required,omitempty"`
	AdditionalProperties json.RawMessage          `json:"additionalProperties,omitempty"`
	Enum                 json.RawMessage          `json:"enum,omitempty"`
	Default              json.RawMessage          `json:"default,omitempty"`
	Example              json.RawMessage          `json:"example,omitempty"`
	Nullable             bool                     `json:"nullable,omitzero"`
	ReadOnly             bool                     `json:"readOnly,omitzero"`
	WriteOnly            bool                     `json:"writeOnly,omitzero"`
	Deprecated           bool                     `json:"deprecated,omitzero"`
}

// CompileOpenAPI reads one bounded strict OpenAPI JSON document and compiles
// every declared POST operation into an inactive manual. Unsupported methods,
// references, parameter channels, and field shapes fail the whole compile;
// discovery can never become silent partial registration.
func CompileOpenAPI(reader io.Reader, baseURL string) (OpenAPICompilation, error) {
	if reader == nil {
		return OpenAPICompilation{}, errors.New("agent tool: openapi reader is absent")
	}
	var source bytes.Buffer
	limited := io.LimitReader(reader, openAPIMaxBytes+1)
	if _, err := io.Copy(&source, limited); err != nil {
		return OpenAPICompilation{}, err
	}
	if source.Len() == 0 || source.Len() > openAPIMaxBytes {
		return OpenAPICompilation{}, errors.New("agent tool: openapi document exceeds the size bound")
	}
	var document openAPIDocument
	if err := strictjson.DecodeBytes(source.Bytes(), &document); err != nil {
		return OpenAPICompilation{}, fmt.Errorf("agent tool: decode openapi document: %w", err)
	}
	if !strings.HasPrefix(document.OpenAPI, "3.0.") && !strings.HasPrefix(document.OpenAPI, "3.1.") {
		return OpenAPICompilation{}, errors.New("agent tool: unsupported openapi revision")
	}
	if !strictjson.HasValue(document.Info) || len(document.Paths) == 0 {
		return OpenAPICompilation{}, errors.New("agent tool: openapi information or paths are absent")
	}
	base, err := url.Parse(baseURL)
	if err != nil || (base.Scheme != "http" && base.Scheme != "https") || base.Host == "" || base.RawQuery != "" || base.Fragment != "" {
		return OpenAPICompilation{}, errors.New("agent tool: openapi base URL must be an exact http(s) endpoint")
	}
	paths := make([]string, 0, len(document.Paths))
	for path := range document.Paths {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	manuals := make([]Manual, 0, len(paths))
	for _, path := range paths {
		item := document.Paths[path]
		if item.Ref != "" || strictjson.HasValue(item.Parameters) || strings.ContainsAny(path, "{}?#") || !strings.HasPrefix(path, "/") {
			return OpenAPICompilation{}, fmt.Errorf("agent tool: openapi path %q uses an unsupported dynamic boundary", path)
		}
		if item.Get != nil || item.Put != nil || item.Delete != nil || item.Options != nil ||
			item.Head != nil || item.Patch != nil || item.Trace != nil {
			return OpenAPICompilation{}, fmt.Errorf("agent tool: openapi path %q declares a non-POST operation", path)
		}
		if item.Post == nil {
			continue
		}
		manual, err := compileOpenAPIOperation(base, path, *item.Post)
		if err != nil {
			return OpenAPICompilation{}, err
		}
		manuals = append(manuals, manual)
		if len(manuals) > openAPIMaxOperations {
			return OpenAPICompilation{}, errors.New("agent tool: openapi operation count exceeds the bound")
		}
	}
	if len(manuals) == 0 {
		return OpenAPICompilation{}, errors.New("agent tool: openapi document has no supported operations")
	}
	slices.SortFunc(manuals, func(left, right Manual) int { return strings.Compare(left.Name, right.Name) })
	if err := validateManualSet(manuals); err != nil {
		return OpenAPICompilation{}, err
	}
	sourceID, err := artifact.IdentifyBytes(artifact.KindFile, source.Bytes())
	if err != nil {
		return OpenAPICompilation{}, err
	}
	return OpenAPICompilation{Source: sourceID, Manuals: manuals, source: bytes.Clone(source.Bytes())}, nil
}

func compileOpenAPIOperation(base *url.URL, path string, operation openAPIOperation) (Manual, error) {
	if strictjson.HasValue(operation.Parameters) || operation.RequestBody == nil || !operation.RequestBody.Required ||
		!strictjson.HasValue(operation.Responses) {
		return Manual{}, fmt.Errorf("agent tool: openapi operation %q requires one JSON request body and no parameters", operation.OperationID)
	}
	if !operation.Effect.Valid() {
		return Manual{}, fmt.Errorf("agent tool: openapi operation %q lacks explicit effect authority", operation.OperationID)
	}
	description := operation.Description
	if description == "" {
		description = operation.Summary
	}
	media, found := operation.RequestBody.Content["application/json"]
	if !found || len(operation.RequestBody.Content) != 1 || media.Schema.Ref != "" || media.Schema.Type != "object" {
		return Manual{}, fmt.Errorf("agent tool: openapi operation %q requires one inline JSON object schema", operation.OperationID)
	}
	if strictjson.HasValue(media.Schema.AdditionalProperties) &&
		!bytes.Equal(bytes.TrimSpace(media.Schema.AdditionalProperties), []byte("false")) {
		return Manual{}, fmt.Errorf("agent tool: openapi operation %q admits undeclared JSON fields", operation.OperationID)
	}
	if len(media.Schema.Properties) > openAPIMaxFields {
		return Manual{}, fmt.Errorf("agent tool: openapi operation %q field count exceeds the bound", operation.OperationID)
	}
	required := make(map[string]bool, len(media.Schema.Required))
	for _, name := range media.Schema.Required {
		if required[name] {
			return Manual{}, fmt.Errorf("agent tool: openapi operation %q repeats required field %q", operation.OperationID, name)
		}
		required[name] = true
	}
	names := make([]string, 0, len(media.Schema.Properties))
	for name := range media.Schema.Properties {
		names = append(names, name)
	}
	slices.Sort(names)
	fields := make([]Field, 0, len(names))
	for _, name := range names {
		schema := media.Schema.Properties[name]
		if schema.Ref != "" || len(schema.Properties) != 0 {
			return Manual{}, fmt.Errorf("agent tool: openapi operation %q field %q is not an inline scalar", operation.OperationID, name)
		}
		kind, err := openAPIFieldKind(schema)
		if err != nil {
			return Manual{}, fmt.Errorf("agent tool: openapi operation %q field %q: %w", operation.OperationID, name, err)
		}
		fields = append(fields, Field{Name: name, Kind: kind, Required: required[name], Description: schema.Description})
		delete(required, name)
	}
	if len(required) != 0 {
		return Manual{}, fmt.Errorf("agent tool: openapi operation %q requires an undeclared field", operation.OperationID)
	}
	endpoint := *base
	endpoint.Path = strings.TrimRight(base.Path, "/") + path
	endpoint.RawPath = ""
	return NewManual(Manual{
		Name: operation.OperationID, Description: description, Effect: operation.Effect,
		Arguments: fields, Transport: Transport{Kind: TransportHTTP, URL: endpoint.String()},
	})
}

func openAPIFieldKind(schema openAPISchema) (FieldKind, error) {
	if schema.Format != "" || strictjson.HasValue(schema.Enum) || strictjson.HasValue(schema.Default) ||
		schema.Nullable || schema.ReadOnly || schema.WriteOnly {
		return "", errors.New("schema constraints are not representable by a manual field")
	}
	switch schema.Type {
	case "string":
		return FieldString, nil
	case "integer":
		return FieldInteger, nil
	case "number":
		return FieldNumber, nil
	case "boolean":
		return FieldBoolean, nil
	case "object":
		return FieldObject, nil
	default:
		return "", fmt.Errorf("unsupported schema type %q", schema.Type)
	}
}
