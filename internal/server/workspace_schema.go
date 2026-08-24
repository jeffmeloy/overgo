package server

import (
	_ "embed"
	"errors"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

//go:embed workspace_schema.json
var workspaceSchemaJSON []byte

type workspaceFieldOption struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type workspaceFieldApplicability struct {
	Field  string `json:"field"`
	Equals string `json:"equals"`
}

type workspaceFieldSchema struct {
	Name         string                       `json:"name"`
	Label        string                       `json:"label"`
	Type         string                       `json:"type"`
	Required     bool                         `json:"required,omitempty"`
	IdentityKind string                       `json:"identity_kind,omitempty"`
	Unit         string                       `json:"unit,omitempty"`
	Pattern      string                       `json:"pattern,omitempty"`
	Minimum      *int64                       `json:"minimum,omitempty"`
	Maximum      *int64                       `json:"maximum,omitempty"`
	MinimumItems *int                         `json:"minimum_items,omitempty"`
	Options      []workspaceFieldOption       `json:"options,omitempty"`
	When         *workspaceFieldApplicability `json:"when,omitempty"`
}

type workspaceObjectSchema struct {
	ID     string                 `json:"id"`
	Label  string                 `json:"label"`
	Fields []workspaceFieldSchema `json:"fields"`
}

type workspaceSchemaCatalog struct {
	Version uint16                  `json:"version"`
	Schemas []workspaceObjectSchema `json:"schemas"`
}

func (h *Handler) workspaceSchema(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	catalog, err := parseWorkspaceSchema()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	id := request.URL.Query().Get("id")
	if id == "" {
		writeJSON(response, http.StatusOK, catalog)
		return
	}
	for _, schema := range catalog.Schemas {
		if schema.ID == id {
			writeJSON(response, http.StatusOK, schema)
			return
		}
	}
	writeError(response, http.StatusNotFound, "not_found", "workspace schema is absent")
}

func parseWorkspaceSchema() (workspaceSchemaCatalog, error) {
	var catalog workspaceSchemaCatalog
	if err := strictjson.DecodeBytes(workspaceSchemaJSON, &catalog); err != nil {
		return workspaceSchemaCatalog{}, err
	}
	if catalog.Version != artifact.InitialDocumentVersion || len(catalog.Schemas) == 0 {
		return workspaceSchemaCatalog{}, errors.New("server: invalid workspace schema catalog")
	}
	seenSchemas := make(map[string]struct{}, len(catalog.Schemas))
	for _, schema := range catalog.Schemas {
		if !textcheck.LowerIdentifier(schema.ID, len(schema.ID)) || schema.Label == "" || len(schema.Fields) == 0 {
			return workspaceSchemaCatalog{}, errors.New("server: invalid workspace object schema")
		}
		if _, duplicate := seenSchemas[schema.ID]; duplicate {
			return workspaceSchemaCatalog{}, errors.New("server: duplicate workspace object schema")
		}
		seenSchemas[schema.ID] = struct{}{}
		if err := validateWorkspaceFields(schema.Fields); err != nil {
			return workspaceSchemaCatalog{}, err
		}
	}
	return catalog, nil
}

func validateWorkspaceFields(fields []workspaceFieldSchema) error {
	seen := make(map[string]workspaceFieldSchema, len(fields))
	for _, field := range fields {
		if !textcheck.LowerIdentifier(field.Name, len(field.Name)) || field.Label == "" || !workspaceFieldType(field.Type) {
			return errors.New("server: invalid workspace field schema")
		}
		if _, duplicate := seen[field.Name]; duplicate {
			return errors.New("server: duplicate workspace field schema")
		}
		if field.Type == "identity" && field.IdentityKind == "" || field.Type == "enum" && len(field.Options) == 0 ||
			field.Minimum != nil && field.Maximum != nil && *field.Minimum > *field.Maximum ||
			field.MinimumItems != nil && *field.MinimumItems <= 0 {
			return errors.New("server: incomplete workspace field constraint")
		}
		if field.When != nil {
			controller, found := seen[field.When.Field]
			if !found || controller.Type != "enum" || field.When.Equals == "" {
				return errors.New("server: invalid workspace field applicability")
			}
			optionFound := false
			for _, option := range controller.Options {
				optionFound = optionFound || option.Value == field.When.Equals
			}
			if !optionFound {
				return errors.New("server: workspace applicability value is undeclared")
			}
		}
		seen[field.Name] = field
	}
	return nil
}

func workspaceFieldType(value string) bool {
	switch value {
	case "text", "identity", "enum", "integer", "number", "duration", "boolean", "string-list":
		return true
	}
	return false
}
