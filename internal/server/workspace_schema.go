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

type workspaceFieldSchema struct {
	Name         string `json:"name"`
	Label        string `json:"label"`
	Type         string `json:"type"`
	Required     bool   `json:"required,omitzero"`
	IdentityKind string `json:"identity_kind,omitzero"`
	Pattern      string `json:"pattern,omitzero"`
	MinimumItems *int   `json:"minimum_items,omitempty"`
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
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		if !textcheck.LowerIdentifier(field.Name, len(field.Name)) || field.Label == "" || !workspaceFieldType(field.Type) {
			return errors.New("server: invalid workspace field schema")
		}
		if _, duplicate := seen[field.Name]; duplicate {
			return errors.New("server: duplicate workspace field schema")
		}
		if field.Type == "identity" && field.IdentityKind == "" || field.MinimumItems != nil && *field.MinimumItems <= 0 {
			return errors.New("server: incomplete workspace field constraint")
		}
		seen[field.Name] = struct{}{}
	}
	return nil
}

func workspaceFieldType(value string) bool {
	switch value {
	case "text", "identity", "string-list":
		return true
	}
	return false
}
