package server

import (
	"reflect"
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

// TestWebUIRouteSchemasMatchHandlerBodies holds every schema the API
// explorer binds to a route to the body that route's handler decodes: each
// schema field is a top-level JSON field of that body, so a form the
// explorer renders sends a request the handler accepts. A schema describing
// part of a body (the automation trigger, the flat peer placement form) was
// bound once and sent fields the strict decoder refuses.
func TestWebUIRouteSchemasMatchHandlerBodies(t *testing.T) {
	t.Parallel()
	bodies := map[string]reflect.Type{
		"/agents/definitions": reflect.TypeFor[AgentDefinitionInput](),
		"/peers/enroll":       reflect.TypeFor[runrecord.PeerEnrollment](),
	}
	catalog, err := parseWorkspaceSchema()
	if err != nil {
		t.Fatal(err)
	}
	schemas := make(map[string]workspaceObjectSchema, len(catalog.Schemas))
	for _, schema := range catalog.Schemas {
		schemas[schema.ID] = schema
	}
	for route, schemaID := range workspaceRouteSchemas {
		body, declared := bodies[route]
		if !declared {
			t.Errorf("explorer binds schema %s to %s, whose decoded body this test does not name", schemaID, route)
			continue
		}
		schema, found := schemas[schemaID]
		if !found {
			t.Errorf("explorer binds %s to undeclared schema %s", route, schemaID)
			continue
		}
		fields := jsonFieldNames(body)
		for _, field := range schema.Fields {
			if !fields[field.Name] {
				t.Errorf("schema %s field %q is not a JSON field of %s, the body %s decodes", schemaID, field.Name, body, route)
			}
		}
	}
	for route := range bodies {
		if _, bound := workspaceRouteSchemas[route]; !bound {
			t.Errorf("%s names a body for %s, which the explorer no longer binds", t.Name(), route)
		}
	}
}

// jsonFieldNames lists the top-level JSON names a struct decodes.
func jsonFieldNames(body reflect.Type) map[string]bool {
	names := map[string]bool{}
	for field := range body.Fields() {
		if !field.IsExported() {
			continue
		}
		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		switch name {
		case "-":
			continue
		case "":
			name = field.Name
		}
		names[name] = true
	}
	return names
}
