package server

import (
	"net/http"

	"overgo/internal/apimanifest"
)

// workspaceRoutesResponse is the route table as the API explorer renders
// it: every declared route with its method and authentication, and the
// workspace schemas a route's request body follows, so the explorer
// derives its forms from the same declarations the manifest records.
type workspaceRoutesResponse struct {
	Routes  []apimanifest.Route    `json:"routes"`
	Schemas []workspaceRouteSchema `json:"schemas"`
}

type workspaceRouteSchema struct {
	Route  string                `json:"route"`
	Schema workspaceObjectSchema `json:"schema"`
}

// workspaceRouteSchemas binds declared workspace schemas to the routes
// whose request bodies they describe.
var workspaceRouteSchemas = map[string]string{
	"/agents/definitions":      "agent-definition",
	"/automations/definitions": "automation-definition",
	"/automations/schedule":    "automation-trigger",
	"/peers/enroll":            "peer-enrollment",
	"/peers/placement":         "peer-placement",
}

// routeTableProjection is bound at init so the route table can name this
// handler without an initialization cycle through the table it publishes.
var routeTableProjection func() []apimanifest.Route

func init() { routeTableProjection = APIManifestRoutes }

func (h *Handler) workspaceRoutes(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	catalog, err := parseWorkspaceSchema()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	schemas := make(map[string]workspaceObjectSchema, len(catalog.Schemas))
	for _, schema := range catalog.Schemas {
		schemas[schema.ID] = schema
	}
	result := workspaceRoutesResponse{Routes: routeTableProjection(), Schemas: []workspaceRouteSchema{}}
	for _, route := range result.Routes {
		if schema, bound := schemas[workspaceRouteSchemas[route.Path]]; bound {
			result.Schemas = append(result.Schemas, workspaceRouteSchema{Route: route.Path, Schema: schema})
		}
	}
	writeJSON(response, http.StatusOK, result)
}
