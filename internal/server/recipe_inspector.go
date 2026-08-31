package server

import (
	"fmt"
	"net/http"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

type RecipeInspectionAPI interface {
	RecipeRuntimeDescription(recipe.Task) (modelrecipe.RuntimeDescription, error)
}

type activeRecipeResponse struct {
	Admitted bool                            `json:"admitted"`
	Recipe   *modelrecipe.RuntimeDescription `json:"recipe,omitempty"`
	Refusal  string                          `json:"refusal,omitzero"`
}

func (h *Handler) activeRecipe(response http.ResponseWriter, request *http.Request) {
	task := recipe.Task(request.URL.Query().Get("task"))
	if !task.Valid() {
		writeInvalidRequest(response, fmt.Errorf("invalid recipe task %q", task))
		return
	}
	inspector, ok := h.generator.(RecipeInspectionAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "recipe inspection is unavailable")
		return
	}
	description, err := inspector.RecipeRuntimeDescription(task)
	if err != nil {
		writeJSON(response, http.StatusOK, activeRecipeResponse{Refusal: err.Error()})
		return
	}
	writeJSON(response, http.StatusOK, activeRecipeResponse{Admitted: true, Recipe: &description})
}
