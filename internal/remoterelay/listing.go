package remoterelay

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"overgo/internal/remoteprovider"
)

// modelsPath is the provider's model listing beside its chat completions.
const modelsPath = "/models"

// maxListingBytes bounds a provider's model listing: a few thousand
// models with names stay under a megabyte; the bound refuses a
// misbehaving provider rather than reading without end.
const maxListingBytes = 4 << 20

// Listed is one model a provider lists: its id, a display name when the
// listing carries one, and the context length it declares (zero when
// unstated).
type Listed struct {
	ID            string `json:"id"`
	Name          string `json:"name,omitzero"`
	ContextLength uint32 `json:"context_length,omitzero"`
}

// ListModels asks a provider for the models it serves (GET the endpoint's
// models route under the key): the OpenAI-compatible listing carries each
// model's id, and OpenRouter's adds its name and context length. The
// provider's endpoint and key variable are all the listing needs; its
// key's absence refuses the listing by the variable's name.
func ListModels(ctx context.Context, provider remoteprovider.Provider, client *http.Client) ([]Listed, error) {
	key, err := remoteprovider.Key(provider)
	if err != nil {
		return nil, err
	}
	client = cmp.Or(client, http.DefaultClient)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, provider.Endpoint+modelsPath, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+key)
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxListingBytes))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("remote relay: %s answered %d to the model listing: %s", provider.Endpoint, response.StatusCode, strings.TrimSpace(string(body)))
	}
	var listing struct {
		Data []Listed `json:"data"`
	}
	if err := json.Unmarshal(body, &listing); err != nil {
		return nil, fmt.Errorf("remote relay: %s model listing: %w", provider.Endpoint, err)
	}
	models := make([]Listed, 0, len(listing.Data))
	for _, model := range listing.Data {
		if strings.TrimSpace(model.ID) != "" {
			models = append(models, model)
		}
	}
	return models, nil
}
