package discovery

import (
	"context"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/pathidentity"
)

// ExecutionDirectory names the directory a model's capability runs from:
// the most specific of its live recorded directories. A composite whose
// inventory roots at a parent it shares with a sibling repository (LiveEdit
// beside the Wan base it conditions on) records its own directory inside
// that parent too, and resolves only from it; any other model runs from the
// first live directory, as artifact.AvailablePath reads it, a recorded file
// standing for the directory that holds it.
func ExecutionDirectory(ctx context.Context, reader artifact.Reader, model artifact.ID) (string, error) {
	locations, err := reader.Locations(ctx, model)
	if err != nil {
		return "", err
	}
	directory := ""
	for _, location := range locations {
		path := location.Value
		if location.Kind == artifact.LocationFile {
			path = filepath.Dir(path)
		} else if location.Kind != artifact.LocationDirectory {
			continue
		}
		live, err := artifact.CanonicalLocalLocation(model, artifact.LocationDirectory, path)
		if err != nil {
			continue
		}
		if directory == "" {
			directory = live.Value
		} else if nested, _ := pathidentity.Contains(directory, live.Value); nested {
			directory = live.Value
		}
	}
	if directory == "" {
		return "", fmt.Errorf("discovery: %s has no live recorded directory", model)
	}
	return directory, nil
}
