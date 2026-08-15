package recipe

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/textcheck"
)

type Catalog struct {
	modules map[ModuleID]Module
}

func NewCatalog(modules ...Module) (*Catalog, error) {
	catalog := &Catalog{modules: make(map[ModuleID]Module, len(modules))}
	for _, module := range modules {
		if err := catalog.register(module); err != nil {
			return nil, err
		}
	}
	return catalog, nil
}

func (c *Catalog) register(module Module) error {
	if c == nil {
		return errors.New("recipe: nil module catalog")
	}
	canonical, err := canonicalModule(module)
	if err != nil {
		return err
	}
	if c.modules == nil {
		c.modules = map[ModuleID]Module{}
	}
	if _, duplicate := c.modules[canonical.ID]; duplicate {
		return fmt.Errorf("recipe: duplicate module %q", canonical.ID)
	}
	c.modules[canonical.ID] = canonical
	return nil
}

func (c *Catalog) Module(id ModuleID) (Module, bool) {
	if c == nil {
		return Module{}, false
	}
	module, ok := c.modules[id]
	return cloneModule(module), ok
}

func (c *Catalog) Modules() []Module {
	if c == nil {
		return nil
	}
	result := make([]Module, 0, len(c.modules))
	for _, module := range c.modules {
		result = append(result, cloneModule(module))
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}

func canonicalModule(module Module) (Module, error) {
	if !textcheck.LowerIdentifier(string(module.ID), maxName) || len(module.Tasks) == 0 || len(module.Placements) == 0 {
		return Module{}, errors.New("recipe: invalid module identity or policy")
	}
	result := cloneModule(module)
	for _, task := range result.Tasks {
		if err := validateTask(task); err != nil {
			return Module{}, err
		}
	}
	for _, placement := range result.Placements {
		if err := validatePlacement(placement); err != nil {
			return Module{}, err
		}
	}
	sort.Slice(result.Tasks, func(i, j int) bool { return result.Tasks[i] < result.Tasks[j] })
	sort.Slice(result.Placements, func(i, j int) bool { return result.Placements[i] < result.Placements[j] })
	result.Tasks = slices.Compact(result.Tasks)
	result.Placements = slices.Compact(result.Placements)
	inputs, err := canonicalPorts(result.Inputs)
	if err != nil {
		return Module{}, fmt.Errorf("recipe: module %q inputs: %w", module.ID, err)
	}
	outputs, err := canonicalPorts(result.Outputs)
	if err != nil {
		return Module{}, fmt.Errorf("recipe: module %q outputs: %w", module.ID, err)
	}
	result.Inputs, result.Outputs = inputs, outputs
	return result, nil
}

func canonicalPorts(ports []Port) ([]Port, error) {
	result := slices.Clone(ports)
	for _, port := range result {
		if err := port.validate(); err != nil {
			return nil, err
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	for index := 1; index < len(result); index++ {
		if result[index-1].Name == result[index].Name {
			return nil, fmt.Errorf("duplicate port %q", result[index].Name)
		}
	}
	return result, nil
}

func cloneModule(module Module) Module {
	module.Tasks = slices.Clone(module.Tasks)
	module.Placements = slices.Clone(module.Placements)
	module.Inputs = slices.Clone(module.Inputs)
	module.Outputs = slices.Clone(module.Outputs)
	return module
}
