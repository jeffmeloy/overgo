// Package peft reads and writes Hugging Face PEFT LoRA adapters. An adapter
// directory holds adapter_config.json and adapter_model.safetensors; a module's
// weight changes by its scale times the product of its two low-rank factors,
// and Merge adds that change into a base safetensors checkpoint.
package peft

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"overgo/internal/hostmath"
)

const (
	// configFile names the config an adapter directory holds.
	configFile = "adapter_config.json"
	// weightsFile names the safetensors file of an adapter's factors.
	weightsFile = "adapter_model.safetensors"
)

// Config is the part of adapter_config.json that decides the adapter's
// arithmetic. Rank and Alpha are the defaults a module with no pattern of its
// own takes; the scale of a module is Alpha over its rank, or over the root of
// its rank when RSLoRA is set.
type Config struct {
	BaseModel     string
	Rank          int
	Alpha         float64
	Dropout       float64
	TargetModules []string
	RSLoRA        bool
}

// Module is one adapted linear layer's low-rank pair. Down is Rank×In and Up is
// Out×Rank, both row-major, so the layer's Out×In weight changes by its scale
// times Up·Down. Alpha is this module's own numerator of that scale.
type Module struct {
	Rank, In, Out int
	Alpha         float64
	Down, Up      []float32
}

// Adapter is a LoRA adapter: its config and its modules, each named by the
// Hugging Face module path whose ".weight" it adapts, such as
// "model.layers.0.self_attn.q_proj".
type Adapter struct {
	Config  Config
	Modules map[string]Module
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// Validate refuses an adapter whose factors do not fill their shapes, whose
// scales are not positive and finite, or whose values are not finite.
func (adapter Adapter) Validate() error {
	config := adapter.Config
	if config.Rank <= 0 || !finite(config.Alpha) || config.Alpha <= 0 || !finite(config.Dropout) || config.Dropout < 0 || config.Dropout >= 1 {
		return fmt.Errorf("peft: config needs a positive rank and alpha and a dropout in [0, 1), has rank %d, alpha %v, dropout %v", config.Rank, config.Alpha, config.Dropout)
	}
	if len(adapter.Modules) == 0 {
		return errors.New("peft: adapter has no modules")
	}
	for name, module := range adapter.Modules {
		if err := module.validate(name); err != nil {
			return err
		}
	}
	return nil
}

func (module Module) validate(name string) error {
	if name == "" || strings.ContainsAny(name, " \t\r\n") || strings.HasSuffix(name, ".weight") {
		return fmt.Errorf("peft: module name %q must be a module path, not a tensor name", name)
	}
	if module.Rank <= 0 || module.In <= 0 || module.Out <= 0 || len(module.Down) != module.Rank*module.In || len(module.Up) != module.Out*module.Rank {
		return fmt.Errorf("peft: module %q has rank %d, %d inputs and %d outputs, which its %d and %d factor values do not fill",
			name, module.Rank, module.In, module.Out, len(module.Down), len(module.Up))
	}
	if !finite(module.Alpha) || module.Alpha <= 0 {
		return fmt.Errorf("peft: module %q has alpha %v, want a positive finite number", name, module.Alpha)
	}
	for _, factor := range [][]float32{module.Down, module.Up} {
		for _, value := range factor {
			if !finite(float64(value)) {
				return fmt.Errorf("peft: module %q has a factor value that is not finite", name)
			}
		}
	}
	return nil
}

// scale is the factor a module's product enters its weight with.
func (adapter Adapter) scale(module Module) float64 {
	if adapter.Config.RSLoRA {
		return module.Alpha / math.Sqrt(float64(module.Rank))
	}
	return module.Alpha / float64(module.Rank)
}

// Delta returns the weight update of the module name: its scale times Up·Down,
// Out×In row-major.
func (adapter Adapter) Delta(name string) ([]float32, error) {
	module, found := adapter.Modules[name]
	if !found {
		return nil, fmt.Errorf("peft: adapter has no module %q", name)
	}
	if err := module.validate(name); err != nil {
		return nil, err
	}
	delta := make([]float32, module.Out*module.In)
	hostmath.Linear(delta, module.Up, hostmath.Transpose2D(module.Down, module.Rank, module.In), module.Out, module.Rank, module.In)
	scale := float32(adapter.scale(module))
	for index := range delta {
		delta[index] *= scale
	}
	return delta, nil
}
