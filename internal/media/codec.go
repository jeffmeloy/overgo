package media

import "fmt"

// CodecOperator: typed media-codec stage.
type CodecOperator uint8

const (
	CodecOperatorNone CodecOperator = iota
	CodecPointwise
	CodecConvolution
	CodecResidual
	CodecAttention
	CodecDownsampleSpatial
	CodecDownsampleSpatiotemporal
	CodecUpsampleSpatial
	CodecUpsampleSpatiotemporal
	CodecHead
)

// CodecOperation: compiled stage plus runtime-owned bindings.
type CodecOperation[Bindings any] struct {
	Operator                      CodecOperator
	Name                          string
	InputChannels, OutputChannels int
	BindingCount                  int
	Bindings                      Bindings
}

// CodecProgram: ordered validated codec topology.
type CodecProgram[Bindings any] struct {
	Operations []CodecOperation[Bindings]
}

func (p CodecProgram[Bindings]) Validate(scope string) error {
	if len(p.Operations) == 0 {
		return fmt.Errorf("%s: operations absent", scope)
	}
	for index, operation := range p.Operations {
		if operation.Operator == CodecOperatorNone || operation.Operator > CodecHead || operation.BindingCount <= 0 || operation.InputChannels <= 0 || operation.OutputChannels <= 0 {
			return fmt.Errorf("%s: operation %d (%s) is invalid", scope, index, operation.Name)
		}
		if index > 0 && operation.InputChannels != p.Operations[index-1].OutputChannels {
			return fmt.Errorf("%s: operation %d (%s) input channels=%d, previous output=%d",
				scope, index, operation.Name, operation.InputChannels, p.Operations[index-1].OutputChannels)
		}
	}
	return nil
}

func (p CodecProgram[Bindings]) Names() []string {
	names := make([]string, len(p.Operations))
	for index, operation := range p.Operations {
		names[index] = operation.Name
	}
	return names
}
