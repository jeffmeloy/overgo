package tensor

// Compatible reports whether two tensors have the same storage type and shape.
func Compatible(left, right *Tensor) bool {
	return left != nil && right != nil && left.Type == right.Type && left.Shape.Equal(right.Shape)
}
