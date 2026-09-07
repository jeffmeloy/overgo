package closurescan

import (
	"go/ast"
	"go/token"
)

// IdentityRule is the name of a restated mathematical fact the scanner
// classifies without a triage row; every other literal stays with review.
type IdentityRule string

const (
	// IdentityDecimalRadix is base ten in a strconv base position.
	IdentityDecimalRadix IdentityRule = "decimal-radix"
	// IdentityBitSize is a Go numeric size in a strconv bit-size position.
	IdentityBitSize IdentityRule = "integer-bit-width"
	// IdentityZeroReset is zero assigned to an existing variable; the identity
	// of a count or delay.
	IdentityZeroReset IdentityRule = "zero-reset"
	// IdentityZeroResult is zero returned beside no error; the empty quantity.
	IdentityZeroResult IdentityRule = "zero-result"
	// IdentityUnitBound is zero or one bounding a builtin min or max; the unit
	// interval or the floor of a count.
	IdentityUnitBound IdentityRule = "unit-bound"
	// IdentityMathArgument is zero or one handed to a math function; the
	// library contract restated.
	IdentityMathArgument IdentityRule = "math-identity"
)

// strconvRadixLast: strconv functions whose base is the last argument.
var strconvRadixLast = map[string]bool{"FormatInt": true, "FormatUint": true, "AppendInt": true, "AppendUint": true}

// strconvRadixPenultimate: strconv functions whose base precedes the bit size.
var strconvRadixPenultimate = map[string]bool{"ParseInt": true, "ParseUint": true}

// strconvBitSizeLast: strconv functions whose bit size is the last argument.
var strconvBitSizeLast = map[string]bool{
	"ParseInt": true, "ParseUint": true, "ParseFloat": true, "ParseComplex": true,
	"FormatFloat": true, "FormatComplex": true, "AppendFloat": true, "AppendComplex": true,
}

// numericBitSizes: the bit sizes strconv accepts; zero selects the platform int.
var numericBitSizes = map[string]bool{"0": true, "8": true, "16": true, "32": true, "64": true, "128": true}

// lastArgument: the final call argument, nil for an empty call.
func lastArgument(args []ast.Expr) ast.Expr {
	if len(args) == 0 {
		return nil
	}
	return args[len(args)-1]
}

// penultimateArgument: the argument before the final one, nil below two.
func penultimateArgument(args []ast.Expr) ast.Expr {
	if len(args) <= 1 {
		return nil
	}
	return args[len(args)-2]
}

// identityRule: the rule a literal restates, or empty when review owns it.
// expression is the literal or its unary wrapper; only an unwrapped literal
// can be a zero or one identity.
func identityRule(literal *ast.BasicLit, expression ast.Expr, parent ast.Node) IdentityRule {
	unwrapped := expression == ast.Expr(literal)
	unit := unwrapped && (literal.Value == "0" || literal.Value == "1")
	switch typed := parent.(type) {
	case *ast.CallExpr:
		switch function := typed.Fun.(type) {
		case *ast.SelectorExpr:
			owner, ok := function.X.(*ast.Ident)
			if !ok {
				return ""
			}
			last, penultimate := lastArgument(typed.Args) == expression, penultimateArgument(typed.Args) == expression
			switch {
			case owner.Name == "strconv" && literal.Value == "10" &&
				(last && strconvRadixLast[function.Sel.Name] || penultimate && strconvRadixPenultimate[function.Sel.Name]):
				return IdentityDecimalRadix
			case owner.Name == "strconv" && last && strconvBitSizeLast[function.Sel.Name] && numericBitSizes[literal.Value]:
				return IdentityBitSize
			case (owner.Name == "math" || owner.Name == "cmplx") && unit:
				return IdentityMathArgument
			}
		case *ast.Ident:
			if unit && (function.Name == "min" || function.Name == "max") {
				return IdentityUnitBound
			}
		}
	case *ast.AssignStmt:
		if unwrapped && literal.Value == "0" && typed.Tok == token.ASSIGN {
			return IdentityZeroReset
		}
	case *ast.ReturnStmt:
		if unwrapped && literal.Value == "0" {
			return IdentityZeroResult
		}
	}
	return ""
}
