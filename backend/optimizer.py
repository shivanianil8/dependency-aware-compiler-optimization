"""
backend/optimizer.py - Dependency-Aware Compiler Optimization

This module implements:
1. Lexer and recursive-descent parser for a simple imperative language subset.
2. AST (Abstract Syntax Tree) representation and code generator.
3. Semantic validator (detects undefined variables, constant division-by-zero).
4. Three standalone optimization passes:
   - Constant Folding: evaluates purely constant arithmetic expressions.
   - Algebraic Simplification: simplifies identity rules (e.g., x + 0, x * 1, x * 0).
   - Dead Code Elimination: removes unused variable assignments based on backward liveness.
5. Detection and execution utilities for optimization pass pipelines.

Note:
- Integer division (`/`) uses truncated-towards-zero semantics (matching C/C++/Java).
- Python's `eval()` is strictly NEVER used.
- Pass dependencies are NOT hardcoded here; passes execute in whichever order the caller specifies.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any, Union, Tuple


# ============================================================================
# 1. Custom Exceptions
# ============================================================================

class OptimizerError(Exception):
    """Base exception for all compiler and optimizer errors."""
    pass


class CompilerSyntaxError(OptimizerError):
    """Raised when input code contains invalid tokens or grammar."""
    pass


class CompilerSemanticError(OptimizerError):
    """Raised when code violates semantic rules (e.g., undefined variable)."""
    pass


class OptimizationError(OptimizerError):
    """Raised during optimization evaluation (e.g., division by zero)."""
    pass


# ============================================================================
# 2. Abstract Syntax Tree (AST) Nodes
# ============================================================================

class ASTNode:
    """Base class for all AST nodes."""
    pass


class ExprNode(ASTNode):
    """Base class for expression nodes."""
    pass


@dataclass
class Num(ExprNode):
    """Represents an integer literal."""
    value: int

    def __repr__(self) -> str:
        return f"Num({self.value})"


@dataclass
class Var(ExprNode):
    """Represents a variable identifier."""
    name: str

    def __repr__(self) -> str:
        return f"Var('{self.name}')"


@dataclass
class UnaryOp(ExprNode):
    """Represents a unary operation (e.g., -x, +5)."""
    op: str
    operand: ExprNode

    def __repr__(self) -> str:
        return f"UnaryOp('{self.op}', {self.operand})"


@dataclass
class BinOp(ExprNode):
    """Represents a binary operation (e.g., a + b, x * y)."""
    op: str
    left: ExprNode
    right: ExprNode

    def __repr__(self) -> str:
        return f"BinOp('{self.op}', {self.left}, {self.right})"


class Statement(ASTNode):
    """Base class for statement nodes."""
    def to_code(self) -> str:
        raise NotImplementedError


@dataclass
class Assignment(Statement):
    """Represents an assignment statement: target = value;"""
    target: str
    value: ExprNode

    def to_code(self) -> str:
        return f"{self.target} = {expr_to_code(self.value)};"

    def __repr__(self) -> str:
        return f"Assignment('{self.target}', {self.value})"


@dataclass
class Print(Statement):
    """Represents a print statement: print(variable);"""
    variable: str

    def to_code(self) -> str:
        return f"print({self.variable});"

    def __repr__(self) -> str:
        return f"Print('{self.variable}')"


@dataclass
class Program(ASTNode):
    """Represents an ordered list of statements forming a program."""
    statements: List[Statement]

    def to_code(self) -> str:
        return "\n".join(stmt.to_code() for stmt in self.statements)

    def __repr__(self) -> str:
        return f"Program({self.statements})"


# ============================================================================
# 3. Code Generation Helper (AST -> Source Code)
# ============================================================================

def expr_to_code(expr: ExprNode) -> str:
    """
    Converts an expression AST node back to readable source code,
    adding parentheses where necessary to preserve operator precedence.
    """
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}

    if isinstance(expr, Num):
        return str(expr.value)

    elif isinstance(expr, Var):
        return expr.name

    elif isinstance(expr, UnaryOp):
        inner = expr_to_code(expr.operand)
        if isinstance(expr.operand, BinOp):
            return f"{expr.op}({inner})"
        return f"{expr.op}{inner}"

    elif isinstance(expr, BinOp):
        left_str = expr_to_code(expr.left)
        right_str = expr_to_code(expr.right)

        # Parenthesize left child if it has strictly lower precedence
        if isinstance(expr.left, BinOp) and prec[expr.left.op] < prec[expr.op]:
            left_str = f"({left_str})"

        # Parenthesize right child if lower precedence, or equal precedence for non-associative ops
        if isinstance(expr.right, BinOp):
            if prec[expr.right.op] < prec[expr.op] or (prec[expr.right.op] == prec[expr.op] and expr.op in ('-', '/')):
                right_str = f"({right_str})"
        elif isinstance(expr.right, Num) and expr.right.value < 0:
            right_str = f"({right_str})"

        return f"{left_str} {expr.op} {right_str}"

    raise OptimizerError(f"Unknown expression node type: {type(expr)}")


# ============================================================================
# 4. Lexer (Tokenizer)
# ============================================================================

@dataclass
class Token:
    type: str  # 'NUMBER', 'IDENT', 'PRINT', 'ASSIGN', 'PLUS', 'MINUS', 'STAR', 'SLASH', 'LPAREN', 'RPAREN', 'SEMI', 'EOF'
    value: Any
    line: int
    column: int


class Lexer:
    """Tokenizes source code into a stream of tokens."""

    def __init__(self, code: str):
        self.code = code
        self.pos = 0
        self.line = 1
        self.column = 1
        self.length = len(code)

    def _peek(self) -> Optional[str]:
        if self.pos < self.length:
            return self.code[self.pos]
        return None

    def _advance(self) -> Optional[str]:
        char = self._peek()
        if char is not None:
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return char

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # Skip whitespace
            if ch in (' ', '\t', '\r', '\n'):
                self._advance()
                continue

            # Skip single-line comments (# or //)
            if ch == '#':
                while self._peek() is not None and self._peek() != '\n':
                    self._advance()
                continue

            if ch == '/' and self.pos + 1 < self.length and self.code[self.pos + 1] == '/':
                while self._peek() is not None and self._peek() != '\n':
                    self._advance()
                continue

            start_line = self.line
            start_col = self.column

            # Integer numbers
            if ch.isdigit():
                num_str = ""
                while self._peek() is not None and self._peek().isdigit():
                    num_str += self._advance()
                tokens.append(Token('NUMBER', int(num_str), start_line, start_col))
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == '_':
                ident_str = ""
                while self._peek() is not None and (self._peek().isalnum() or self._peek() == '_'):
                    ident_str += self._advance()

                if ident_str == 'print':
                    tokens.append(Token('PRINT', 'print', start_line, start_col))
                else:
                    tokens.append(Token('IDENT', ident_str, start_line, start_col))
                continue

            # Single-character tokens
            char_map = {
                '=': 'ASSIGN',
                '+': 'PLUS',
                '-': 'MINUS',
                '*': 'STAR',
                '/': 'SLASH',
                '(': 'LPAREN',
                ')': 'RPAREN',
                ';': 'SEMI',
            }

            if ch in char_map:
                tokens.append(Token(char_map[ch], ch, start_line, start_col))
                self._advance()
                continue

            raise CompilerSyntaxError(
                f"Unexpected character '{ch}' at line {start_line}, column {start_col}."
            )

        tokens.append(Token('EOF', '', self.line, self.column))
        return tokens


# ============================================================================
# 5. Recursive Descent Parser
# ============================================================================

class Parser:
    """
    Parses a token stream into an AST Program.

    Grammar:
      program     ::= (statement)* EOF
      statement   ::= assignment | print_stmt
      assignment  ::= IDENT '=' expr ';'
      print_stmt  ::= 'print' '(' IDENT ')' ';'
      expr        ::= term (('+' | '-') term)*
      term        ::= factor (('*' | '/') factor)*
      factor      ::= ('+' | '-') factor | primary
      primary     ::= NUMBER | IDENT | '(' expr ')'
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _peek_type(self) -> str:
        return self._current().type

    def _match(self, expected_type: str) -> bool:
        if self._peek_type() == expected_type:
            self.pos += 1
            return True
        return False

    def _expect(self, expected_type: str, context_message: str = "") -> Token:
        token = self._current()
        if token.type != expected_type:
            ctx = f" while {context_message}" if context_message else ""
            raise CompilerSyntaxError(
                f"Syntax Error at line {token.line}, column {token.column}: "
                f"Expected '{expected_type}', found '{token.type}' ('{token.value}'){ctx}."
            )
        self.pos += 1
        return token

    def parse(self) -> Program:
        statements: List[Statement] = []
        while self._peek_type() != 'EOF':
            statements.append(self._parse_statement())
        return Program(statements=statements)

    def _parse_statement(self) -> Statement:
        tok = self._current()

        if tok.type == 'PRINT':
            return self._parse_print()
        elif tok.type == 'IDENT':
            return self._parse_assignment()
        else:
            raise CompilerSyntaxError(
                f"Syntax Error at line {tok.line}, column {tok.column}: "
                f"Unexpected token '{tok.value}'. Expected assignment (e.g. 'x = 10;') "
                f"or print statement (e.g. 'print(x);')."
            )

    def _parse_print(self) -> Print:
        self._expect('PRINT', "parsing print statement")
        self._expect('LPAREN', "opening print argument")
        var_token = self._expect('IDENT', "reading variable inside print(...)")
        self._expect('RPAREN', "closing print statement")
        self._expect('SEMI', "terminating print statement with ';'")
        return Print(variable=var_token.value)

    def _parse_assignment(self) -> Assignment:
        target_token = self._expect('IDENT', "reading assignment variable")
        self._expect('ASSIGN', f"reading '=' in assignment to '{target_token.value}'")
        value_expr = self._parse_expr()
        self._expect('SEMI', f"terminating assignment to '{target_token.value}' with ';'")
        return Assignment(target=target_token.value, value=value_expr)

    def _parse_expr(self) -> ExprNode:
        node = self._parse_term()
        while self._peek_type() in ('PLUS', 'MINUS'):
            op_tok = self._current()
            self._match(op_tok.type)
            right = self._parse_term()
            node = BinOp(op=op_tok.value, left=node, right=right)
        return node

    def _parse_term(self) -> ExprNode:
        node = self._parse_factor()
        while self._peek_type() in ('STAR', 'SLASH'):
            op_tok = self._current()
            self._match(op_tok.type)
            right = self._parse_factor()
            node = BinOp(op=op_tok.value, left=node, right=right)
        return node

    def _parse_factor(self) -> ExprNode:
        tok = self._current()
        if tok.type in ('PLUS', 'MINUS'):
            self._match(tok.type)
            operand = self._parse_factor()
            return UnaryOp(op=tok.value, operand=operand)
        return self._parse_primary()

    def _parse_primary(self) -> ExprNode:
        tok = self._current()

        if tok.type == 'NUMBER':
            self._match('NUMBER')
            return Num(value=tok.value)

        elif tok.type == 'IDENT':
            self._match('IDENT')
            return Var(name=tok.value)

        elif tok.type == 'LPAREN':
            self._match('LPAREN')
            expr = self._parse_expr()
            self._expect('RPAREN', "closing parenthesized expression")
            return expr

        raise CompilerSyntaxError(
            f"Syntax Error at line {tok.line}, column {tok.column}: "
            f"Unexpected token '{tok.value}' in expression."
        )


# ============================================================================
# 6. Semantic Analysis and Validation
# ============================================================================

def safe_int_div(a: int, b: int) -> int:
    """
    Computes integer division truncated towards zero (C99/C++/Java semantics).
    Examples:
        safe_int_div(7, 2)   ==  3
        safe_int_div(-7, 2)  == -3
        safe_int_div(7, -2)  == -3
        safe_int_div(-7, -2) ==  3
    Raises ZeroDivisionError if b == 0.
    """
    if b == 0:
        raise ZeroDivisionError("Integer division by zero")
    # Exact integer division truncated toward zero:
    if (a < 0) ^ (b < 0):
        return -(abs(a) // abs(b))
    return abs(a) // abs(b)


def get_variables_used(expr: ExprNode) -> Set[str]:
    """Returns the set of variable names referenced in an expression."""
    if isinstance(expr, Num):
        return set()
    elif isinstance(expr, Var):
        return {expr.name}
    elif isinstance(expr, UnaryOp):
        return get_variables_used(expr.operand)
    elif isinstance(expr, BinOp):
        return get_variables_used(expr.left) | get_variables_used(expr.right)
    return set()


def validate_program(program: Program) -> None:
    """
    Validates that:
    1. Every variable used is defined before its use.
    2. Expressions do not contain static division by zero.
    """
    defined_vars: Set[str] = set()

    for idx, stmt in enumerate(program.statements, start=1):
        if isinstance(stmt, Assignment):
            used_vars = get_variables_used(stmt.value)
            for var_name in used_vars:
                if var_name not in defined_vars:
                    raise CompilerSemanticError(
                        f"Semantic Error in statement #{idx} ('{stmt.to_code()}'): "
                        f"Variable '{var_name}' is used before assignment."
                    )
            # Check for constant division-by-zero in expression
            _check_constant_div_zero(stmt.value, idx)
            defined_vars.add(stmt.target)

        elif isinstance(stmt, Print):
            if stmt.variable not in defined_vars:
                raise CompilerSemanticError(
                    f"Semantic Error in statement #{idx} ('{stmt.to_code()}'): "
                    f"Variable '{stmt.variable}' is printed before assignment."
                )


def _check_constant_div_zero(expr: ExprNode, stmt_idx: int) -> None:
    """Recursively checks for division by zero in constant subexpressions."""
    if isinstance(expr, BinOp):
        _check_constant_div_zero(expr.left, stmt_idx)
        _check_constant_div_zero(expr.right, stmt_idx)
        if expr.op == '/':
            if is_constant_expr(expr.right):
                try:
                    val = eval_constant_expr(expr.right)
                    if val == 0:
                        raise CompilerSemanticError(
                            f"Semantic Error in statement #{idx_str(stmt_idx)}: "
                            f"Division by zero detected in expression '{expr_to_code(expr)}'."
                        )
                except ZeroDivisionError:
                    raise CompilerSemanticError(
                        f"Semantic Error in statement #{idx_str(stmt_idx)}: "
                        f"Division by zero detected in expression '{expr_to_code(expr)}'."
                    )


def idx_str(idx: int) -> str:
    return str(idx)


# ============================================================================
# 7. Constant Expression Evaluation (No eval() used)
# ============================================================================

def is_constant_expr(expr: ExprNode) -> bool:
    """Returns True if the expression contains only constants and operators (no variables)."""
    if isinstance(expr, Num):
        return True
    elif isinstance(expr, Var):
        return False
    elif isinstance(expr, UnaryOp):
        return is_constant_expr(expr.operand)
    elif isinstance(expr, BinOp):
        return is_constant_expr(expr.left) and is_constant_expr(expr.right)
    return False


def eval_constant_expr(expr: ExprNode) -> int:
    """
    Evaluates an expression tree containing purely constant values.
    Never uses Python's eval().
    """
    if isinstance(expr, Num):
        return expr.value

    elif isinstance(expr, UnaryOp):
        val = eval_constant_expr(expr.operand)
        if expr.op == '+':
            return val
        elif expr.op == '-':
            return -val
        raise OptimizationError(f"Unsupported unary operator: {expr.op}")

    elif isinstance(expr, BinOp):
        left_val = eval_constant_expr(expr.left)
        right_val = eval_constant_expr(expr.right)

        if expr.op == '+':
            return left_val + right_val
        elif expr.op == '-':
            return left_val - right_val
        elif expr.op == '*':
            return left_val * right_val
        elif expr.op == '/':
            if right_val == 0:
                raise OptimizationError("Division by zero in constant expression")
            return safe_int_div(left_val, right_val)
        raise OptimizationError(f"Unsupported binary operator: {expr.op}")

    raise OptimizationError(f"Cannot evaluate non-constant node of type {type(expr)}")


# ============================================================================
# 8. Optimization Pass 1: Constant Folding
# ============================================================================

def _fold_expr(expr: ExprNode) -> Tuple[ExprNode, bool]:
    """
    Recursively folds constant subexpressions bottom-up.
    Returns (new_expr, changed_flag).
    """
    if isinstance(expr, (Num, Var)):
        return expr, False

    elif isinstance(expr, UnaryOp):
        new_operand, child_changed = _fold_expr(expr.operand)
        if is_constant_expr(new_operand):
            val = eval_constant_expr(UnaryOp(expr.op, new_operand))
            return Num(val), True
        return UnaryOp(expr.op, new_operand), child_changed

    elif isinstance(expr, BinOp):
        new_left, left_changed = _fold_expr(expr.left)
        new_right, right_changed = _fold_expr(expr.right)
        child_changed = left_changed or right_changed

        if is_constant_expr(new_left) and is_constant_expr(new_right):
            val = eval_constant_expr(BinOp(expr.op, new_left, new_right))
            return Num(val), True

        return BinOp(expr.op, new_left, new_right), child_changed

    return expr, False


def constant_folding_pass(program: Program) -> Tuple[Program, List[Dict[str, Any]]]:
    """
    Applies Constant Folding to all statements in the program.
    Evaluates expressions composed entirely of constants.
    """
    new_statements: List[Statement] = []
    log: List[Dict[str, Any]] = []

    for stmt in program.statements:
        if isinstance(stmt, Assignment):
            new_val, changed = _fold_expr(stmt.value)
            if changed:
                new_stmt = Assignment(target=stmt.target, value=new_val)
                log.append({
                    "pass_name": "Constant Folding",
                    "original_statement": stmt.to_code(),
                    "replacement": new_stmt.to_code(),
                    "details": f"Folded constant expression in assignment to '{stmt.target}'"
                })
                new_statements.append(new_stmt)
            else:
                new_statements.append(stmt)
        else:
            new_statements.append(stmt)

    return Program(statements=new_statements), log


# ============================================================================
# 9. Optimization Pass 2: Algebraic Simplification
# ============================================================================

def _simplify_node_identities(expr: ExprNode) -> Tuple[ExprNode, bool]:
    """
    Simplifies known integer identities on a single node:
    - x + 0 -> x,  0 + x -> x
    - x - 0 -> x
    - x * 1 -> x,  1 * x -> x
    - x * 0 -> 0,  0 * x -> 0
    - x / 1 -> x
    """
    if isinstance(expr, BinOp):
        # Addition with zero
        if expr.op == '+' and isinstance(expr.right, Num) and expr.right.value == 0:
            return expr.left, True
        if expr.op == '+' and isinstance(expr.left, Num) and expr.left.value == 0:
            return expr.right, True

        # Subtraction with zero
        if expr.op == '-' and isinstance(expr.right, Num) and expr.right.value == 0:
            return expr.left, True

        # Multiplication with one
        if expr.op == '*' and isinstance(expr.right, Num) and expr.right.value == 1:
            return expr.left, True
        if expr.op == '*' and isinstance(expr.left, Num) and expr.left.value == 1:
            return expr.right, True

        # Multiplication with zero
        if expr.op == '*' and isinstance(expr.right, Num) and expr.right.value == 0:
            return Num(0), True
        if expr.op == '*' and isinstance(expr.left, Num) and expr.left.value == 0:
            return Num(0), True

        # Division by one
        if expr.op == '/' and isinstance(expr.right, Num) and expr.right.value == 1:
            return expr.left, True

    return expr, False


def _simplify_expr_recursively(expr: ExprNode) -> Tuple[ExprNode, bool]:
    """Recursively traverses expressions and applies algebraic identities."""
    if isinstance(expr, (Num, Var)):
        return expr, False

    elif isinstance(expr, UnaryOp):
        new_operand, changed = _simplify_expr_recursively(expr.operand)
        return UnaryOp(expr.op, new_operand), changed

    elif isinstance(expr, BinOp):
        new_left, left_changed = _simplify_expr_recursively(expr.left)
        new_right, right_changed = _simplify_expr_recursively(expr.right)
        current = BinOp(expr.op, new_left, new_right)
        changed = left_changed or right_changed

        # Apply identity simplification iteratively at this node
        while True:
            simplified, rule_applied = _simplify_node_identities(current)
            if rule_applied:
                current = simplified
                changed = True
            else:
                break

        return current, changed

    return expr, False


def algebraic_simplification_pass(program: Program) -> Tuple[Program, List[Dict[str, Any]]]:
    """
    Applies Algebraic Simplification to simplify arithmetic identities
    such as x + 0, x - 0, x * 1, x * 0, and x / 1.
    """
    new_statements: List[Statement] = []
    log: List[Dict[str, Any]] = []

    for stmt in program.statements:
        if isinstance(stmt, Assignment):
            new_val, changed = _simplify_expr_recursively(stmt.value)
            if changed:
                new_stmt = Assignment(target=stmt.target, value=new_val)
                log.append({
                    "pass_name": "Algebraic Simplification",
                    "original_statement": stmt.to_code(),
                    "replacement": new_stmt.to_code(),
                    "details": f"Simplified algebraic identity in assignment to '{stmt.target}'"
                })
                new_statements.append(new_stmt)
            else:
                new_statements.append(stmt)
        else:
            new_statements.append(stmt)

    return Program(statements=new_statements), log


# ============================================================================
# 10. Optimization Pass 3: Dead Code Elimination
# ============================================================================

def dead_code_elimination_pass(program: Program) -> Tuple[Program, List[Dict[str, Any]]]:
    """
    Removes unused variable assignments using backward liveness analysis.

    Semantics:
    1. Print statements are side-effectful and always preserved. The printed variable
       is marked live.
    2. An assignment target = expr is LIVE if 'target' is currently needed.
       When kept:
         - 'target' is removed from needed variables (since this statement defines it).
         - All variables used in 'expr' are added to needed variables.
    3. If 'target' is NOT needed, the assignment is eliminated as dead code.
    4. If the program has NO print statements, the final assignment to each variable
       is treated as an observable output to prevent deleting the whole program.
    """
    statements = program.statements
    has_prints = any(isinstance(stmt, Print) for stmt in statements)

    live_vars: Set[str] = set()
    if not has_prints:
        # Fallback when no prints exist: mark the last assignment to each variable as live
        seen_targets: Set[str] = set()
        for stmt in reversed(statements):
            if isinstance(stmt, Assignment):
                if stmt.target not in seen_targets:
                    live_vars.add(stmt.target)
                    seen_targets.add(stmt.target)

    new_statements_rev: List[Statement] = []
    log_rev: List[Dict[str, Any]] = []

    for stmt in reversed(statements):
        if isinstance(stmt, Print):
            new_statements_rev.append(stmt)
            live_vars.add(stmt.variable)

        elif isinstance(stmt, Assignment):
            target = stmt.target
            if target in live_vars:
                new_statements_rev.append(stmt)
                # target is satisfied by this definition
                live_vars.remove(target)
                # variables used in the RHS now become required
                live_vars.update(get_variables_used(stmt.value))
            else:
                log_rev.append({
                    "pass_name": "Dead Code Elimination",
                    "original_statement": stmt.to_code(),
                    "replacement": None,
                    "details": f"Removed dead assignment to '{target}' (value is never used)"
                })
        else:
            new_statements_rev.append(stmt)

    new_statements = list(reversed(new_statements_rev))
    log = list(reversed(log_rev))
    return Program(statements=new_statements), log


# ============================================================================
# 11. Pass Registry and Applicability Detection
# ============================================================================

PASS_REGISTRY = {
    "Constant Folding": constant_folding_pass,
    "Algebraic Simplification": algebraic_simplification_pass,
    "Dead Code Elimination": dead_code_elimination_pass,
}

# Alias mapping for flexible case-insensitive lookups
PASS_ALIASES = {
    "constant folding": "Constant Folding",
    "constant_folding": "Constant Folding",
    "cf": "Constant Folding",
    "algebraic simplification": "Algebraic Simplification",
    "algebraic_simplification": "Algebraic Simplification",
    "as": "Algebraic Simplification",
    "dead code elimination": "Dead Code Elimination",
    "dead_code_elimination": "Dead Code Elimination",
    "dce": "Dead Code Elimination",
}


def normalize_pass_name(name: str) -> str:
    """Normalizes pass name string to canonical form."""
    clean = name.strip()
    if clean in PASS_REGISTRY:
        return clean
    lower = clean.lower()
    if lower in PASS_ALIASES:
        return PASS_ALIASES[lower]
    valid = list(PASS_REGISTRY.keys())
    raise ValueError(f"Unknown optimization pass: '{name}'. Valid passes are: {valid}")


def parse_program(code: str, validate: bool = True) -> Program:
    """
    Parses source code into an AST Program.
    Validates semantics by default.
    """
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    if validate:
        validate_program(program)
    return program


def detect_applicable_passes(code_or_ast: Union[str, Program]) -> List[str]:
    """
    Inspects code/AST and returns the list of pass names that can perform
    at least one optimization transformation on the program.
    """
    if isinstance(code_or_ast, str):
        program = parse_program(code_or_ast, validate=True)
    else:
        program = code_or_ast

    applicable: List[str] = []

    for pass_name, pass_func in PASS_REGISTRY.items():
        # Test on a deep copy of the AST to preserve original structure
        test_ast = copy.deepcopy(program)
        _, test_log = pass_func(test_ast)
        if len(test_log) > 0:
            applicable.append(pass_name)

    return applicable


def run_optimization_passes(
    code_or_ast: Union[str, Program],
    pass_order: List[str]
) -> Dict[str, Any]:
    """
    Executes a specified sequence of optimization passes on the input program.

    Parameters:
        code_or_ast: Source code string or parsed Program AST.
        pass_order: List of pass names to execute in order.

    Returns:
        Structured dictionary with:
        - original_code: original program string
        - optimized_code: optimized program string
        - applicable_passes: passes applicable before running
        - remaining_applicable_passes: passes applicable after running
        - executed_passes: sequence of passes that were executed
        - transformation_log: list of transformation records
        - stats: summary statistics
    """
    if isinstance(code_or_ast, str):
        original_code_str = code_or_ast
        current_program = parse_program(code_or_ast, validate=True)
    else:
        current_program = copy.deepcopy(code_or_ast)
        original_code_str = current_program.to_code()

    # Detect passes applicable initially
    initial_applicable = detect_applicable_passes(current_program)

    canonical_order = [normalize_pass_name(p) for p in pass_order]
    cumulative_log: List[Dict[str, Any]] = []

    # Execute passes strictly in the user-specified order
    for pass_name in canonical_order:
        pass_func = PASS_REGISTRY[pass_name]
        current_program, pass_log = pass_func(current_program)
        cumulative_log.extend(pass_log)

    optimized_code_str = current_program.to_code()
    remaining_applicable = detect_applicable_passes(current_program)

    return {
        "original_code": original_code_str,
        "optimized_code": optimized_code_str,
        "applicable_passes": initial_applicable,
        "remaining_applicable_passes": remaining_applicable,
        "executed_passes": canonical_order,
        "transformation_log": cumulative_log,
        "stats": {
            "transformations_count": len(cumulative_log),
            "original_statement_count": len(original_code_str.strip().split('\n')) if original_code_str.strip() else 0,
            "optimized_statement_count": len(current_program.statements),
        }
    }


# ============================================================================
# 12. Demonstration and Self-Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dependency-Aware Compiler Optimization - Optimizer Self-Tests")
    print("=" * 60)

    # Test 1: Constant Folding
    print("\n--- Test 1: Constant Folding ---")
    code1 = """
    x = 2 + 3 * 4;
    y = (10 - 2) / 4;
    print(x);
    print(y);
    """
    res1 = run_optimization_passes(code1, ["Constant Folding"])
    print("Optimized Code:\n" + res1["optimized_code"])
    print("Log:", res1["transformation_log"])

    # Test 2: Algebraic Simplification
    print("\n--- Test 2: Algebraic Simplification ---")
    code2 = """
    x = 10;
    a = x + 0;
    b = x * 1;
    c = x * 0;
    d = x - 0;
    print(a);
    print(b);
    print(c);
    print(d);
    """
    res2 = run_optimization_passes(code2, ["Algebraic Simplification"])
    print("Optimized Code:\n" + res2["optimized_code"])
    print("Log:", res2["transformation_log"])

    # Test 3: Dead Code Elimination
    print("\n--- Test 3: Dead Code Elimination ---")
    code3 = """
    unused_1 = 100;
    unused_2 = unused_1 + 50;
    live_var = 5;
    result = live_var + 10;
    print(result);
    """
    res3 = run_optimization_passes(code3, ["Dead Code Elimination"])
    print("Optimized Code:\n" + res3["optimized_code"])
    print("Log:", res3["transformation_log"])

    # Test 4: Combined Multi-Pass Optimization
    print("\n--- Test 4: Pipeline Execution ---")
    code4 = """
    x = 4 + 6;
    y = x * 1;
    dead_val = 999;
    z = y + 0;
    print(z);
    """
    order = ["Constant Folding", "Algebraic Simplification", "Dead Code Elimination"]
    res4 = run_optimization_passes(code4, order)
    print("Initial Applicable:", res4["applicable_passes"])
    print("Optimized Code:\n" + res4["optimized_code"])
    print("Transformations Count:", res4["stats"]["transformations_count"])
    for item in res4["transformation_log"]:
        print(f" - [{item['pass_name']}] {item['original_statement']} -> {item['replacement']}")

    # Test 5: Reassignment / Dead Store Elimination
    print("\n--- Test 5: Dead Store Overwriting ---")
    code5 = """
    x = 10;
    x = 20;
    print(x);
    """
    res5 = run_optimization_passes(code5, ["Dead Code Elimination"])
    print("Optimized Code:\n" + res5["optimized_code"])
    print("Log:", res5["transformation_log"])

    # Test 6: Syntax and Semantic Error Rejection
    print("\n--- Test 6: Error Handling ---")
    try:
        parse_program("x = 10 + ;")
    except CompilerSyntaxError as e:
        print("[PASSED] Caught expected syntax error:", e)

    try:
        parse_program("print(unassigned_var);")
    except CompilerSemanticError as e:
        print("[PASSED] Caught expected semantic error:", e)

    try:
        parse_program("x = 10 / 0;")
    except CompilerSemanticError as e:
        print("[PASSED] Caught expected constant zero division error:", e)

    print("\n" + "=" * 60)
    print("All basic optimizer tests completed successfully!")
    print("=" * 60)
