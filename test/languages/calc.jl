module Calc

using Base.Test
using DebuggerFramework
using AbstractTrees

include("utilities.jl")

# This file implementes a simple calculator. It is meant to illustrate
# and test line-wise stepping, display of local variables and
# language-specic REPL prompts.

@enum(TokenKind,
    # Punctuation
    LPAREN,
    RPAREN,
    # Operators
    PLUS,
    MINUS,
    TIMES,
    DIV,
    # Literals
    INT,
    # Backreferences
    BACKREF,
    # Meta
    WHITESPACE,
    NEWLINE,
    ENDMARKER,
    ERROR,
)
is_op_kind(kind::TokenKind) = kind in (PLUS, MINUS, TIMES, DIV)

# Lexer
struct Token
    kind::TokenKind
    startbyte::UInt
    endbyte::UInt
    val::Union{Int, Void}
end

mutable struct Lexer
    io::IO
    current_char::Char
    charstore::IOBuffer
    token_start_pos::UInt
    current_row::Int
    current_col::Int
end
Lexer(io::IO) = Lexer(io, '\0', IOBuffer(), 0, 0, 0)
peekchar(l::Lexer) = peekchar(l.io)
dpeekchar(l::Lexer) = dpeekchar(l.io)

function emit(l::Lexer, kind::TokenKind)
    text = String(take!(l.charstore))
    val = kind == INT ? parse(Int, text) :
          kind == BACKREF ? parse(Int, text[2:end]) :
          nothing
    Token(kind, l.token_start_pos, position(l.io), val)
end

function readchar(l::Lexer)
    l.current_char = readchar(l.io)
    write(l.charstore, l.current_char)
    if l.current_char == '\n'
        l.current_row += 1
        l.current_col = 1
    elseif !eof(l.current_char)
        l.current_col += 1
    end
    return l.current_char
end

const operators = Dict(
    '+' => PLUS,
    '-' => MINUS,
    '*' => TIMES,
    '/' => DIV,
)

function accept(l::Lexer, f::Union{Function, Char, Vector{Char}, String})
    c = peekchar(l)
    if isa(f, Function)
        ok = f(c)
    elseif isa(f, Char)
        ok = c == f
    else
        ok = c in f
    end
    ok && readchar(l)
    return ok
end

function accept_number(l::Lexer, f::F) where {F}
    while true
        pc, ppc = dpeekchar(l)
        if pc == '_' && !f(ppc)
            return
        elseif f(pc) || pc == '_'
            readchar(l)
        else
            return
        end
    end
end

"""
accept_batch(l::Lexer, f)
Consumes all following characters until `accept(l, f)` is `false`.
"""
function accept_batch(l::Lexer, f)
    ok = false
    while accept(l, f)
        ok = true
    end
    return ok
end

function lex_digit(l::Lexer)
    accept_number(l, isdigit)
    emit(l, INT)
end

is_non_nl_whitespace(c) = c != '\n' && iswhitespace(c)

function lex_whitespace(l::Lexer)
    accept_batch(l, is_non_nl_whitespace)
    return emit(l, WHITESPACE)
end

eof(io::IO) = Base.eof(io)
eof(c::Char) = c === EOF_CHAR

function start_token!(l::Lexer)
    l.token_start_pos = position(l.io)
end

function Base.next(l::Lexer)
    start_token!(l)
    c = readchar(l)
    if eof(c)
        return emit(l, ENDMARKER)
    elseif c in keys(operators)
        return emit(l, operators[c])
    elseif c == '('
        return emit(l, LPAREN)
    elseif c == ')'
        return emit(l, RPAREN)
    elseif isdigit(c)
        return lex_digit(l)
    elseif is_non_nl_whitespace(c)
        return lex_whitespace(l)
    elseif c == '_'
        accept_number(l, isdigit)
        return emit(l, BACKREF)
    elseif c == '\n'
        return emit(l, NEWLINE)
    else
        return emit(l, ERROR)
    end
end
Base.start(l::Lexer) = nothing
Base.next(l::Lexer, state) = next(l), state
Base.done(l::Lexer, state) = eof(l.io)
Base.iteratorsize(::Type{Lexer}) = Base.SizeUnknown()

# CST Parser
struct CSTNode
    fullspan::UInt
    span::UnitRange{UInt}
end

abstract type CalcExpr; end

# Leaf nodes
struct Punctuation
    n::CSTNode
    kind::TokenKind
end
function Base.show(io::IO, punct::Punctuation)
    print(io, Punctuation, " ", punct.kind)
    print(io, "  ", punct.n.fullspan, " (", convert(UnitRange{Int}, punct.n.span), ")")
end
AbstractTrees.printnode(io::IO, lit::Punctuation) = show(io, lit)


struct Literal <: CalcExpr
    n::CSTNode
    value::Int64
end
function Base.show(io::IO, lit::Literal)
    print(io, Literal, "[", lit.value, "]")
    print(io, "  ", lit.n.fullspan, " (", convert(UnitRange{Int}, lit.n.span), ")")
end
AbstractTrees.printnode(io::IO, lit::Literal) = show(io, lit)

struct Backref <: CalcExpr
    n::CSTNode
    value::Int64
end
function Base.show(io::IO, br::Backref)
    print(io, Backref, "[_", br.value, "]")
    print(io, "  ", br.n.fullspan, " (", convert(UnitRange{Int}, br.n.span), ")")
end
AbstractTrees.printnode(io::IO, lit::Backref) = show(io, Backref)


function spansum(args...)
    fullspan = sum(arg->arg.n.fullspan, args)
    span = first(first(args).n.span):(
        fullspan - (last(args).n.fullspan -
                    last(last(args).n.span)))
    CSTNode(fullspan, span)
end

# Higher level nodes
struct Operator <: CalcExpr
    n::CSTNode
    lhs::CalcExpr
    kind::Punctuation
    rhs::CalcExpr
end
function Operator(lhs, kind, rhs)
    Operator(spansum(lhs, kind, rhs), lhs, kind, rhs)
end
AbstractTrees.children(o::Operator) = (o.lhs, o.kind, o.rhs)

struct ParenthesizedExpr <: CalcExpr
    n::CSTNode
    lparen::Punctuation
    expr::CalcExpr
    rparen::Punctuation
end
function ParenthesizedExpr(lparen, expr, rparen)
    ParenthesizedExpr(spansum(lparen, expr, rparen), lparen, expr, rparen)
end
AbstractTrees.children(o::ParenthesizedExpr) = (o.lparen, o.expr, o.rparen)

function AbstractTrees.printnode(io::IO, x::T) where T <: CalcExpr
    print(io, T, "  ", x.n.fullspan, " (", convert(UnitRange{Int}, x.n.span), ")")
end
function Base.show(io::IO, x::CalcExpr)
    if get(io, :compact, false)
        AbstractTrees.printnode(io, x)
    else
        AbstractTrees.print_tree(io, x, 3)
    end
end

mutable struct Parser
    l::Lexer
    first_leading_ws::Union{Token, Void}
    saw_newline::Bool
    current_token::Token
    had_error::Bool
end

function Parser(io::IO)
    l = Lexer(io)
    ws = nothing
    t = next(l)
    if t.kind == WHITESPACE || t.kind == NEWLINE
        ws = t
        while t.kind == WHITESPACE || t.kind == NEWLINE
            t = next(l)
        end
    end
    return Parser(l, ws, false, t, false)
end
tok(p::Parser) = p.current_token

macro propagate_error(p, x)
    quote
        x = $(esc(x))
        if $(esc(p)).had_error
            return nothing
        end
        x
    end
end

function emit_error(p::Parser, text)
    p.had_error = true
    return nothing
end

function INSTANCE(p::Parser)
    t = tok(p)
    fullspan = 0
    if p.first_leading_ws != nothing
        fullspan += t.startbyte - p.leading_ws.startbyte
    end
    p.first_leading_ws = nothing
    p.saw_newline = false
    span_start = fullspan
    # Gobble up whitespace
    nt = next(p.l)
    while nt.kind == WHITESPACE
        nt = next(p.l)
    end
    fullspan = nt.startbyte - t.endbyte
    # If the next token is a newline, add it to our trailing whitespace
    if nt.kind == NEWLINE
        p.saw_newline = true
        fullspan += 1
        nt = next(p.l)
        # Accumulate any leading whitespace now and get us to the first non-ws token
        if nt.kind == NEWLINE || nt.kind == WHITESPACE
            p.first_leading_ws = nt
            while nt.kind == NEWLINE || nt.kind == WHITESPACE
                nt = next(p.l)
            end
        end
    end
    p.current_token = nt
    t_span = t.endbyte - t.startbyte
    node = CSTNode(fullspan + t_span, span_start + (1:t_span))
    if t.kind == INT
        return Literal(node, t.val)
    elseif t.kind == BACKREF
        return Backref(node, t.val)
    elseif t.kind in (LPAREN, RPAREN) || is_op_kind(t.kind)
        return Punctuation(node, t.kind)
    else
        error("Unrecognized leaf token")
    end
end

function parse_operand(p::Parser)
    t = tok(p)
    if t.kind == LPAREN
        lparen = INSTANCE(p)
        expr = @propagate_error p parse_expr(p)
        rparen = INSTANCE(p)
        return ParenthesizedExpr(lparen, expr, rparen)
    elseif t.kind == INT || t.kind == BACKREF
        return INSTANCE(p)
    else
        return emit_error(p, "Any operand must always be a parenthesized expression, a backref or a literal")
    end
end

function parse_expr(p::Parser)
    lhs = @propagate_error p parse_operand(p)
    t = tok(p)
    if p.saw_newline || tok(p).kind == ENDMARKER
        return lhs
    end
    if !is_op_kind(t.kind)
        return emit_error(p, "Invalid non-operator after LHS expression")
    end
    op = INSTANCE(p)
    rhs = @propagate_error p parse_operand(p)
    return Operator(lhs, op, rhs)
end

function parse_toplevel(p::Parser)
    expr = @propagate_error p parse_expr(p)
    if !(p.saw_newline || tok(p) == ENDMARKER)
        return emit_error(p, "Extra token after the end of an expression")
    end
    expr
end

function parseall(data::String)
    p = Parser(IOBuffer(data))
    code = CalcExpr[]
    while !p.had_error && tok(p).kind != ENDMARKER
        expr = @propagate_error p parse_toplevel(p)
        push!(code, expr)
    end
    ParsedSourceCode(data, code)
end

# Interpreter
struct ParsedSourceCode
   text::String
   parsed::Vector{CalcExpr}
end

mutable struct InterpreterState
    backrefs::Vector{Any}
    code::ParsedSourceCode
    valstack::Vector{Any}
    it::Any
    pc::Int
end

function interesting_node_iterator(ast)
    Iterators.filter(x->!(x isa Calc.Punctuation || x isa Calc.ParenthesizedExpr), PostOrderDFS(ast))
end

function enter(code::ParsedSourceCode)
    it = interesting_node_iterator(code.parsed[1])
    InterpreterState(
        Any[],
        code,
        Any[],
        (it, start(it),),
        1
    )
end

const op_to_func_map = Dict(
    PLUS => +,
    MINUS => -,
    TIMES => *,
    DIV => /,
)

function _step!(s::InterpreterState)
    it, state = s.it
    expr, ns = next(it, deepcopy(state))
    if expr isa Literal
        push!(s.valstack, expr.value)
    elseif expr isa Backref
        push!(s.valstack, s.backrefs[expr.value])
    elseif expr isa Operator
        op2 = pop!(s.valstack)
        op1 = pop!(s.valstack)
        push!(s.valstack, op_to_func_map[expr.kind.kind](op1, op2))
    else
        error("Unrecognized expression")
    end
    s.it = (it, ns)
end

function step!(s::InterpreterState)
    it, state = s.it
    if !done(it, state)
        _step!(s)
    end
    it, state = s.it
    if done(it, state)
        if s.pc == length(s.code.parsed)
            return false
        end
        s.pc = s.pc + 1
        it = interesting_node_iterator(s.code.parsed[s.pc])
        s.it = (it, start(it))
        push!(s.backrefs, s.valstack[end])
        empty!(s.valstack)
    end
    return true
end

# Debugger
function skip_literals!(s::InterpreterState)
    while true
        it, state = s.it
        expr, ns = next(it, deepcopy(state))
        if !(expr isa Literal || expr isa Backref)
            break
        end
        step!(s)
    end
end

function DebuggerFramework.debug(code::ParsedSourceCode, args...)
    s = enter(code)
    skip_literals!(s)
    DebuggerFramework.RunDebugger([s], args...)
end

DebuggerFramework.locdesc(s::InterpreterState) = "statement $(s.pc)"
function DebuggerFramework.locinfo(s::InterpreterState)
    it, state = s.it
    pc_range = 1:(done(it, state) ? s.pc : s.pc - 1)
    offset = isempty(pc_range) ? 0 : sum(i->s.code.parsed[i].n.fullspan, pc_range)
    DebuggerFramework.BufferLocInfo(s.code.text, DebuggerFramework.compute_line(
        DebuggerFramework.SourceFile(s.code.text), offset), 0, 1)
end

function DebuggerFramework.print_next_state(io::IO, state, s::InterpreterState)
    it, state = s.it
    if done(it, state)
        if s.pc == length(s.code.parsed)
            return false
        end
        expr = first(interesting_node_iterator(s.code.parsed[s.pc + 1]))
    else
        expr, _ = next(it, deepcopy(state))
    end
    print(io, "About to run: ")
    if expr isa Literal
        print(io, expr.value)
    elseif expr isa Backref
        print(io, "_", expr.value)
    elseif expr isa Operator
        print(io, s.valstack[end-1], " ", op_to_func_map[expr.kind.kind], " ", s.valstack[end])
    end
    println(io)
    return true
end

function DebuggerFramework.execute_command(state, s::InterpreterState, cmd::Union{Val{:s},Val{:si}}, command)
    if cmd == Val{:si}()
        if !step!(s)
            shift!(state.stack)
            return false
        end
        return true
    elseif cmd == Val{:s}()
        if !step!(s)
            shift!(state.stack)
            return false
        end
        skip_literals!(s)
        return true
    end
    return false
end

function DebuggerFramework.eval_code(state, s::InterpreterState, command)
    p = Parser(IOBuffer(command))
    expr = parse_expr(p)
    ns = enter(ParsedSourceCode(command, CalcExpr[expr]))
    ns.backrefs = copy(s.backrefs)
    while step!(ns); end
    ns.valstack[end]
end

using Base: REPL, LineEdit
function DebuggerFramework.language_specific_prompt(state, frame::InterpreterState)
    if haskey(state.language_modes, :calc)
        return state.language_modes[:calc]
    end
    calc_prompt = LineEdit.Prompt(DebuggerFramework.promptname(state.level, "calc");
        # Copy colors from the prompt object
        prompt_prefix = state.repl.prompt_color,
        prompt_suffix = (state.repl.envcolors ? Base.input_color : repl.input_color),
        on_enter = Base.REPL.return_callback)
    calc_prompt.hist = state.main_mode.hist
    calc_prompt.hist.mode_mapping[:calc] = calc_prompt

    calc_prompt.on_done = (s,buf,ok)->begin
        if !ok
            LineEdit.transition(s, :abort)
            return false
        end
        xbuf = copy(buf)
        command = String(take!(buf))
        ok, result = DebuggerFramework.eval_code(state, command)
        Base.REPL.print_response(state.repl, ok ? result : result[1], ok ? nothing : result[2], true, true)
        println(state.repl.t)
        LineEdit.reset_state(s)
    end
    calc_prompt.keymap_dict = LineEdit.keymap([Base.REPL.mode_keymap(state.main_mode);state.standard_keymap])
    state.language_modes[:calc] = calc_prompt
    return calc_prompt
end

end