using Base.Test
using DebuggerFramework

# This file will define a minimalistic programming language and implement a
# debugger for this language. The language is not supposed to be useful, it
# is a proving ground for the DebuggerFramework APIs.

# Language semantics:
# Every line has to start with its line number relative to the current program,
# followed by a colon. The colon is followed by a whitespace separated list of
# integers, indicating either a line to call, or 0 which indicates a return.
# This language has no conditional branches, so all loops are infinite. The
# computed value is the number of calls that were made

module BrainStack
    using DebuggerFramework
    struct BrainStackAST
        stmts::Vector{Vector{Int}}
    end

    function Base.parse(::Type{BrainStackAST}, data)
        BrainStackAST(map(enumerate(split(data, '\n', keep=false))) do args
            lineno, line = args
            parts = split(line, r"\s", keep=false)
            @assert parts[1] == "$lineno:"
            map(x->parse(Int,x), parts[2:end])
        end)
    end

    mutable struct BrainStackInterpreterState
        ast::BrainStackAST
        stack::Vector{Tuple{Int, Int}}
        idx::Int
        pos::Int
        ncalls::Int
    end

    # Helpfully split up the interpreter to be re-usable from the debugger below
    enter(ast::BrainStackAST) = BrainStackInterpreterState(ast, [(0,0)], 1, 1, 0)
    function step_one!(state::BrainStackInterpreterState)
        if state.idx == 0
            return false
        end
        op = state.ast.stmts[state.idx][state.pos]
        if op == 0
            state.idx, state.pos = pop!(state.stack)
            state.pos += 1
        else
            push!(state.stack, (state.idx, state.pos))
            state.idx, state.pos = op, 1
            state.ncalls += 1
        end
        return true
    end
    function interpret(ast::BrainStackAST)
        state = enter(ast)
        while step_one!(state); end
        state.ncalls
    end


    # Debugger support
    struct BrainStackStackRef <: DebuggerFramework.StackFrame
        state::BrainStackInterpreterState
        idx::Int
    end

    function DebuggerFramework.debug(ast::BrainStackAST, args...)
        state = enter(ast)
        stack = [BrainStackStackRef(state, 0)]
        DebuggerFramework.RunDebugger(stack, args...)
    end

    function idxpos(frame)
        if frame.idx == 0
            idx, pos = frame.state.idx, frame.state.pos
        else
            idx, pos = frame.state.stack[end-(frame.idx-1)]
        end
        idx, pos
    end

    function DebuggerFramework.print_status_synthtic(io::IO, state, frame::BrainStackStackRef, lines_before, total_lines)
        ast = frame.state.ast
        idx, pos = idxpos(frame)
        print(io, "$idx: ")
        for (opno, op) in enumerate(ast.stmts[idx])
            if opno == pos
                print_with_color(:yellow, io, string(op); bold = true)
            else
                print(io, op)
            end
            (opno != length(ast.stmts[idx])) && print(io, ' ')
       end
       return 1
    end

    DebuggerFramework.locdesc(frame::BrainStackStackRef) = "statement $(idxpos(frame)[1])"

    function update_stack!(ds, state, stack)
        ds.stack = [BrainStackStackRef(state, i) for i = 0:length(stack)-1]
        if ds.level > length(ds.stack)
            ds.level = length(ds.stack)
        end
    end

    function DebuggerFramework.execute_command(state, frame::BrainStackStackRef, cmd::Val{:si}, command)
        step_one!(frame.state)
        update_stack!(state, frame.state, frame.state.stack)
        return true
    end
end