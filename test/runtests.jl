using DebuggerFramework
using Test
using REPL
using TerminalRegressionTests

include("languages/brainstack.jl")
include("languages/calc.jl")

function setup_repl(emuterm)
    repl = REPL.LineEditREPL(emuterm, true)
    repl.interface = REPL.setup_interface(repl)
    repl.specialdisplay = REPL.REPLDisplay(repl)
    repl
end

const thisdir = dirname(@__FILE__)

# Test stepping in/out of frames
# Language: BrainStack
proga = """
1: 2 3 0
2: 0
3: 0
"""

asta = parse(BrainStack.BrainStackAST, proga)
@test BrainStack.interpret(asta) == 2

TerminalRegressionTests.automated_test(
                joinpath(thisdir,"brainstack/simple.multiout"),
               ["si\n", "f 2\n", "si\n", "si\n", "\n", "\n"]) do emuterm
    DebuggerFramework.debug(asta, setup_repl(emuterm), emuterm)
end

# Test test entering language specific prompt
# Language: Calc
prog = """
1 + (9 * (7 / 2))
_1 * 2
_1 + _2
_1 / 3
"""
calc_code = Calc.parseall(prog)

TerminalRegressionTests.automated_test(
        joinpath(thisdir,"calc/lsrp.multiout"),
        [
        # Evaluate the first three expressions
        "s\ns\ns\n",
        # Look at the first backref
        "`_1\n\b",
        # Make sure we can type ` without erroring (#2)
        "abc`\b\b\b\b`_1\n",
        # Exit
        "\x4"]) do emuterm
    DebuggerFramework.debug(calc_code, setup_repl(emuterm), emuterm)
end
